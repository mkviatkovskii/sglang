"""
Unit tests for NativeSparseAttnBackend._forward_triton.

Tests the triton-based sparse MLA attention kernel against a float32 reference
implementation.  On Hopper/SM90+ hardware, also cross-checks against
_forward_flashmla_sparse and _forward_fa3 to make sure all three backends
agree (within bfloat16 tolerance).

Run with:
    python test/manual/layers/attention/nsa/test_nsa_triton_backend.py
"""

import math
import unittest

import torch

from sglang.srt.layers import dp_attention as _dp_attn

# Patch DP-attention globals before importing backends (same pattern as test_nsa_indexer.py)
_dp_attn.get_attention_tp_size = lambda: 1

from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

# DeepSeek MLA dimensions
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK  # 512
NUM_Q_HEADS = 128
NSA_TOPK = 64
PAGE_SIZE = 1
MAX_CONTEXT_LEN = 4096
MAX_BS = 32


def _make_server_args(prefill="triton", decode="triton"):
    args = ServerArgs(model_path="dummy")
    args.enable_dp_attention = False
    args.nsa_prefill_backend = prefill
    args.nsa_decode_backend = decode
    return args


def _make_mock_model_runner(nsa_decode="triton", nsa_prefill="triton"):
    """Build a minimal mock model runner that NativeSparseAttnBackend accepts."""
    device = "cuda"
    total_slots = MAX_BS * MAX_CONTEXT_LEN

    hf_config = type(
        "HfConfig",
        (),
        {
            "architectures": ["DeepseekV3ForCausalLM"],
            "index_topk": NSA_TOPK,
            "index_head_dim": 128,
            "index_n_heads": 32,
        },
    )()

    model_config = type(
        "ModelConfig",
        (),
        {
            "context_len": MAX_CONTEXT_LEN,
            "attention_arch": None,
            "num_attention_heads": NUM_Q_HEADS,
            "kv_lora_rank": KV_LORA_RANK,
            "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
            "qk_nope_head_dim": KV_LORA_RANK,
            "hf_config": hf_config,
        },
    )()

    req_to_token_pool = type(
        "TokenPool",
        (),
        {
            "size": MAX_BS,
            "req_to_token": torch.zeros(
                MAX_BS, MAX_CONTEXT_LEN, dtype=torch.int32, device=device
            ),
        },
    )()

    # Use MLATokenToKVPool with use_nsa=True to get the same layout as in production
    token_to_kv_pool = MLATokenToKVPool(
        size=total_slots,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        use_nsa=True,
    )

    server_args = type(
        "ServerArgs",
        (),
        {
            "kv_cache_dtype": "auto",
            "speculative_eagle_topk": None,
            "speculative_num_draft_tokens": 0,
            "enable_deterministic_inference": False,
            "nsa_prefill_backend": nsa_prefill,
            "nsa_decode_backend": nsa_decode,
            "triton_attention_num_kv_splits": 8,
        },
    )()

    return type(
        "ModelRunner",
        (),
        {
            "device": device,
            "page_size": PAGE_SIZE,
            "model_config": model_config,
            "req_to_token_pool": req_to_token_pool,
            "token_to_kv_pool": token_to_kv_pool,
            "server_args": server_args,
            "kv_cache_dtype": "auto",
        },
    )()


def _reference_sparse_mla(q_all, kv_cache, page_table_1, nsa_seqlens, sm_scale):
    """
    Float32 reference: for each query token i, gather the nsa_seqlens[i]
    selected KV slots from kv_cache and run full softmax attention.

    q_all:        [total_q, num_heads, 576]
    kv_cache:     [total_slots, 1, 576]  (bfloat16)
    page_table_1: [total_q, topk]  int32 absolute KV slot indices
    nsa_seqlens:  [total_q]  int32
    Returns:      [total_q, num_heads, 512]  float32
    """
    total_q, num_heads, _ = q_all.shape
    v_head_dim = KV_LORA_RANK
    q_f32 = q_all.float()
    kv_f32 = kv_cache.float()
    ref_o = torch.zeros(total_q, num_heads, v_head_dim, dtype=torch.float32, device=q_all.device)
    for i in range(total_q):
        n = nsa_seqlens[i].item()
        slots = page_table_1[i, :n]
        k_sel = kv_f32[slots, 0, :]              # [n, 576]
        v_sel = kv_f32[slots, 0, :v_head_dim]    # [n, 512]
        scores = torch.einsum("hd,kd->hk", q_f32[i], k_sel) * sm_scale
        ref_o[i] = torch.einsum("hk,kd->hd", torch.softmax(scores, dim=-1), v_sel)
    return ref_o


def _assert_close(label, actual, expected, atol_max=0.05, atol_mean=0.005):
    actual_f32 = actual.float()
    diff = (actual_f32 - expected).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    print(f"  {label}: max_diff={max_d:.6f}  mean_diff={mean_d:.6f}")
    assert max_d < atol_max, f"{label}: max_diff {max_d:.6f} >= {atol_max}"
    assert mean_d < atol_mean, f"{label}: mean_diff {mean_d:.6f} >= {atol_mean}"


@unittest.skipIf(not torch.cuda.is_available(), "Requires CUDA")
class TestNSATritonBackend(unittest.TestCase):
    """Tests for NativeSparseAttnBackend._forward_triton."""

    @classmethod
    def setUpClass(cls):
        server_args = _make_server_args()
        set_global_server_args_for_scheduler(server_args)
        cls.device = "cuda"
        cls.sm_major = torch.cuda.get_device_capability()[0]

    def setUp(self):
        torch.manual_seed(42)
        self.model_runner = _make_mock_model_runner()
        self.backend = NativeSparseAttnBackend(self.model_runner)
        # Fill kv_cache with random data
        kv_pool = self.model_runner.token_to_kv_pool
        kv_pool.kv_buffer[0].normal_()
        self.kv_cache = kv_pool.get_key_buffer(0)  # [total_slots+1, 1, 576]
        self.sm_scale = 1.0 / math.sqrt(KV_CACHE_DIM)

    def _run_triton(self, q_all, page_table_1, nsa_seqlens):
        return self.backend._forward_triton(
            q_all=q_all,
            kv_cache=self.kv_cache,
            page_table_1=page_table_1,
            nsa_cache_seqlens=nsa_seqlens,
            sm_scale=self.sm_scale,
            logit_cap=0.0,
            v_head_dim=V_HEAD_DIM,
        )

    def _make_sparse_inputs(self, total_q, topk, max_slot=None):
        """Create random q, page_table_1, and nsa_seqlens."""
        max_slot = max_slot or (self.kv_cache.shape[0] - 1)
        q_all = torch.randn(total_q, NUM_Q_HEADS, KV_CACHE_DIM,
                            dtype=torch.bfloat16, device=self.device)
        # Each query gets a random permutation of topk distinct KV slots
        page_table_1 = torch.stack([
            torch.randperm(max_slot, device=self.device)[:topk]
            for _ in range(total_q)
        ]).to(torch.int32)
        nsa_seqlens = torch.full((total_q,), topk, dtype=torch.int32, device=self.device)
        return q_all, page_table_1, nsa_seqlens

    # ------------------------------------------------------------------
    # Correctness tests: compare against float32 reference
    # ------------------------------------------------------------------

    def test_decode_full_topk(self):
        """Decode mode: all queries have exactly topk valid KV tokens."""
        print("\ntest_decode_full_topk")
        bs, topk = 4, 32
        q, pt, sl = self._make_sparse_inputs(bs, topk)
        ref = _reference_sparse_mla(q, self.kv_cache, pt, sl, self.sm_scale)
        out = self._run_triton(q, pt, sl)
        _assert_close("triton vs reference", out, ref)

    def test_decode_variable_seqlens(self):
        """Decode mode: queries have different numbers of valid KV tokens."""
        print("\ntest_decode_variable_seqlens")
        topk = 32
        bs = 6
        q, pt, _ = self._make_sparse_inputs(bs, topk)
        # Variable valid counts: some queries have fewer than topk tokens
        nsa_seqlens = torch.tensor([5, 32, 12, 32, 1, 20], dtype=torch.int32, device=self.device)
        # Mark out-of-range entries as -1 (as production code does)
        for i in range(bs):
            pt[i, nsa_seqlens[i]:] = -1

        ref = _reference_sparse_mla(q, self.kv_cache, pt, nsa_seqlens, self.sm_scale)
        out = self._run_triton(q, pt, nsa_seqlens)
        _assert_close("triton vs reference (variable seqlens)", out, ref)

    def test_extend_mode(self):
        """Extend/prefill: total_q = sum of extend lengths across sequences."""
        print("\ntest_extend_mode")
        total_q, topk = 24, 16
        q, pt, sl = self._make_sparse_inputs(total_q, topk)
        ref = _reference_sparse_mla(q, self.kv_cache, pt, sl, self.sm_scale)
        out = self._run_triton(q, pt, sl)
        _assert_close("triton vs reference (extend)", out, ref)

    def test_single_token_batch(self):
        """Edge case: bs=1."""
        print("\ntest_single_token_batch")
        q, pt, sl = self._make_sparse_inputs(total_q=1, topk=NSA_TOPK)
        ref = _reference_sparse_mla(q, self.kv_cache, pt, sl, self.sm_scale)
        out = self._run_triton(q, pt, sl)
        _assert_close("triton vs reference (bs=1)", out, ref)

    def test_logit_cap(self):
        """Softcap (logit_cap) as used by DeepSeek."""
        print("\ntest_logit_cap")
        logit_cap = 50.0
        bs, topk = 4, 16
        q, pt, sl = self._make_sparse_inputs(bs, topk)

        # Reference with softcap
        q_f32 = q.float()
        kv_f32 = self.kv_cache.float()
        ref = torch.zeros(bs, NUM_Q_HEADS, V_HEAD_DIM, dtype=torch.float32, device=self.device)
        for i in range(bs):
            n = sl[i].item()
            slots = pt[i, :n]
            k_sel = kv_f32[slots, 0, :]
            v_sel = kv_f32[slots, 0, :V_HEAD_DIM]
            scores = torch.einsum("hd,kd->hk", q_f32[i], k_sel) * self.sm_scale
            scores = logit_cap * torch.tanh(scores / logit_cap)
            ref[i] = torch.einsum("hk,kd->hd", torch.softmax(scores, dim=-1), v_sel)

        out = self.backend._forward_triton(
            q_all=q, kv_cache=self.kv_cache, page_table_1=pt,
            nsa_cache_seqlens=sl, sm_scale=self.sm_scale,
            logit_cap=logit_cap, v_head_dim=V_HEAD_DIM,
        )
        _assert_close("triton vs reference (logit_cap)", out, ref)

    # ------------------------------------------------------------------
    # Cross-backend comparison: triton vs flashmla_sparse / fa3
    # Only runs on SM90+ (Hopper) where both backends are available.
    # ------------------------------------------------------------------

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
        "flashmla_sparse requires SM90+",
    )
    def test_triton_vs_flashmla_sparse(self):
        """On Hopper: triton and flashmla_sparse must agree within bfloat16 tolerance."""
        print("\ntest_triton_vs_flashmla_sparse")
        bs, topk = 4, NSA_TOPK
        q, pt, sl = self._make_sparse_inputs(bs, topk)

        out_triton = self._run_triton(q, pt, sl)
        out_flashmla = self.backend._forward_flashmla_sparse(
            q_all=q,
            kv_cache=self.kv_cache,
            page_table_1=pt,
            sm_scale=self.sm_scale,
            v_head_dim=V_HEAD_DIM,
        )
        _assert_close("triton vs flashmla_sparse", out_triton, out_flashmla.float(),
                      atol_max=0.05, atol_mean=0.005)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
        "fa3 (flash_attn_with_kvcache with qv) requires SM90+",
    )
    def test_triton_vs_fa3(self):
        """On Hopper: triton and fa3 must agree within bfloat16 tolerance."""
        print("\ntest_triton_vs_fa3")
        bs, topk = 4, NSA_TOPK
        # fa3 uses nsa_cache_seqlens for cu_seqlens_k
        # Build cu_seqlens for fa3 interface
        from sglang.srt.layers.attention.nsa.nsa_backend_mtp_precompute import compute_cu_seqlens

        q, pt, sl = self._make_sparse_inputs(bs, topk)
        q_nope = q[:, :, :V_HEAD_DIM]  # [bs, num_heads, 512]
        q_rope = q[:, :, V_HEAD_DIM:]  # [bs, num_heads, 64]
        cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_k = compute_cu_seqlens(sl)

        out_triton = self._run_triton(q, pt, sl)
        out_fa3 = self.backend._forward_fa3(
            q_rope=q_rope,
            kv_cache=self.kv_cache,
            v_head_dim=V_HEAD_DIM,
            q_nope=q_nope,
            page_table=pt,
            cache_seqlens=sl,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            sm_scale=self.sm_scale,
            logit_cap=0.0,
            page_size=1,
        )
        _assert_close("triton vs fa3", out_triton, out_fa3.float(),
                      atol_max=0.05, atol_mean=0.005)


if __name__ == "__main__":
    unittest.main(verbosity=2)
