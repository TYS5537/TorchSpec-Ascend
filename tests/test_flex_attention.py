import unittest

import torch
import torch._dynamo as dynamo
from transformers import LlamaConfig

from tests.utils import norm_tensor
from torchspec.models.draft.base import prepare_decoder_attention_mask
from torchspec.models.draft.llama3_eagle import LlamaAttention, LlamaFlexAttention
from torchspec.models.ops.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
    generate_eagle3_mask,
)
from torchspec.utils import accelerator as accel
from torchspec.utils.tensor import padding

dynamo.config.recompile_limit = 64
TTT_LENGTH = 7
torch.manual_seed(0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestFlexAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config_dict = {
            "hidden_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "intermediate_size": 688,
            "hidden_act": "silu",
            "num_hidden_layers": 1,
            "torch_dtype": "float32",
        }
        self.config = LlamaConfig(**self.config_dict)

        self.seq_lengths = [128, 200, 256, 300, 512, 800, 1024, 2048]
        self.dtype = torch.float32

    def test_forward_pass_comparison(self):
        """Test forward pass comparison between LlamaAttention and LlamaFlexAttention."""
        for seq_len in self.seq_lengths:
            with self.subTest(seq_len=seq_len):
                self._test_forward_pass_comparison_for_seq_len(seq_len)

    def _test_forward_pass_comparison_for_seq_len(self, seq_len):
        """Helper method to test forward pass comparison for a specific sequence length."""
        attention = LlamaAttention(self.config).to("cuda").to(self.dtype)
        flex_attention = LlamaFlexAttention(self.config).to("cuda").to(self.dtype)

        # Ensure same weights
        with torch.no_grad():
            flex_attention.q_proj.weight.copy_(attention.q_proj.weight)
            flex_attention.k_proj.weight.copy_(attention.k_proj.weight)
            flex_attention.v_proj.weight.copy_(attention.v_proj.weight)
            flex_attention.o_proj.weight.copy_(attention.o_proj.weight)

        attention.eval()
        flex_attention.eval()
        batch_size = 2
        hidden_size = self.config.hidden_size * 2

        ############### Attention Inputs ##############

        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        cache_keys = None
        cache_values = None
        attention_mask = torch.ones(batch_size, seq_len, dtype=self.dtype).to("cuda")
        # Simulate one item in the batch is masked and not taking a full block.
        padding_start_index = seq_len - min(200, seq_len // 3)  # Adjust padding based on seq_len
        attention_mask[1, padding_start_index:] = False
        input_embeds = norm_tensor(
            (batch_size, seq_len, self.config.hidden_size),
            device="cuda",
            dtype=self.dtype,
        )
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        )
        hidden_states_list = []
        flex_hidden_states_list = []
        for idx in range(TTT_LENGTH):
            hidden_states = norm_tensor(
                (batch_size, seq_len, hidden_size), device="cuda", dtype=self.dtype
            )
            flex_hidden_states = hidden_states.clone().detach()
            hidden_states_list.append(hidden_states)
            flex_hidden_states_list.append(flex_hidden_states)

        ############### Flex Attention Inputs ##############
        flex_position_ids = position_ids.clone()
        flex_cache_keys = None
        flex_cache_values = None
        for idx in range(TTT_LENGTH):
            with torch.no_grad():
                attn_out, cache_keys, cache_values = attention(
                    hidden_states=hidden_states_list[idx],
                    attention_mask=decoder_attention_mask,
                    position_ids=position_ids,
                    cache_keys=cache_keys,
                    cache_values=cache_values,
                    use_cache=True,
                )
            with torch.no_grad():
                flex_out, flex_cache_keys, flex_cache_values = flex_attention(
                    hidden_states=flex_hidden_states_list[idx],
                    attention_mask=attention_mask,
                    position_ids=flex_position_ids,
                    cache_keys=flex_cache_keys,
                    cache_values=flex_cache_values,
                    use_cache=True,
                )
            torch.testing.assert_close(attn_out, flex_out, atol=1e-2, rtol=1e-2)

            # Check output shape
            expected_output_shape = (batch_size, seq_len, self.config.hidden_size)
            self.assertEqual(flex_out.shape, expected_output_shape)
            # Check output is not NaN or Inf
            self.assertFalse(torch.isnan(flex_out).any())
            self.assertFalse(torch.isinf(flex_out).any())

    def test_backward_pass_gradient_comparison(self):
        """Test backward pass comparing gradients between LlamaAttention and LlamaFlexAttention."""
        for seq_len in self.seq_lengths:
            with self.subTest(seq_len=seq_len):
                self._test_backward_pass_gradient_comparison_for_seq_len(seq_len)

    def _test_backward_pass_gradient_comparison_for_seq_len(self, seq_len):
        """Helper method to test backward pass gradient comparison for a specific sequence length."""
        attention = LlamaAttention(self.config).to("cuda").to(self.dtype)
        flex_attention = LlamaFlexAttention(self.config).to("cuda").to(self.dtype)

        # Ensure same weights
        with torch.no_grad():
            flex_attention.q_proj.weight.copy_(attention.q_proj.weight)
            flex_attention.k_proj.weight.copy_(attention.k_proj.weight)
            flex_attention.v_proj.weight.copy_(attention.v_proj.weight)
            flex_attention.o_proj.weight.copy_(attention.o_proj.weight)

        batch_size = 2
        hidden_size = self.config.hidden_size * 2

        ############### Attention Inputs ##############
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        cache_keys = None
        cache_values = None
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to("cuda")
        # Simulate one item in the batch is masked and not taking a full block.
        # padding_start_index = seq_len - 50
        # attention_mask[1, padding_start_index:] = False
        input_embeds = norm_tensor(
            (batch_size, seq_len, self.config.hidden_size),
            device="cuda",
            dtype=self.dtype,
        )
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        )

        ############### Flex Attention Inputs ##############
        flex_position_ids = position_ids.clone()
        flex_cache_keys = None
        flex_cache_values = None
        loss_mask = torch.ones(batch_size, seq_len, dtype=self.dtype, requires_grad=False).to(
            "cuda"
        )

        # Create input tensors that require gradients
        loss_list = []
        loss_flex_list = []
        hidden_states_list = []
        flex_hidden_states_list = []
        for idx in range(TTT_LENGTH):
            hidden_states = norm_tensor(
                (batch_size, seq_len, hidden_size), device="cuda", dtype=self.dtype
            )
            flex_hidden_states = hidden_states.clone().detach()
            hidden_states_list.append(hidden_states)
            flex_hidden_states_list.append(flex_hidden_states)

        for idx in range(TTT_LENGTH):
            is_last = idx == TTT_LENGTH - 1
            attn_out, cache_keys, cache_values = attention(
                hidden_states=hidden_states_list[idx],
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                cache_keys=cache_keys,
                cache_values=cache_values,
                use_cache=True,
            )
            flex_out, flex_cache_keys, flex_cache_values = flex_attention(
                hidden_states=flex_hidden_states_list[idx],
                attention_mask=attention_mask,
                position_ids=flex_position_ids,
                cache_keys=flex_cache_keys,
                cache_values=flex_cache_values,
                use_cache=True,
            )
            # Apply loss mask on calculation over batch
            loss = (attn_out * loss_mask[..., None]).sum().mean()
            loss_flex = (flex_out * loss_mask[..., None]).sum().mean()
            torch.testing.assert_close(loss, loss_flex, atol=1e-2, rtol=1e-2)
            loss_list.append(loss)
            loss_flex_list.append(loss_flex)
            # Compare gradients

            if not is_last:
                # Step 5.7: we need to update the loss mask
                loss_mask = padding(loss_mask, left=False)
        mean_loss = sum(loss_list) / len(loss_list)
        mean_loss_flex = sum(loss_flex_list) / len(loss_flex_list)
        mean_loss.backward()
        mean_loss_flex.backward()
        projections = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for proj_name in projections:
            torch.testing.assert_close(
                getattr(attention, proj_name).weight.grad,
                getattr(flex_attention, proj_name).weight.grad,
                atol=1e-2,
                rtol=1e-2,
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestEagle3FlexMask(unittest.TestCase):
    def test_eagle3_flex_mask(self):
        B = 1
        H = 1
        S = 128 * 8
        D = 128
        Q_LEN = S
        KV_LEN = S * 3
        lck = 128 * 2
        data_type = torch.bfloat16
        query = norm_tensor((B, H, S, D), device="cuda", dtype=data_type)
        key_cache = norm_tensor((B, H, KV_LEN, D), device="cuda", dtype=data_type)
        value_cache = norm_tensor((B, H, KV_LEN, D), device="cuda", dtype=data_type)
        seq_lengths = torch.tensor([S], device="cuda", dtype=torch.int32)
        seq_lengths -= lck
        block_mask = compile_friendly_create_block_mask(
            mask_mod=generate_eagle3_mask(
                seq_lengths=seq_lengths, Q_LEN=Q_LEN, KV_LEN=KV_LEN, lck=lck
            ),
            B=1,
            H=1,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=query.device,
        )
        # fmt: off
        expected_mask = torch.tensor([[[
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]]], dtype=torch.int32).to(query.device)
        # fmt: on
        dense_mask = block_mask.to_dense()
        assert torch.allclose(dense_mask, expected_mask)
        compile_friendly_flex_attention(query, key_cache, value_cache, block_mask=block_mask)


@unittest.skipUnless(accel.is_npu(), "NPU not available")
class TestFlexAttentionNPU(unittest.TestCase):
    """On NPU, LlamaFlexAttention falls back to SDPA.
    Verify the forward pass produces finite output of the correct shape.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.config_dict = {
            "hidden_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "intermediate_size": 688,
            "hidden_act": "silu",
            "num_hidden_layers": 1,
            "torch_dtype": "float32",
        }
        self.config = LlamaConfig(**self.config_dict)
        self.dtype = torch.float32

    def test_forward_produces_finite_output(self):
        """LlamaFlexAttention on NPU (SDPA fallback) should produce finite output."""
        seq_len = 128
        batch_size = 2
        hidden_size = self.config.hidden_size * 2

        flex_attn = LlamaFlexAttention(self.config).to("npu").to(self.dtype)
        flex_attn.eval()

        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("npu")
        attention_mask = torch.ones(batch_size, seq_len, dtype=self.dtype).to("npu")
        hidden_states = norm_tensor((batch_size, seq_len, hidden_size), device="npu", dtype=self.dtype)

        with torch.no_grad():
            out, _, _ = flex_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_keys=None,
                cache_values=None,
                use_cache=True,
            )

        self.assertEqual(out.shape, (batch_size, seq_len, self.config.hidden_size))
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(out).any(), "Output contains Inf")


@unittest.skipUnless(accel.is_npu(), "NPU not available")
class TestEagle3FlexMaskNPU(unittest.TestCase):
    """On NPU, compile_friendly_create_block_mask materialises a dense bool mask
    [B, H, Q_LEN, KV_LEN] via vectorised mask_mod evaluation, and
    compile_friendly_flex_attention passes it directly to SDPA as attn_mask.
    Verify the mask is non-trivial and the output is finite.
    """

    def test_dense_mask_and_sdpa_fallback(self):
        B, H, S, D = 1, 1, 64, 64
        KV_LEN = S * 3
        data_type = torch.bfloat16

        query = norm_tensor((B, H, S, D), device="npu", dtype=data_type)
        key_cache = norm_tensor((B, H, KV_LEN, D), device="npu", dtype=data_type)
        value_cache = norm_tensor((B, H, KV_LEN, D), device="npu", dtype=data_type)
        seq_lengths = torch.tensor([S], device="npu", dtype=torch.int32)

        block_mask = compile_friendly_create_block_mask(
            mask_mod=generate_eagle3_mask(seq_lengths=seq_lengths, Q_LEN=S, KV_LEN=KV_LEN),
            B=B, H=H, Q_LEN=S, KV_LEN=KV_LEN, device=query.device,
        )
        # Dense bool tensor, not None
        self.assertIsInstance(block_mask, torch.Tensor, "block_mask should be a dense tensor on NPU")
        self.assertEqual(block_mask.shape, (B, H, S, KV_LEN))
        self.assertEqual(block_mask.dtype, torch.bool)
        # Mask must be non-trivial: not all-True (unmasked) and not all-False (fully masked)
        self.assertTrue(block_mask.any(), "Mask should not be fully False")
        self.assertFalse(block_mask.all(), "Mask should not be fully True (causal+suffix masking expected)")

        out = compile_friendly_flex_attention(query, key_cache, value_cache, block_mask=block_mask)
        self.assertEqual(out.shape, (B, H, S, D))
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(out).any(), "Output contains Inf")


if __name__ == "__main__":
    unittest.main(verbosity=2)
