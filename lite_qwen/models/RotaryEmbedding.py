import torch, math
import torch.nn as nn
from typing import Optional, Tuple
import logging
from .model_config import Qwen2Config

logger = logging.getLogger(__name__)

def _compute_default_rope_parameters(
    config = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple["torch.Tensor", float]:
    """
    根据原始 RoPE 实现计算逆频率。
    
    参数:
        config (`~transformers.LlamaConfig` 可选):
            模型的配置。
        device (`torch.device` 可选):
            用于初始化逆频率的设备。
        seq_len (`int` 可选):
            当前序列长度。对于此类型的RoPE未使用。
        rope_kwargs (`Dict` 可选):
            向后兼容参数, 将在v4.45中移除。
    
    返回:
        一个元组, 包含RoPE嵌入的逆频率 (`torch.Tensor`), 形状为 [head_dim//2] 和应用于cos/sin的后处理缩放因子 (`float`)。
    """
    # 限制单个参数传递的方式，要么用规范的config，要么用旧的
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    # 尝试从 rope_kwargs 中提取 base 和 dim
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    # 否则，从 config 中提取参数
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_heads)
        
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # 注意力缩放因子，当前类型的RoPE未使用
    
    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor

ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
}

# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2Config] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning(
                "`Qwen2RotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        # cos = cos.view(batch_size * seq_len, -1)
        # sin = sin.view(batch_size * seq_len, -1)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


import unittest
import torch

class TestQwen2RotaryEmbedding(unittest.TestCase):
    def setUp(self):
        self.config = Qwen2Config()
        self.config.rope_theta = 10000
        # self.config.partial_rotary_factor = 1.0
        self.config.head_dim = 16
        self.config.hidden_size = 128
        self.config.num_heads = 8
        self.config.rope_scaling = {
            "factor": 8,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
            "original_max_position_embeddings": 100,  # 设置较小的 original_max_position_embeddings
            "rope_type": "default"
        }
        self.config.max_position_embeddings = 100

    def test_default_rope_parameters(self):
        rotary_emb = Qwen2RotaryEmbedding(
            dim=4,  # head_dim * partial_rotary_factor = 2 * 1 = 2, dim=2 * 2=4 (因步长=2)
            max_position_embeddings=50,
            base=10000,
            device=torch.device("cpu"),
            scaling_factor=1.0,
            rope_type="default",
            config=None
        )
        inv_freq, attention_scaling = rotary_emb.rope_init_fn(None, torch.device("cpu"), **rotary_emb.rope_kwargs)
        self.assertEqual(inv_freq.shape[0], self.config.head_dim // 2)  # dim=4, step=2 -> 2
        self.assertEqual(attention_scaling, 1.0)

    def test_forward_output_shape(self):
        # 测试前向传播的输出形状
        rotary_emb = Qwen2RotaryEmbedding(
            config=self.config,
            device=torch.device("cpu"),
        )
        # prefill 阶段
        """
        position_ids,  tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        """
        batch_size = 2
        seq_length = 10
        x = torch.randn(batch_size, seq_length, self.config.hidden_size)
        position_ids = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, seq_length)
        cos, sin = rotary_emb(x, position_ids)
        self.assertEqual(cos.shape, (batch_size, seq_length, self.config.head_dim))
        self.assertEqual(sin.shape, (batch_size, seq_length, self.config.head_dim))
if __name__ == '__main__':
    unittest.main()