"""Tiny random BitNet checkpoints for CPU tests."""

import torch
from transformers import BitNetConfig, BitNetForCausalLM

BITNET_QUANT = {"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": "online"}


def make_tiny_bitnet(path: str, dtype: torch.dtype = torch.bfloat16, seed: int = 0) -> BitNetConfig:
    """Save a 3-layer BitNet with latent weights at the real checkpoint's scale (mean|w| ~ 1.6)."""
    config = BitNetConfig(
        vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        tie_word_embeddings=True, rms_norm_eps=1e-5, quantization_config=BITNET_QUANT,
    )
    config.architectures = ["BitNetForCausalLM"]
    torch.manual_seed(seed)
    model = BitNetForCausalLM(config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("proj.weight"):
                param.normal_(0, 2.0)
            elif "norm" in name:
                param.uniform_(0.5, 1.5)
    model.to(dtype).save_pretrained(path)
    return config
