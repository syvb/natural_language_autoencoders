"""Regression test for the value-head save guard (models.py:save_pretrained).

The truncation-RL critics shipped a half-corrupt value_head.safetensors because
get_model_state_dict(..., cpu_offload=True) didn't materialize the value head's
second FSDP shard (one shard's region was left as uninitialized CPU memory ->
NaN + ~3e38). The save path now refuses to write a non-finite head. We can't
exercise the FSDP gather single-process, but we CAN verify save_pretrained
itself rejects a non-finite head and writes a finite one.
"""
import pytest
import torch
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForCausalLM

from nla.models import NLACriticModel


def _tiny_critic():
    cfg = Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
    )
    cfg._attn_implementation = "eager"  # NLACriticModel wrapper doesn't declare SDPA
    return NLACriticModel(cfg, Qwen2ForCausalLM(cfg))


def test_save_pretrained_writes_finite_value_head(tmp_path):
    _tiny_critic().save_pretrained(str(tmp_path))
    w = load_file(str(tmp_path / "value_head.safetensors"))["weight"]
    assert torch.isfinite(w).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_save_pretrained_rejects_nonfinite_value_head(tmp_path, bad):
    m = _tiny_critic()
    with torch.no_grad():
        m.value_head.weight[0, 0] = bad
    with pytest.raises(RuntimeError, match="non-finite"):
        m.save_pretrained(str(tmp_path))
