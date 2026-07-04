"""Actor-SFT rollout: no generation — tokenize prompt+response, stash activation.

Pattern follows miles/rollout/sft_rollout.py. The data_buffer yields Samples
whose .prompt is a list[dict] (from NLADataSource, <INJECT>→㊗ already substituted)
and whose .metadata["response"] is the <explanation>...</explanation> string.
"""

import os

import torch

from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer

from nla.schema import MM_ACTIVATION_KEY


_TOKENIZER = None
_MASK_GEN = None
# v2 (item 5): when set, never put loss on the turn-terminating EOS / <|im_end|>
# so the actor is not taught to STOP — it generalizes to arbitrarily long output
# bounded only by the generation cap. Pair with item-mode truncation at RL time.
_NO_TRAIN_EOS = os.environ.get("NLA_NO_TRAIN_EOS") == "1"
_EOS_IDS: set[int] | None = None


def _resolve_eos_ids(tok) -> set[int]:
    ids: set[int] = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return ids


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    assert args.rollout_global_dataset

    global _TOKENIZER, _MASK_GEN, _EOS_IDS
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    if _MASK_GEN is None:
        _MASK_GEN = MultiTurnLossMaskGenerator(_TOKENIZER, tokenizer_type=args.loss_mask_type)
    if _EOS_IDS is None:
        _EOS_IDS = _resolve_eos_ids(_TOKENIZER)
        if _NO_TRAIN_EOS:
            print(f"[NLA] NLA_NO_TRAIN_EOS=1 — masking loss at EOS ids {sorted(_EOS_IDS)} "
                  f"(actor never trained to stop).", flush=True)

    samples = data_buffer.get_samples(args.rollout_batch_size)

    for group in samples:
        (sample,) = group
        messages = sample.prompt
        assert isinstance(messages, list), (
            f"actor SFT requires list[dict] prompt (got {type(messages).__name__}). "
            f"NLADataSource must use apply_chat_template=False."
        )
        response = sample.metadata["response"]
        messages = messages + [{"role": "assistant", "content": response}]

        token_ids, loss_mask = _MASK_GEN.get_loss_mask(messages)
        response_length = _MASK_GEN.get_response_lengths([loss_mask])[0]

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0.0
        sample.loss_mask = loss_mask[-response_length:]
        # Never-train-EOS: zero loss at the turn terminator(s) within the response
        # span. response_length (the span) is unchanged — only the mask is edited —
        # so the EOS token still provides context, it just receives no gradient.
        if _NO_TRAIN_EOS:
            resp_ids = token_ids[-response_length:]
            sample.loss_mask = [
                0 if tid in _EOS_IDS else m
                for tid, m in zip(resp_ids, sample.loss_mask)
            ]

        activation = torch.tensor(
            sample.metadata["activation_vector"], dtype=torch.float32
        ).view(1, -1)
        sample.multimodal_train_inputs = {MM_ACTIVATION_KEY: activation}

    return samples
