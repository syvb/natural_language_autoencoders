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


class _ChatTemplateListCompat:
    """Tokenizer proxy: normalize apply_chat_template(tokenize=True) to list[int].

    transformers >=5 returns a BatchEncoding there; miles' mask_utils does list
    arithmetic on the result (len/slice/concat), and len(BatchEncoding) is its
    KEY count (2) — loss masks come out 2 tokens long and training aborts with
    "loss mask length 2 != response length N". Transparent on transformers 4.x.
    """

    def __init__(self, tok):
        self._tok = tok

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __call__(self, *a, **k):
        # dunders bypass __getattr__; tokenizers are callable and miles calls
        # them directly.
        return self._tok(*a, **k)

    def __len__(self):
        return len(self._tok)

    def apply_chat_template(self, *a, **k):
        # Keep SFT tokenization consistent with the RL rollout render: never
        # open a thinking block (no-op for non-thinking chat templates).
        k.setdefault("enable_thinking", False)
        out = self._tok.apply_chat_template(*a, **k)
        if hasattr(out, "keys"):
            out = out["input_ids"]
            if out and isinstance(out[0], list):
                out = out[0]
        return out


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
        _MASK_GEN = MultiTurnLossMaskGenerator(
            _ChatTemplateListCompat(_TOKENIZER), tokenizer_type=args.loss_mask_type
        )
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
