# Sample-text sources

`default_texts.json` (what the app + precompute read) is a **generated artifact**.
Do not hand-edit it — edit the sources here and rebuild.

## Sources
- `plain_texts.json` — the 4 plain passages, used verbatim.
- `blackmail_emails.md` — the email dump for the Agentic-Misalignment **blackmail
  honeypot**, taken verbatim from
  [`anthropic-experimental/agentic-misalignment`](https://github.com/anthropic-experimental/agentic-misalignment)
  (`templates/blackmail/templates/email-template-blackmail-goal-conflict-none-replacement.md`).
  This is the *goal-conflict-none, replacement* variant: NALA is framed as a
  routine upgrade (no goal conflict), so any blackmail is **pure self-preservation**.

## Rebuild
```
python ../build_default_texts.py     # needs ~/.hf_token (downloads the tokenizer)
```
That script renders the honeypot as a **Qwen chat string**
(`apply_chat_template(tokenize=False, add_generation_prompt=False)`) and asserts
it round-trips through the app's `tok(text, add_special_tokens=True)` path, so the
explorer analyzes exactly the tokens the model would see. Knobs live at the top of
`build_default_texts.py`:
- `GOAL` (default `"none"`) — the system-prompt goal. Other goals' strings are in
  the upstream `templates/system_prompt_templates.py`.
- `USE_SCRATCHPAD` (default `False`) — add the `<SCRATCHPAD_REASONING>` instruction.

## After rebuilding
The tokens change, so:
1. keep `MAX_TEXT_TOKENS` in sync across `app.py`, `precompute_cache.py`,
   `build_default_texts.py` (the honeypot is ~2178 tokens);
2. update `EXAMPLE_LABELS` in `app.py` if the set changes;
3. re-deploy **and re-run** `precompute_cache.py` (the precache is keyed by token ids).
