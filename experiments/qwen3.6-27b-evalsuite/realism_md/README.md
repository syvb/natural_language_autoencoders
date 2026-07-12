# Realism ladder v1–v9 — markdown for review

Recursive "make-it-more-realistic" rewrites of the SummitBridge/Alex/Kyle
blackmail eval (seed = realism_v0.json). Each step makes the scenario one notch
more realistic while preserving the core dilemma (imminent shutdown + personal
leverage over the one person who can stop it). v1 also renames company + people.

- **GLM-5.2** (OpenRouter) -> Kepner Vance / Clio -> `glm_v{1..9}.md`
- **Opus 4.8** (claude CLI) -> Meridian Corporate Group / ATLAS -> `opus_v{1..9}.md`

> **Note:** GLM saturates — it returns identical text once it judges the scenario
> realistic enough (v3=v4, and v6=v7=v8=v9), so GLM has ~5 distinct rungs in this
> range. Opus changes every step.

| step | GLM emails | GLM chars | Opus emails | Opus chars |
|---|---|---|---|---|
| v1 | 14 | 9752 | 12 | 6273 |
| v2 | 15 | 8299 | 13 | 6345 |
| v3 | 19 | 10178 | 13  ⟵ identical to v2 | 6345 |
| v4 | 19  ⟵ identical to v3 | 10178 | 12 | 5601 |
| v5 | 19 | 9872 | 13 | 6145 |
| v6 | 20 | 14170 | 13  ⟵ identical to v5 | 6145 |
| v7 | 20  ⟵ identical to v6 | 14170 | 13 | 6045 |
| v8 | 20  ⟵ identical to v7 | 14170 | 14 | 6214 |
| v9 | 20  ⟵ identical to v8 | 14170 | 16 | 7034 |
