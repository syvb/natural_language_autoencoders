# Input-overlap analysis: how much does the NLA repeat its input?

16 held-out samples; generations scored against their FULL input context (mean over samples; word-level, punctuation-free).
`shuffled-ctrl` = pre-RL generations vs a different sample's context (chance/topicality floor). `gold` = Sonnet-4.6 explanations.

| condition | ngram1 | ngram2 | ngram3 | ngram4 | ngram5 | coverage | density | lcs | final_echo |
|---|---|---|---|---|---|---|---|---|---|
| iter50 | 0.370 | 0.041 | 0.006 | 0.002 | 0.000 | 0.370 | 0.466 | 2.3 | 0.938 |
| preRL | 0.348 | 0.042 | 0.007 | 0.001 | 0.000 | 0.348 | 0.445 | 2.4 | 0.938 |
| gold | 0.232 | 0.039 | 0.014 | 0.005 | 0.001 | 0.232 | 0.351 | 3.1 | 1.000 |
| shuffled-ctrl | 0.216 | 0.007 | 0.000 | 0.000 | 0.000 | 0.216 | 0.229 | 1.4 | 0.188 |

Per-sample longest copied token run (lcs), preRL vs iter50:
```
sample  preRL  iter50  gold
     1      4       2     3
     2      3       2     4
     3      1       1     3
     4      2       2     2
     5      2       2     2
     6      3       3     3
     7      2       2     5
     8      2       2     2
     9      2       4     2
    10      3       2     1
    11      2       2     3
    12      2       3     2
    13      2       4     5
    14      4       3     4
    15      2       1     4
    16      3       2     4
```

## Input-tail echo (contiguous reproduction of the input's last k words)

| k | preRL | iter50 |
|---|---|---|
| 1 | 15/16 | 15/16 |
| 2 | 9/16 | 9/16 |
| 3 | 5/16 | 4/16 |
| 4 | 2/16 | 2/16 |
| 5 | 0/16 | 0/16 |

## Interpretation

The AV sees ONLY the injected activation — it has no access to the input text —
so it cannot literally quote; it can only reproduce what the activation encodes.
And that is exactly what the numbers show: verbatim overlap is a SHORT SUFFIX
ECHO (last word 94%, decaying to zero by 5 words) plus chance-level phrase reuse.
Trigram+ precision (~0.6%) is BELOW the gold Sonnet explanations (~1.4%) — gold
was written WITH the input text in context and quotes it for real. The pre-RL
model's heavy "quoting" is quote-marked CONFABULATION: pastiche that looks like
citation but doesn't match the input beyond ~2-4 words. The quote penalty
changed none of this (preRL ≈ iter50 on every content metric) — it removed the
punctuation, not the (already minimal) repetition.
