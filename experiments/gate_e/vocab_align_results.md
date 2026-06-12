# Vocabulary-geometry alignment of counterfactual deltas (Gate F pre-check)

cos(delta, M[src_kw] - M[tgt_kw]) over 9,228 single-token keyword pairs
(delta = h20(X_src) - h20(X_tgt); shuffled-pair null +0.0006 both channels):

| channel | mean | median | dist 0-2 | dist 3-14 | dist 15+ |
|---|---|---|---|---|---|
| embedding E    | +0.0100 | +0.0086 | +0.0162 | +0.0064 | +0.0012 |
| unembedding U  | +0.0205 | +0.0176 | +0.0210 | +0.0194 | +0.0214 |

Embedding channel = local lexical echo (decays with distance).
Unembedding channel = persistent, distance-FLAT encoding of the output-
distribution shift — the delta carries its behavioral effect in the
model's own output-vocabulary codebook wherever it is measured. This is
the mechanism for effect-targeted (promote/demote) labels generalizing
across edit types where E3's cause-labels learned type-specific codebooks.

## Gate F pre-checks 2+3 (probe-on-composite, cross-layer)

Probe trained on delta20 (61% top-1 clean, chance 2.8%):
- composite payload + natural layer-diff carrier (carrier 3.4x payload
  norm): top-1 0.444 / top-5 0.622; carrier x2: 0.339/0.519 — linear
  superposition holds; carrier mass barely intersects discriminative dirs.
- cross-layer: d20-probe on delta16 0.605 (native 0.624); on delta24
  0.551 (native 0.556) — content directions layer-stable across L16-L24.

All three Gate F mechanisms measured: effect channel (34x null, flat),
carrier robustness, layer invariance.
