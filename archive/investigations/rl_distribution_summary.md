# RL Distribution Analysis: Honest Summary

## Key Numbers

| Metric | Base | QED-only | QED+MW | QED+MW+Div |
|--------|------|----------|--------|------------|
| QED mean | 0.742 | 0.798 | 0.818 | 0.837 |
| QED std  | 0.157 | 0.125 | 0.118 | 0.108 |
| MW mean  | 295.0 | 293.0 | 330.4 | 317.7 |
| MW std   | 38.1 | 28.0 | 24.5 | 26.1 |
| LogP mean| 2.325 | 2.691 | 2.664 | 2.711 |
| LogP std | 1.342 | 1.034 | 0.863 | 0.994 |

## Five-Sentence Summary

1. **Did QED-only produce a tighter MW distribution than base?** Yes: QED-only MW std = 28.0 Da vs. base 38.1 Da (narrower by 10.1 Da), suggesting that optimizing QED without an explicit MW constraint compressed the molecular weight distribution—likely because high-QED molecules tend to share similar structural profiles.

2. **Did QED-only produce a tighter QED distribution?** Yes: QED std dropped from 0.157 (base) to 0.125 (QED-only), indicating the RL policy learned to avoid both very-low- and very-high-QED molecules and concentrate around a narrow band.

3. **Is the QED compression "bad"?** Likely a mode-collapse signal: A tighter QED distribution is not intrinsically bad—it reflects the reward signal doing its job—but a moderate reduction in QED variance combined with also MW compression suggests the model may be converging to a limited set of scaffold types (mode narrowing), reducing structural diversity even if individual quality metrics improve.

4. **Did the MW constraint broaden the distribution or just shift the mean?** The QED+MW constraint did NOT substantially broaden the MW distribution (std: QED-only=28.0 → QED+MW=24.5 Da), meaning the constraint primarily shifted the mean (from 293.0 to 330.4 Da) without recovering the diversity lost during QED-only training.

5. **Strongest evidence-based motivation for MW constraint:** The QED-only policy compressed MW variance to 28.0 Da (down from 38.1 Da in base), concentrating molecules in a narrow band and reducing the fraction of drug-like-range molecules (200–400 Da); the MW penalty was therefore motivated not merely by mean drift but by the loss of distributional coverage across the drug-like weight range.
