# Human review checklist

No item is automatically confirmed.

- [ ] Are input-only selected boxes representative enough, and are relevant objects visible? Record additional manual coordinates before examining output severity if more cases are needed.
- [ ] Do added endpoint pixels remove visible highlight texture, or occur in already featureless/light-source areas? Distinguish input-existing and newly saturated pixels; no paired GT is available.
- [ ] Does dark-region gain reveal structure, noise, or both? This experiment has no calibrated noise/detail ground truth.
- [ ] Do boundary crops show halos or smooth transitions? The contact sheet alone does not prove an artifact mechanism.
- [ ] Are normalized RGB changes visible as hue/chroma changes? Near-black ratios can be unstable; exact-black pixels are excluded and counts reported. Color difference is not a quality loss.
- [ ] Would floating-point, prequantization outputs change endpoint-transition conclusions? Current PNG analysis cannot identify actual preclamp >1 pixels individually.
- [ ] Are results consistent across all8 dev images rather than only the4 illustrated images? Regional JSON covers all8.
- [ ] Review limitations of the8-image development split and 1280x720 evaluation before any generalization statement.

H1 is not confirmed by this report. No theoretical defect or new loss is proposed.
