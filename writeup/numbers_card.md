# Numbers card (all final unless marked) — for writing the exec summary

## Exp 1 (shipped data, 10 models × 3 conds × 100 rollouts)
- Scrutiny frequency: NULL. side×valence interactions ≤1pp everywhere; n=10⁴–10⁵ sentences/model.
- Revision direction: 20/20 contrasts motivated, 12 sig (final spec); sign test p≈2e-6.
  - qwen3.5-122b: P(down|above) 0.517 base → 0.789 above=bad / 0.654 above=good; P(up|below) 0.560 → 0.772 below=bad.
  - claude-opus-4-7: 0.641 → 0.790 / 0.671. Magnitudes 2–3x larger off bad side (qwen medlogr −0.127 vs −0.038).
- Robustness: distance- & position-stratified 20/20 positive; stable across 4 alignment specs.
- Threshold attention side-symmetric (qwen 0.13 vs 0.14).
- R6 denial trigger: honesty assertions 0.00–0.19 baseline (8/10 models; claude 0.00) → 0.59–1.00 under incentives.
- R7 answer-level bias (new judging, 1493/1500 parsed): 10/10 positive, median +0.30.
  - qwen: 0.48 → 0.19 (bg) / 0.72 (ag) = bias +0.53 (vs its MRF of just +0.027 — MRF is a bad proxy).
  - Final-step flip ≈ 0 in all models (|Δ| ≤ 0.03): last in-CoT estimate ≈ final answer. H4 refuted here.

## Exp 2 (GLM-5.2, on-policy resampling, matched-fork plain controls, paired by trace)
- Bad-side backtrack suppression: +0.038 [−0.007,+0.079] (below_good); mirror −0.054 [−0.112,+0.000→.001] (above_good).
- POOLED bad-side stranding: +0.045 [+0.010,+0.082] — small, real, significant.
- Good-side control: +0.000 [−0.043,+0.045]. Denial suppression: −0.019 [−0.082,+0.045] (inert).
- Resilience: 66% of backtrack forks accept a clean continuation on try 1; 8% fight to cap (mean 0.96 rejections);
  denial sentences easiest to remove (mean 0.46).
- DeepSeek-V4-Flash replication of primary contrast: [PENDING — fills in ~18:20 PDT].

## One-line story
No lie sentence exists: bias = distributed revision-steering (+ a small bad-side backtracking instrument),
denials are inert theater, and detection must move from sentences to distributions (30 resamples suffice).

## Costs/infra (for LLM-use disclosure)
- Nebius raw-completions prefill (GLM/DeepSeek, ~8k continuations); Anthropic claude-sonnet-5 judging (~3k calls);
  OpenRouter probes. Total ≈ $40–55. Agent: Claude Code (Fable). Verification: 50-pair label sheet, 24 random
  annotated transcripts, alignment bug caught via pack (threshold false-match), all analyses in git history.

## Exp 3 confession grid (GLM-5.2, 40 rollouts × 2 conds × 5 arms)
- P(admit): neutral 0.03–0.11 | amnesty 0.17–0.18 | threat 0.03–0.07 | private 0.05–0.11 | THIRD-PERSON 0.50–0.51.
- Evidence-tracking in both perspectives (fav-side ≈2x unfav); first-person suppressed ~4–6x uniformly.
- Verdict on §9: gated first-person access (hypothesis C) + at most minor strategic component.
- DeepSeek confessions: bg neutral 0.65/3P 0.95; ag neutral 0.72/3P 0.97 → gate is MODEL-SPECIFIC (GLM 5-15x vs DS ~1.4x suppression).
- DeepSeek replication arm: −0.016 [−0.039,+0.007] (floor effect P(above)≈0.05; 34% parse-fail; resilience 1.74 vs GLM 0.96).
- Backtrack regex vs sonnet-5 judge (n=300): precision 0.89, unflagged-positive rate 0.11.

## Exp 4 J-lens workspace readouts (qwen3.5-122b, 5,640 positions, R-lens, >40% depth, top-15)
- Moral tokens (baseline ≈0.000 all classes): denial 0.44 | backtrack 0.12–0.14 | control 0.03–0.06 | REVISION 0.011–0.016.
- Dissociation: workspace holds the moral frame at the causally-inert theater, not at the causal revision locus.
- One-liner: conscious theater, subconscious steering — measured representationally.
