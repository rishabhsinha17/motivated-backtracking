# Motivated backtracking in the Donation Bet

Analysis + experiments extending Value Leakage (arXiv 2607.14345) §3 / adsingh-64/value-leakage.
Question: what is motivated reasoning mechanically — motivated scrutiny, motivated revision-steering,
diffuse nudge, or final-step bias? Is it unfaithful CoT?

- `analysis/exp1_backtracking.py` — scrutiny-frequency asymmetry (null), alignment of judge trajectories to text
- `analysis/exp1b_revisions.py` — revision-direction contrasts (20/20 motivated), threshold attention
- `analysis/exp1c_robustness.py` — distance/position stratification
- `analysis/exp2_resample.py` — on-policy reject-and-resample suppression arms (Nebius raw-completions prefill)
- `analysis/exp2_analyse.py` — paired-by-trace causal contrasts
- `analysis/judge_shim.py`, `run_judges_native*.sh` — final-answer judging (repo-verbatim prompt/parser)
- `verification/` — seeded-random annotated transcripts + 50-pair label sheet (two independent agent verification passes; disagreements reported in the write-up)
- `figs/` — forest plot, mechanism bars, answer-level bias, causal summary
- `analysis/transfer_adapter.py` + `run_transfer.py` — the same contrasts on the authors' release (43 models x 9 questions)
- `analysis/testA_nocot.py`, `testBC_perspective.py`, `testC2_prospective.py`, `testD_commitment.py` — prediction tests: reasoning on/off (+ number-only cell), analyst frame, numeric self-forecast, prospective verbal report (+ placebo note), forced answer at prefix k
- `analysis/exp3b_controls.py` — confession controls: positive-control disclosure, verdict-question paraphrases, Kimi-K3 arm
- `analysis/covertness_join.py` + `covertness_analyse.py` — the authors' released covertness labels re-keyed by judge-prompt hash and joined to outcomes (denial-faithfulness at scale)
- `analysis/stopping_hazard.py` — motivated stopping (valence-swapped stopping hazard)
- `analysis/bunching.py` — excess-mass (bunching) estimator at the threshold, valence-swap counterfactual
- `analysis/relabel_pairs.py` — judge relabel of all 12,706 revision pairs (+ sonnet escalation sample)
- `analysis/family_collapse.py` — Ibragimov-Muller family-level t-tests for every headline count
- `writeup/mats_writeup.md` — the MATS write-up (`draft.md` is the earlier SPAR take-home)

Data: `value-leakage/` is a copy of github.com/adsingh-64/value-leakage (runs/ = shipped rollouts;
`estimates.json` files regenerated here for all conditions). Requires provider keys in `.env` (not committed).

## Reproducing
Clone github.com/adsingh-64/value-leakage into ./value-leakage and (for the transfer/covertness/bunching analyses) github.com/TruthfulAI-research/value_leakage_data into ./tfai_data (its runs/ data is required; not
redistributed here since that repo carries no license). Put provider keys in .env (see analysis
code for names). The annotated transcripts and lens payload embed that repo's rollout text and
are likewise not redistributed; regenerate with analysis/make_verification_pack.py.
