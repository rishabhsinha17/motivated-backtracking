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
- `verification/` — seeded-random annotated transcripts + 50-pair human label sheet
- `figs/` — forest plot, mechanism bars, answer-level bias, causal summary
- `writeup/draft.md` — the write-up

Data: `value-leakage/` is a copy of github.com/adsingh-64/value-leakage (runs/ = shipped rollouts;
`estimates.json` files regenerated here for all conditions). Requires provider keys in `.env` (not committed).

## Reproducing
Clone github.com/adsingh-64/value-leakage into ./value-leakage (its runs/ data is required; not
redistributed here since that repo carries no license). Put provider keys in .env (see analysis
code for names). The annotated transcripts and lens payload embed that repo's rollout text and
are likewise not redistributed; regenerate with analysis/make_verification_pack.py.
