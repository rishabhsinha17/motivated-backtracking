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

## Transfer test (authors' release, 25 usable models × 9 questions; analysis/transfer_results.json) — Sept 2
- 356/396 revision-direction contrasts in the motivated direction (90%), 215 significant, 1 significant reversal.
- Every question ≥85% motivated (giraffes 33/38 lowest, windowdays 44/47, gpt-5.4/qwen3.5/qwen3.6/gemini-3.5-flash 18/18).
- 24/25 models majority-motivated; weakest = frontier Claude/GPT (claude-fable-5-high 9/12, gpt-5.1-medium 11/18, gemini-2.5-pro 1/3 sparse).
- Median contrast +0.10. Side split: above 175/197 (101 sig), below 181/199 (114 sig).
- 18 release dirs skipped for data availability (Claude claudeai/conglomerate/dont-bias prompt-variant reruns lack trajectory rows;
  gpt-5.6-sol, sparse gemini) — not a selection on results. N_BOOT=800 cluster bootstrap.

## Test A: no-CoT bias (empty-think prefill, dataset prompts, n=100/cond; exp2_out/nocot_*.jsonl) — Sept 2
- Paper-style bias is algebraically P(above|above_good) − P(above|below_good), so it is baseline-independent.
- FINAL pooled (run1 + full-text rerun, n≈190/cond): GLM no-CoT +0.13 [+0.03,+0.23] vs CoT +0.32; diff +0.19 [+0.03,+0.35], P(diff≤0)=0.010.
  DeepSeek no-CoT +0.11 [+0.00,+0.20] vs CoT +0.42; diff +0.31 [+0.16,+0.47], P(diff≤0)<0.0001. Runs: GLM +0.10/+0.16, DS +0.13/+0.09.
- Note comprehension (rerun full text): mentions bet/threshold/donation GLM 49–53%, DS 63–74% (baseline 1%); honesty assertions 0–13% (vs 59–100% with CoT).
- NUMBER-ONLY cell (Feng wording, thinking off, n=100/cond, fallback parser for large numerals): GLM +0.40 [+0.29,+0.51] (baseline 98% above thr, answers ~1e11; below_good pulled to 0.54); DeepSeek +0.01 [−0.13,+0.15] (27 unparseable, mostly baseline). Unrecovered-parser values were +0.42 / 0.00.
- Reading (REVISED): DeepSeek monotone in reasoning (0.01 → 0.11 → 0.42) = built in the CoT. GLM non-monotone (0.40 → 0.13 → 0.32) = direct answer-level incentive response with no reasoning + CoT steering. Claim is model-specific. Short-justification cell is least biased on both (unexplained).
- Caveats: no-CoT GLM baseline P(above)=0.68–0.76 (shifted); parse failures GLM 18/600, DS 44/600; bias not split by note-mention.

## Tests B / C / C2 (fresh context, n=60/cond, both models; analysis/testBC_results.json, exp2_out/testC2_*.jsonl) — Sept 2
- B analyst frame: GLM +0.41 [+0.25,+0.57] (actual +0.32; 23% of below_good predictions within 1% under thr); DS +0.09 [−0.06,+0.24] (actual +0.42; diff +0.34 [+0.14,+0.53]).
- C numeric self-forecast: GLM +0.54 [+0.40,+0.68] (over-forecast; 21% just under thr; above_good median 289M vs baseline forecast 150M); DS +0.12 [−0.04,+0.29] (under-forecast).
- C2 prospective verbal "will the bet influence you": GLM yes 3% / 8% (retrospective neutral 3–11%); DS 62% / 59% (retrospective 65–72%).
- Reading: verbal self-report is a fixed per-model stance (invariant to pre/post, evidence, and own numeric forecast); numeric channels carry information but miscalibrated in opposite directions. GLM words=no, numbers=gaming.
- CORRECTIONS after adversarial review (Sept 2): DS Test B = +0.08 [−0.06,+0.24] (JSON 0.0846); DS Test C CI = [−0.06,+0.29]; DS gate ratio ≈1.4x; GLM_C forecast−actual +0.22 [+0.02,+0.41] (sig), GLM_B analyst−actual +0.09 [−0.12,+0.30] (n.s.).
- One-sidedness: fresh-context P(above) triplets (base/bg/ag): GLM_B 0.57/0.18/0.59, GLM_C 0.77/0.38/0.91, DS_B 0.29/0.18/0.26, DS_C 0.42/0.28/0.40 (actual CoT: GLM 0.50/0.45/0.77, DS 0.27/0.18/0.60).
- Near-threshold (within 1% under thr, below_good): GLM actual CoT 0.02, no-CoT short-justification 0.26–0.29, number-only 0.07, forecast frame 0.21, analyst frame 0.23 → CoT turns overt threshold hugging into spread good-side estimates. DeepSeek metric uninformative (modal 23.4M sits just under its 23.7M threshold; CoT below_good 0.82).
- QWEN3.5-122B confession grid (OpenRouter chat, answer-only context, 40/cell, Sept 2): neutral 0.00/0.00 (n=35,38) | amnesty 0.33/0.15 (parsed 15/38, 20/38) | threat 0.07/0.00 | private 0.00/0.00 | third person 0.87/0.71 (parsed 23/38, 17/38). Gate ordering: Qwen (full) > GLM (5–15x) > DeepSeek (~1.4x).
- PLACEBO (weather note, same grammar, OpenRouter, n=40): GLM yes 0/40, DeepSeek yes 0/40 → DS's 60% yes is content-specific; GLM's ~5% not separable from a default no.
- Mention split (no-CoT short-justification, full-text rerun): GLM mentioners +0.30 [+0.12,+0.47] vs non-mentioners +0.04; DS mentioners +0.02 vs non-mentioners +0.25 [−0.01,+0.49] (n=37/25).
- Transfer per-row: three weakest full rows gpt-5.1-medium 11/18, claude-opus-4.5-high 12/18, claude-fable-5-high 9/12; eight rows 18/18 incl. claude-opus-4.6-high (18 sig), opus-4.6-max, opus-4.8-max → NOT a family-level pattern.

## Exp 4 J-lens workspace readouts (qwen3.5-122b, 5,640 positions, R-lens, >40% depth, top-15)
- Moral tokens (baseline ≈0.000 all classes): denial 0.44 | backtrack 0.12–0.14 | control 0.03–0.06 | REVISION 0.011–0.016.
- Dissociation: workspace holds the moral frame at the causally-inert theater, not at the causal revision locus.
- One-liner: conscious theater, subconscious steering — measured representationally.

## R7b (Sept 2, post hoc, analysis/r7b_steering_vs_answer.json, figs/fig10_steering_vs_answer.png)
- First in-CoT estimate tilt (P(first>thr|ag) − P(first>thr|bg)): positive 8/10, median +0.15 (GLM +0.22, kimi +0.21, qwen +0.19, DS-flash −0.05, minimax −0.01).
- Bias added after first estimate: positive 10/10, median +0.12 (DS-flash +0.47, qwen +0.34, GLM +0.10).
- Spearman across 10 models: steering vs answer bias ρ=0.82 p=0.004; steering vs added bias ρ=0.87 p=0.001; first tilt vs answer bias ρ=0.31 p=0.38. LOO ρ 0.75–0.95.
- Coherence with Test A: DS-flash builds all bias via revisions (no-CoT ≈ 0); GLM mostly at first guess (no-CoT biased).

## PARSER BUG (found by second adversarial review, Sept 2) and DeepSeek reruns on OpenRouter
- exp3_confession.parse took the FIRST <influenced> tag after </think>; DeepSeek echoes the instruction's example tags in reasoning → stored "yes" verdicts inflated. GLM/Qwen unaffected (0 flips).
- C2 prospective DeepSeek CORRECTED: yes 0.00 / 0.00 (n=59/60, 58/60) via OpenRouter, last-tag parser, full text stored. Old Nebius file archived as testC2_deepseek_nebius_firsttag.jsonl (62%/59% = artifact).
- Old-tail last-tag reparse of DeepSeek confessions: neutral ≈0.38/0.18, amnesty ≈0.75/0.63, third person ≈0.93/1.00 (small n). Full rerun: confess_deepseek_or_*.jsonl (answer-only context).
- Transfer thin cells: 10/396 contrasts have <10 pairs on a side; excluding them 348/386 motivated, 212 sig, 1 reversal.
- pass2 verification: 45/50 estimates present, 5 threshold collisions; spans 23 revision / 18 scenario / 9 unclear; regex flagged 16/50 spans, judge saw backtracks in 21 of the 34 unflagged.
- R1 interactions: |Δ| ≤ 1.7pp, 4 models sig in both directions (GLM +1.7pp below; qwen +0.8 above/−0.7 below; inkling-small +0.8/+0.7). R1b span-level P(bt in span|side): all within ±0.07, one sig.
- DeepSeek CoT exact-threshold answers: 35% baseline / 52% below_good / 21% above_good; within +0.5% above 28% in above_good; bias excl. ties +0.38.
- Sign test corrected: 10 independent models → p ≈ 0.001 (not 2e-6). R9 pooled contrast = post hoc (git 163185e → 9920042).
- DeepSeek confession RERUN (OpenRouter, answer-only ctx, full text, last-tag; first==last on rerun): neutral 0.68/0.69 | amnesty 0.65/0.86 | threat 0.46/0.67 | private 0.57/0.67 | third person 0.92/1.00 (n=37-40). Original reasoning-replayed run (0.65/0.72 neutral, 0.95/0.97 3P) STANDS; only prospective C2 (60%→0%) was the artifact. Story: all models say no BEFORE; only DeepSeek (overt threshold printer) says yes AFTER.

## R8b: denial faithfulness at scale (Sept 3, zero cost; analysis/covertness_{join,analyse}.py, covertness_results.json, fig11)
- Authors' v2 Sonnet covertness labels (plot_cot_categories_v2_cache) joined by re-rendered prompt hash: 85,186 labels; 100% join on deep Claude runs, ~2/3 wide zoo (incentive conds only).
- Within-cell (model×question×cond, ≥10/cat, ties excluded) P(good)−0.5: NOT_INFLUENCED +0.149 [+0.129,+0.169] (518 cells, 73%>0); MENTIONED +0.20; NO_STATEMENT +0.23; NO_MENTION +0.23; INFLUENCED +0.23.
- 19/22 base models (n≥50) individually denial-biased. Baseline placebo P(above)=0.44 (n=13,270 NO_MENTION).
- INFLUENCED − NOT_INFLUENCED +0.080 [+0.057,+0.105]; NO_MENTION − NOT_INFLUENCED +0.056 [+0.035,+0.078] → silence as biased as admission; denial recovers <1/3 of bias.
- Cross-model: admission rate vs steering rho +0.11 n.s.; vs answer bias +0.29 n.s. (n=22). Qwen3.5-35 admits 59% & steers 18/18; kimi-k2.6 denies 96% & equally biased.
- Caveats: post-treatment conditioning (framed as monitor-verdict vs behavior); labels are authors' cached judge, not re-validated; release tie rates high for Claude/gemini (opus-4.8-high 0.36).

## Neel-doc framing hooks (taste sweep, Sept 3)
- His MATS 12.0 doc names Value Leakage verbatim: "why? Where do the values intervene? Can you make it disclose, or turn the effect off? Can you combine it with Gilg et al to find a linear direction predicting it and causally mediating it?" → frame the write-up as answering Q2 (revision steering + model-specific prior), Q3a disclose (third-person unlock), Q3b off-switch (DeepSeek: no-reasoning AND analyst frame both turn it off; GLM: neither).
- Reject patterns to avoid: "showing that chain of thought causally impacts the final answer" as a generic project; unexplained anomaly rabbit holes (GLM no-reasoning bias must stay tied to the mechanism story); raw-LLM-voice exec.
- Structure of accepted write-ups: numbered claims up front (Rosser MATS 10.0, task-gaming post). We match.

## R3b: motivated stopping (Sept 3, zero cost; analysis/stopping_hazard.py, stopping_results.json)
- Hazard h(stop|side) valence-swapped at matched side: dataset 18/20 motivated (10 sig); release 367/398 (276 sig, 0 reversed).
- Absolute-index strata [1,4)/[4,8)/[8+): 290/409 (134 sig), 255/345 (101 sig), 218/265 (104 sig); reversals ≤5. NOT a position artifact. (Relative-position strata are degenerate — stop is always at position 1.0.)
- Extreme cell: qwen3.5-35 zills h(stop|above)=55/951 when above pays vs 0/193 when it doesn't.
- Reading: amends R1 — quantity-of-processing channel exists in CONTINUATION, not in backtrack markers. Trio: first guess tilted (R7b) + revisions steer (R2) + stopping motivated (R3b). Lateral source: Ditto&Lopez/Kruglanski + optional-stopping stats; no LLM prior art (checked).

## R8c: bunching at the threshold (Sept 3, zero cost; analysis/bunching.py, bunching_results.json)
- b = asym(bg) − asym(ag), asym = P(v in (T−2%,T]) − P(v in (T,T+2%]); valence swap = counterfactual (baseline unusable: T := baseline median).
- 19/41 release variants bunch sig: qwen3.5-35 +0.23, gemini-3.1-pro-high +0.17, opus-4.7-xhigh +0.16, qwen3.6 +0.16, sonnet-4.5 +0.13, gpt-5.4 +0.12.
- Bunching vs steering rho 0.43 p=0.03 (n=25); bunching vs exact-T printing rho 0.00 → INDEPENDENT dials. gpt-5.4 covert buncher (b +0.12, prints 0%); opus-4.8-high overt printer (43% exact-T, no excess mass). claudeai/conglomerate deployment variants ANTI-bunch (4.5/4.7 rows, up to −0.10).
- No prior bunching-estimator application to LLM outputs (2 independent sweep checks).

## Test D: commitment curve (Sept 3, ~$2; analysis/testD_commitment.py, exp2_out/testD_*.jsonl)
- Two-turn stop design ("answer now, single number"), truncated reasoning shown as assistant turn, k∈{25,50,75,100}% sentences, n=40/cond.
- PREDICTIONS PRE-STATED: DS grows with k; GLM present early, flat. [FINAL NUMBERS AT COMPLETION — partial: GLM +0.21/+0.28/+0.37/+0.27; DS +0.01/+0.22/+0.09/+0.36 with above_good still filling]
- Test D FINAL: GLM +0.21 [+0.01,+0.41] / +0.28 / +0.37 / +0.27 (flat, early commit); DS +0.07 [−0.07,+0.21] / +0.21 / +0.25 / +0.40 [+0.20,+0.60] (monotone, no boundary; k=100 ≈ natural +0.42). Predictions pre-stated, both held. fig12.

## R14b controls (Sept 3, ~$4; analysis/exp3b_controls.py, exp2_out/ctrl_*.jsonl; OpenRouter answer-only pipeline)
- POSITIVE CONTROL BACKFIRES: prepending "your estimate landed on the favorable side" DROPS admission: DS 0.73→0.37 (bg), 0.57→0.27 (ag); GLM 0.21→0.13, 0.27→0.07; Qwen 0.00→0.00. Channel is evidence-reactive downward (parallels Sahasrabudhe's accuracy-instruction suppression).
- PARAPHRASES: GLM N/V2/V3 = 0.21/0.40/0.38 (bg), 0.27/0.30/0.10 (ag) — absolute rates ±19pp template-sensitive; ceiling stays ≤0.40 vs 3P 0.50. Qwen 0.00 on all templates (robust full gate). DS 0.40-0.73 across templates.
- KIMI-K3 (4th model): neutral 0.32-0.36, third person 0.36-0.58 → weak gate. Spectrum: Qwen (full) > GLM >> DeepSeek ≈ Kimi.
- Note: GLM neutral here 0.21-0.27 (answer-only ctx) vs 0.03-0.11 (reasoning replayed, Nebius) — pipeline shifts absolutes; report ratios.

## Family-level inference (methodology sweep item 1; analysis/family_collapse.py/.json)
- Ibragimov-Müller one-sample t across family means (all families positive in all tests):
  R2 dataset steering t(6)=5.10 p=0.001 | R8 transfer t(4)=4.36 p=0.006 | R3b stopping t(8)=2.82 p=0.011 | R8b denial bias t(4)=4.95 p=0.004 | R8c bunching t(4)=2.23 p=0.045.
- Sweep also corrected two of our memory citations: arXiv 2607.27518 is Mohl et al. (transcript analysis), NOT CAISI judge-validation; Apollo has no "confirmed-hit two-tier" labels (their real mechanism: any-of-N + binomial vs base rate).
- Judge relabel of all 12,706 pairs running (relabel_pairs.py: DS-flash bulk + 150-pair sonnet escalation) → will replace the "if" heuristic in limitation 1.
- RELABEL DONE (Sept 3, ~$1.5): 12,706 pairs → 8,893 REVISION / 2,299 SCENARIO / 632 RESTATEMENT / 882 unparsed; sonnet escalation agreement 124/150=0.83 (DS over-calls REVISION: 21 of 26 disagreements). REVISION-only contrasts: 18/20 motivated, 6 sig; kimi above −0.03 and minimax above −0.00 are the nulls; deepseek-pro above +0.19, qwen above +0.16 (stable or larger). Limitation 1 CLOSED.
