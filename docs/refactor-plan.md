# Refactor Plan — four-repo split (science-first)

Output of the grill-with-docs planning session (2026-11-23). Decisions are
recorded in `docs/adr/0001..0003` and `GLOSSARY.md`. This document is the
execution plan.

## Repositories

Dependency DAG: `seismic-ml-utils ← seismic-velocity-nn ← {velocity-foundation-models, reparameterized-fwi}`

| Repo (package) | Contents |
|---|---|
| `seismic-ml-utils` (`seismic_ml_utils`) | Logging, serialization, MLflow integration (wrapper, skorch callbacks, utils), generic skorch callbacks (PredictionHistory, gradient clipping, LR-scheduler helpers, TwoPhaseTraining, ParamLogger), generic task/thread framework, harness pattern (`lib.sh` reference copy) |
| `seismic-velocity-nn` (`seismic_velocity_nn`) | Solution-definition layer: architectures (`EncoderDecoder`, TransUNet, future VAE; `Encoder`/`Generator`/`GeneratorInversion` splits) + velocity-model data pipeline (dataset builders, preprocessing, tensorization) |
| `velocity-foundation-models` (`velocity_foundation_models`) | Per-arch experiment subdirs (`src/fm/ae/`, `src/fm/vae/`): gs grids, refit configs, wiring; **one latent-eval module + protocol doc** (cosine-distance pair strategies, interpolation/extrapolation α grids, MSE/MAE/SSIM per α, eval seed); harness copy (`vfm.*`) |
| `reparameterized-fwi` (`reparameterized_fwi`) | `src/inv/trivial/` (model-space latent recovery), `src/inv/fwi/` with pluggable gradient engines (adjoint-state ↔ autodiff behind one interface); shot-count and multi-scale studies as configurations; harness copy (`rfwi.*`) |

Rules: `reparameterized-fwi` imports `seismic_velocity_nn` (definitions +
data), never the training repo. Training runs record the
`seismic-velocity-nn` commit for checkpoint reproducibility.

## Conventions

- **MLflow** (ADR-0003): repo-prefixed experiments — `vfm.<arch>.<res>.<stage>[.seed.<s>]`,
  `rfwi.<kind>[.<engine>].<study>`; per-repo smoke sandboxes (`vfm.smoke*`,
  `rfwi.smoke*`), kept and never deleted; inversion runs tag lineage
  (`fm.refit_run_id`, `fm.arch`, `fm.dataset`).
- **Stochasticity** (ADR-0002): FMs — the paper's 10-seed MCCV
  (7, 79, 181, 307, 433, 571, 701, 853, 997, 1129). Inversions — J random
  latent-init draws per target model, reported mean±std; seed families
  decoupled. **Seed-coupling constraint (user, 2026-11-24)**: in the
  current `ls-fwi` code ONE seed controls both the train/test split and
  the latent-init randomness (`random`/`randomn` draws, and which
  training example `trained` starts from) — so this campaign has
  effectively J=1 draw per (seed, config), and duplicates are numerical
  replicates, not fresh draws. `reparameterized-fwi` must decouple them:
  the inversion takes the split from the refit (indices carried with the
  checkpoint/config) and uses an independent latent-init seed family, so
  J draws vary only the latent seed.
- **Environment**: the reference `dt2_v4` container for all repos; editable
  installs; Python packages named as above.
- **Harness**: per-repo copies trimmed per stage; `run_info.txt` manifests
  per run; `lib.sh` synced by hand if changed.
- **Docs**: each new repo starts with copies of the relevant ADRs and
  glossary; `ls-fwi` remains the legacy playground (non-authoritative);
  each new repo proves parity via cluster smoke (`validate_smoke.sh`)
  before deprecating the corresponding `ls-fwi` concern.

## Known migration notes (from the cluster validation)

- `get_train_callbacks` (`src/utl/classes.py`) couples generic callbacks to
  app-specific `GlobalVars`: generic callbacks → `seismic-ml-utils`;
  the callback *factory* → `velocity-foundation-models`.
- Deferred cleanups to do inside the new repos: hardcoded latent shape
  `(1, 512, 21, 25)` and inverted test indices `[1, 10, 20]`;
  `InferenceGrid` hardcodes; restart feature unimplemented (implement
  properly in the new harness if needed); preprocessing scaler-state
  serialization (`pipeline_serialization_todo.py`).

## Phases (science-first; ~1 week)

- **Phase 0 — now, in `ls-fwi`**: finish the trivial-inversion analysis →
  the fast-fail verdict. No refactor blocking.

  **Data inventory (as of 2026-11-23, verified against MLflow)**: the
  init × lr × epochs sweep is COMPLETE — 460 FINISHED runs in experiment
  `encdec.highres.inversion` (Nov 9–12): 396 sweep runs with logged
  params + 64 earlier param-less runs. Sweep dimensions: 4 latent inits
  (`trained`, `zeros`, `random`, `randomn`) × 5 lrs (0.001–1.0) × epoch
  budgets {50, 100, 500, 1000} (plus exploratory 20/25/200) × 4 refit
  models — encdec 3×3 (seed 7, `6d115baa…`), encdec 5×5 (seed 79,
  `9247757e…`), encdec 7×7 (seed 571, `b01fdd28…`), transencdec 5×5
  (seed 571, `32409900…`) — each inverting the SAME 3 target models:
  the hardcoded inverted test indices `[1, 10, 20]`
  (`task_processing.py: models_to_invert`), hand-picked difficult
  examples from the refit's test set. The refit's 10% MCCV holdout
  (100 models; 900/100 split of the 1000-sample dataset) is logged as
  context (`_x_test_indexes`), not the inversion target set. All sweep runs share
  `lr_scheduler=cosine_annealing_wr` and `patience=300`. A 3-run sanity
  family (`tst__inverting_on_train_dataset`, refit 3×3) inverts
  *train*-split models. The 64 param-less runs are an earlier launcher
  generation (40 on a `50-samples` variant, 24 full-samples; 25 carry
  only `dur`+`train_loss`). **Cleaning policy (user-confirmed
  2026-11-24):** skip every `50-samples` run (debugging: 68 runs + the
  3 `tst__` sanity runs); per grid combo keep the LAST duplicate unless
  it is problematic (44 duplicates dropped; verified — 0 problematic
  last-runs, all 18 invalid-final-metric runs were debug runs). Clean
  analysis set: 321 combos = complete 4×5×4 grid × 4 refit models (320)
  + 1 exploratory (`trained, lr=1.0, 200ep`; it fits the monotone budget
  trend 0.0044→0.0037 of its config). User decision 2026-11-24: the
  notebook's analysis set = the 320 grid combos; the probe stays as
  budget-curve evidence only. Dropped duplicates are same-config re-runs:
  the seed controls both the data split and the latent draw, so re-runs
  share the same latent init — numerical replicates (measured relative
  spread exactly 0 for most combos; exceptions sit at the stability edge
  lr=1.0/ep=50, ≤5%). They confirm the sweep is deterministic per seed
  and that the high-lr zeros degradation is real, not noise.

  **Headline results** (CLEAN set; final model-space MSE, mean over the
  4 refit models; each run = mean over its 3 inverted targets — the
  hand-picked difficult examples [1, 10, 20]; @1000ep):
  trained 0.0065 @ lr=1e-3 then FLAT 0.0043–0.0044 for lr≥0.01 (best
  0.0043 @ 0.1) — lr-robust; zeros 0.0047 @ lr≤0.01 degrading to 0.0105
  @ 1.0; random 0.0045 @ lr=0.01–0.1; randomn 0.0045 @ 0.1. SSIM ≈ 0.87
  at the sweet spot. Budget: @lr=0.01–0.1 most convergence by 500ep
  (0.0049–0.0054 → 0.0046–0.0048 @1000ep); @lr=1e-3 slow, still
  improving @1000ep (0.0057). Per refit (clean grid): all-mean
  0.0086–0.0093, min 0.0037 (3×3) … 0.0047 (7×7) — architecture matters
  little. Baseline (refit run metrics, mean over its 100-model holdout):
  MSE 0.00083–0.00117, SSIM 0.89–0.91. **Signal: OOD latent inits reach
  the manifold (random ≈ trained within ~5%); open question — every init
  plateaus ~4–5× above the refit MSE, now known to have THREE candidate
  contributors: difficulty-selected targets (3 hard examples vs a
  100-model mean), the warm-restart final-state artifact, and a true
  optimization limit (see deliverables).** (Retracted: the earlier
  "trained best 0.0040 @ ep=200" was a `50-samples` debug run.)

  **Interpretation notes (user, 2026-11-24)**:
  (i) Per-model difficulty: model-space MSE varies a lot across test
  examples — easy models reconstruct/invert much better (cf. the
  results section of "On the Calibration of Spatial Support in VMB").
  CORRECTION (user, 2026-11-24): every sweep run inverts the same 3
  hand-picked difficult targets ([1, 10, 20], hardcoded in
  `task_processing.py`), while the refit baseline is a mean over its
  100-model holdout — the comparison is difficulty-MISMATCHED and the
  inversion-vs-refit gap is inflated. The fair form is a per-model
  paired comparison against the refit's metrics for [1, 10, 20]
  (final-experiments checklist).
  (ii) The final-state-at-peak-lr reporting issue is confirmed real:
  fix it for the FINAL experiments; this fast-fail analysis keeps the
  numbers as-is and carries the caveat.
  (iii) `latent_init=trained` = the latent of a RANDOMLY CHOSEN
  TRAINING-set example (an in-distribution manifold point) — NOT the
  encoder's latent of the target model. All inits therefore converge
  to the same ~0.0045 attractor from manifold points, noise, or zeros
  alike; the encoder-of-target init was never tested.

  **Remaining analysis deliverables**: (a) light per-epoch `train_loss`
  extraction for key configs — quantifies the warm-restart artifact
  (sawtooth shape, dip depth, best-so-far vs final) as presentation
  material, not as a campaign fix; plus per-run `inversion_mse_min/std/
  max_test_set` to quantify the difficulty spread; (b) the fast-fail
  PRESENTATION for the AE loop (FM gs/refit → trivial inversion →
  verdict + caveats).
- **Phase 1**: create `seismic-ml-utils` + `seismic-velocity-nn`; imports
  verified inside the reference container. *(DONE locally, 2026-11-24 —
  commits `f2b83b2` / `8e8acee`; all modules compile, container import
  check OK on gr14b01n10, `DataPaths` getters byte-identical to
  `ProjectDirs` for both layouts, Encoder/Generator split parity PASS.
  Pending: GitHub remotes + push, editable installs, pytest in container,
  GPU transformer-path test. ProjectDirs decision implemented: nn
  builders receive `DataPaths`; no path policy in the nn repo.)*
- **Phase 2**: create `reparameterized-fwi` — trivial inversion ported and
  smoke-validated; FWI skeleton with the gradient-engine interface
  (actual FWI physics is post-refactor research).
- **Phase 3**: create `velocity-foundation-models` — ae port (gs/refit/
  latent-eval), vae next; cluster smoke proves parity.

  **AE re-run checklist for the FINAL experiments (post-refactor; AE +
  VAE + possibly other FMs)**: (1) fix final-state reporting —
  best-so-far checkpointing or a cold schedule end; (2) add an
  `init=encoder-of-target` variant to measure the manifold-coverage
  ceiling directly in the inversion framework; (3) per-model paired
  difficulty analysis (interpretation note i); (4) 10-seed MCCV per
  ADR-0002.
- Days 6–7: buffer.
