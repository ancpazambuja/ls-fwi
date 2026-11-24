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
  decoupled.
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
- **Phase 1**: create `seismic-ml-utils` + `seismic-velocity-nn`; imports
  verified inside the reference container.
- **Phase 2**: create `reparameterized-fwi` — trivial inversion ported and
  smoke-validated; FWI skeleton with the gradient-engine interface
  (actual FWI physics is post-refactor research).
- **Phase 3**: create `velocity-foundation-models` — ae port (gs/refit/
  latent-eval), vae next; cluster smoke proves parity.
- Days 6–7: buffer.
