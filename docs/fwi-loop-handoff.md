# FWI-loop test — handoff from the model-space study

Written 2026-11-26 for a session that will drive the FWI loop from
another project. Read together with `notebooks/marmousi01_read_view.ipynb`
(cells 8, 17, 21 and the audit section are executable recipes) and
`notebooks/data/DATA_DICTIONARY.md`.

## The one standing rule

Any latent optimization runs against the generator pinned to **eval
mode**. The fix lives in `GeneratorInversion.train()`
(`src/processing/ml_models.py`): skorch re-enters train mode on every
step; the override forces the frozen generator back to eval (dropout
off, BatchNorm running stats). Without it the objective is stochastic
and plateaus at a dropout noise floor ~10–20× above the true floor.
The campaign's SGDR schedule (`cosine_annealing_wr`, T_0=68, T_mult=2,
eta_min=1e-6) is destructive **only in train mode** — in eval mode
constant lr and SGDR land on the same floor (the 2×2 experiment,
notebook). Do not inherit SGDR blindly in the FWI loop; constant lr=0.1
Adam was used everywhere in the study.

## Runtime-verified facts (notebook audit, 2026-11-26)

- Only the latent moves: generator params `requires_grad=False`;
  5,066,657 parameters byte-identical after a full 1000-epoch inversion
  (`torch.equal` True, max |Δ| 0.0).
- Recovery floors (trained init, eval, lr=0.1 Adam, 1000 epochs):
  in-distribution test targets 0.00012–0.00030; Marmousi (resampled,
  near-distribution) 0.00030; Sigsbee (farthest OOD) 0.00076.
- No dataset leakage: nearest-neighbor rel-L2 from each benchmark to
  all 1000 dataset models — Marmousi 0.169, Sigsbee 0.269 — against a
  dataset-internal calibration of p5 0.143 / median 0.185. Marmousi is
  *near*-distribution (inside the dataset's own spread); say "not in
  the training set", not "out-of-distribution", for Marmousi.
- Framing: the model-space recovery is a **fit** (it uses the true
  model for 1000 steps); the AE prediction is a one-pass inference.
  The FWI loop replaces the model-space target with the seismic
  misfit — that test is the open question this handoff enables.

## Recipes (exact and executable in the notebook)

- **Checkpoint reconstruction** (no yaml — the logged `inference_grid`
  param is an unparseable Python repr): notebook cell 8. `module__*`
  from the refit run's params via `ast.literal_eval`; attention config
  via `MLFlowUtils.string_to_dict` + `ml_collections.ConfigDict` +
  patches override (`{"size": (1, 1)}`); `f_params` regexed from the
  best run's logged grid.
- Refit run: `9247757e7e0f49459ee4fe8e897fcd14` (encdec 5×5, seed 79,
  `use_transformers=False`); best campaign run `aa821c27…` (trained
  init, lr=0.1, 1000 ep, MSE 0.00439 — noise-floor-limited); its
  checkpoint path is logged inside the run's `inference_grid` param
  (`REFIT__ENCDEC_ENCDEC_5x5_…__params.pt` under
  `data_new_and_bigger/deeptomograph_results/predictions/`).
- **Trained init**: `z0 = Encoder((raw_train[10] / absmax)[:672, :800])`
  — split position 10 = dataset example 182, the same start as every
  campaign trained run. `set_latent` AFTER `initialize()` (in-place
  `copy_`), never re-initialize after.
- **Normalization**: per-lateral-column abs-max fitted on TRAIN split
  models only: `absmax = np.max(np.abs(raw_train), axis=(0, 1))`;
  models are used as `model / absmax` cropped (680, 800) → (672, 800).
  Scaler semantics: `src/preprocessing/pipeline.py::BatchAbsMaxScaler`.
  AE operates on the (672, 800) grid; latent shape (1, 512, 21, 25).
- **Inversion net**: `SkorchGeneratorInversionBuilder` (Adam hard-wired,
  batch_size=1, shuffle off → exactly one latent step per epoch).

## Environment and standing rules

- Cluster runs go to node **gr14b01n01** (`LOCAL_ENVIRONMENT.md` has
  the srun/alias commands; `/gemini` is exported per machine and bound
  into the container).
- MLflow REST: `http://gr14sm02.ep.petrobras.com.br:31947/api/2.0/mlflow`;
  artifacts via `/get-artifact?run_id=…&path=…` (the `artifacts/get`
  API 404s on this server version). **ACTIVE_ONLY runs only** —
  deleted runs are never used as results (standing rule, 2026-11-24).
- Host `python3` is 3.6, stdlib only (no numpy/pandas); the notebook
  container is py3.11 with torch/skorch/ml_collections.
- The 1000-model dataset's provenance is not documented in-repo (it
  was built in the 4th-semester project). Benchmark targets live
  read-only in `projetos/Benchmarks_FWI_ML/Dados/` (`Marmousi01/`,
  `Sigsbee01/`); `sigsbee.npy` is the true model (the `_init` and
  LSRTM files in that directory are not velocity models).
