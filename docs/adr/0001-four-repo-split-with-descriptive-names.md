# Four-repo split with descriptive names

The research program is split into four repositories rather than a
monorepo or a two/three-repo layout:

- `seismic-ml-utils` — shared infrastructure: logging, serialization,
  MLflow integration, generic training callbacks, task/thread
  parallelism, and the experiment-harness shell pattern.
- `seismic-velocity-nn` — the PyTorch-side domain package for velocity
  models: architecture definitions (`EncoderDecoder`/TransUNet, the
  future VAE, and the `Encoder`/`Generator`/`GeneratorInversion` splits)
  plus the velocity-model data pipeline (dataset builders,
  preprocessing, tensorization). Not general seismic infrastructure.
- `velocity-foundation-models` — foundation-model research: grid search,
  refit, data handling, and latent-space evaluation.
- `reparameterized-fwi` — inversion studies: trivial (model-space)
  latent recovery and FWI, loading trained foundation models by path.

Both the foundation-models repo and the inversion repo import
`seismic-velocity-nn` (architectures + data pipeline), which keeps
architectures and data handling versioned together — checkpoint
compatibility and consistent loading/normalization — and avoids making
the experiments repo a dependency of the inversion studies. Descriptive names were chosen over keeping the
`ls-fwi-*` brand so each repo's purpose is self-evident. The original
`ls-fwi` repo is kept as a playground/historical record; the new repos
are authoritative.

## Considered Options

- **Monorepo** (`packages/` per concern): rejected — weaker per-project
  isolation and no reuse outside the program.
- **Three repos** (architectures inside the foundation-models repo):
  rejected — inversion would depend on the experiments repo, and
  architecture changes would untangle from checkpoint compatibility.
- **Three repos** (architectures inside seismic-ml-utils): rejected —
  dilutes the infrastructure repo with torch/model code.
- **`ls-fwi-*` naming family**: rejected — names the utilities and
  inversion repos after the research topic rather than their purpose.
