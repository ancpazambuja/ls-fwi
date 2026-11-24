# Repo-prefixed MLflow experiment naming

All MLflow experiments are named with a repo-derived prefix, mirroring the
four-repo split:

- `vfm.<arch>.<res>.<stage>[.seed.<s>]` — velocity-foundation-models
  (e.g., `vfm.ae.highres.gs`, `vfm.ae.highres.refit.seed.7`,
  `vfm.ae.highres.latent-eval`)
- `rfwi.<kind>[.<engine>].<study>` — reparameterized-fwi (e.g.,
  `rfwi.trivial.latent-recovery`, `rfwi.fwi.adjoint.shots-study`)

Each repo also owns a smoke sandbox following the established convention
(`vfm.smoke*`, `rfwi.smoke*`): kept, never deleted, never mixed with real
runs. Statistics extraction filters by prefix; the owning repo is visible
in the experiment name. Legacy runs (experiment `encdec.highres.*`,
smoke experiments `encdec.highres.smoke*`) keep their historical names.

Inversion runs tag their lineage (`fm.refit_run_id`, `fm.arch`,
`fm.dataset`) so results can be grouped by foundation model; configuration
fidelity continues to be read from the refit run's attributes.
