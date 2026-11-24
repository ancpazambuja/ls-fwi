# Velocity Foundation Model Research

Glossary for the reparameterized-FWI research program and its three-repo
structure. Definitions only — no implementation details.

## Repositories

**Seismic ML Utils (`seismic-ml-utils`)**:
The shared library repo: logging, serialization, MLflow integration,
generic training callbacks, and the experiment harness reused across
projects.
_Avoid_: utils repo, common repo

**Velocity Foundation Models (`velocity-foundation-models`)**:
The repo that defines the foundation-model architectures over velocity
models (autoencoder, VAE), trains them (grid search + refit), and runs
their latent-space evaluations.
_Avoid_: FM repo, models repo

**Reparameterized FWI (`reparameterized-fwi`)**:
The inversion-studies repo: trivial (model-space) latent recovery and FWI,
loading trained foundation models by path.
_Avoid_: the inversion repo, ls-fwi

**Seismic Velocity NN (`seismic-velocity-nn`)**:
The solution-definition layer: for each foundation model, the neural
network itself plus its input/output data pipeline (autoencoder/TransUNet
family, future VAE; dataset builders, preprocessing, tensorization) and
the Encoder/Generator/GeneratorInversion splits. The stable contract that
both the training repo and the inversion repo import; training runs
record its commit for checkpoint reproducibility.
_Avoid_: model zoo (in speech), the zoo repo

**Legacy playground (`ls-fwi`)**:
The original working repo, kept as scratch and historical record after the
split; authoritative for nothing.
_Avoid_: the main repo, the project repo

## Research concepts

**Foundation model (FM)**:
A generative model over velocity models (autoencoder, VAE) trained on the
deep-tomography datasets; its latent space and frozen decoder define the
search manifold used by the inversions.
_Avoid_: encdec (an architecture, not the concept)

**Trivial inversion**:
Latent recovery against the *known* target velocity model (model-space
MSE) with the frozen generator — the fast-fail validation task that
precedes FWI.
_Avoid_: direct inversion, simple inversion

**FWI inversion**:
Latent recovery against the wave-physics data misfit (observed gathers),
with the generator translating the latent into the velocity model.
_Avoid_: full inversion

**Gradient engine**:
The pluggable backend that produces the misfit gradient through the
frozen generator — adjoint-state or automatic differentiation.
_Avoid_: solver, backend
