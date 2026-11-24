# Two decoupled stochasticity protocols

Foundation-model training follows the methodology of "On the Calibration
of Spatial Support in Deep Learning-Assisted Velocity Model Building": J=10
MCCV seeds (7, 79, 181, 307, 433, 571, 701, 853, 997, 1129), each seed
controlling the train/test split, weight initialization, mini-batch
ordering, and dropout masks.

Inversions run their own repetition protocol, decoupled from the FM seeds:
for each target model, J random latent initializations (seeds controlling
the latent init and any stochastic draws such as shot subsampling),
reported as mean ± std. The two seed families are independent, so
variance can be attributed to foundation-model training versus inversion
initialization without a factorial explosion of runs.

## Considered Options

- **Full factorial** (every FM seed × every inversion seed): most rigorous
  attribution; rejected for run-count cost.
- **Single-seed inversions** on 10-seed foundation models: cheapest;
  rejected — inversion results would conflate init luck with model
  quality.
