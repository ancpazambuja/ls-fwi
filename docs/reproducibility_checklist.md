# Reproducibility checklist

Complete before sharing the repository or referencing it in a manuscript.

- [ ] All source files and documentation are in English.
- [ ] License is chosen and compatible with third-party dependencies.
- [ ] Installation and smoke-run instructions have been tested from a clean
      environment (reference container).
- [ ] Container base image and key dependencies are pinned (shared reference
      environment with spatial-support-vmb).
- [ ] Smoke run works without the restricted research data (local subset or
      documented alternative).
- [ ] Research dataset restrictions and access procedure are explicit.
- [ ] All experiment configurations and seeds are included (per-stage
      commands in the user guide / protocol).
- [ ] Training, evaluation, and analysis stages are documented.
- [ ] Inputs, outputs, units, shapes, defaults, and failure behavior are
      documented.
- [ ] Expected outputs are described, including tolerances for
      non-deterministic results.
- [ ] No restricted data, secrets, or local paths are committed.
- [ ] A versioned release is created and tested against the documented
      commands.
