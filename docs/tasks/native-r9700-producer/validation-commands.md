# Native R9700 Producer — Validation Commands

This is the active shared command ledger for current F1–F6, P1–P5, and Q1 task packets. The complete committed C0–C3 command history is preserved verbatim in [`validation-commands-c0-c3.md`](../../archive/tasks/native-r9700-producer/validation-commands-c0-c3.md); new task packets must add concrete commands here before execution.

## Fixed environment

Use this Python for Python-side validation in this repo:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3
```

Do not rely on `python3` from `PATH`.

For AMD eGPU/tinygrad comparison runs that intentionally use tinygrad:

```sh
DEV=AMD
JITBEAM=2
HF_HOME=${HOME}/Development/ml/models
```

Native R9700 producer commands must not import or call tinygrad unless explicitly running a labeled comparison/control command outside the producer path.

## Exact commands known now

### Existing Python regression suite

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v
```


### Existing harness syntax check

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m py_compile tinygrad_kv_worker/harness.py
```


### Existing Phase 0 GPU parity command

This is a regression/control command for the validated tinygrad producer path, not a Native R9700 producer command:

```sh
DEV=AMD JITBEAM=2 HF_HOME=${HOME}/Development/ml/models \
  ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m tinygrad_kv_worker.harness \
  --gguf mlx_models/meta-Llama-3.2-1B-Instruct.F16.gguf \
  --mlx mlx_models/meta-Llama-3.2-1B-Instruct \
  --out docs/path-a-validation-results.md \
  --run-tag meta-f16-final
```

Historical Phase 0 acceptance evidence lives in `docs/path-a-validation-results.md` and `docs/archive/tasks/tinygrad-kv-worker/phase-0-parity.md`.

### Documentation whitespace check

Use this after task-doc or design-doc edits:

```sh
git diff --check
```

## Command discovery policy

Each new task packet must record its exact focused test, broader regression, native build, hardware smoke, log path, and expected observable result here before the command is used as promotion evidence. Cite the current roadmap phase and task document; do not attach new commands to archived C0–C3 packets.

## Log requirements for all GPU/native runs

Every GPU/native run must write a reviewable local log under `logs/` or record an explicit remote log artifact path. Logs must include:

- command line;
- timestamp;
- runtime substrate and device identity if discoverable;
- model/config path or note that no model is used;
- prompt length or input shape;
- output comparison result or digest;
- exit status;
- failure traceback/error text when failing.

Logs and model files must not be committed.

## Gate reminders

- Producer-swap acceptance is token-exact `P == R`, not semantic equivalence.
- mlx-lm injected decode uses an imported `S-1` prefix cache plus the final prompt token.
- Llama-3 RoPE scaling must match the MLX sidecar config.
- Native R9700 producer code must not depend on tinygrad.
