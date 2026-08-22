# C1R-W0 Worker Alias Guard

Implemented a pre-side-effect lexical path-conflict guard in `run_native_prefill` using `os.path.normpath(os.fspath(...))`. Equivalent output and log paths now return the fail-closed `output_path_conflict` result with blocked acceptance, matching the native runner's failure text, before command construction or any filesystem/subprocess action.

No commands were run, per assignment constraint.
