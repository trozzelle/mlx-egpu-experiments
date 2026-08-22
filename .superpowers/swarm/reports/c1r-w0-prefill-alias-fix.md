# C1R-W0 Prefill Alias Guard

Implemented the native rejection-branch early return for worker results with `failure_stage == "output_path_conflict"`. It preserves the existing concise stderr error and returns failure before generic output removal or log writing. Other rejected native-worker results retain their existing cleanup and logging path.

No commands were run, per assignment constraint.
