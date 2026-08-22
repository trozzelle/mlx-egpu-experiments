# C1R-W0 Runner Exit Fix

Updated `_normalize_result` so a nonzero native runner process status overrides any parsed `exit_status` before acceptance validation. This preserves the existing acceptance failure route (`runner exit_status is nonzero`), opens the result, and drives rejected-NPZ cleanup. A zero process status continues to retain the parsed `exit_status` behavior.

No commands were run, per assignment constraint.
