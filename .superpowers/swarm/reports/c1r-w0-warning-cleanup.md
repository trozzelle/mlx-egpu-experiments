# C1R-W0 warning cleanup

Removed the unused static helper definitions `json_escape` and `json_number_or_zero` from `native_r9700/runtime.cpp`.

The supervisor compile is expected to be warning-free after this dead-code removal.
