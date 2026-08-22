# C1R-W1 fixture catalog RED contract

## Delivered

- Added `tests/native_r9700/test_fixture_catalog.py`.
- The contract requires a frozen, uniquely named catalog with stable lookup and no duplicate array declarations.
- It requires the catalog to cover the committed NPZ archive set and validates each declared archive's exact array set, per-array shape and dtype, and SHA-256 digest.

## Validation

The assignment prohibits commands, so the RED test was not run. It intentionally imports the not-yet-created `native_r9700.fixture_catalog` module and will remain RED until the catalog implementation is delivered.
