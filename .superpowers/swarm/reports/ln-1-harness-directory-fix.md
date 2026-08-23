# LN-1 publication harness directory fix

## Root cause

`run_case` created `.trace.staging` directly, but each supplied case root did not yet exist. The filesystem setup therefore failed before any publication success or fault assertion could execute.

## Harness change

Before creating staging, `run_case` now creates the case root with the error-code directory API. Root and staging setup failures both emit the affected path and filesystem error before returning false. The publication fault plans and assertions are unchanged.

## Validation

Not run, per assignment.
