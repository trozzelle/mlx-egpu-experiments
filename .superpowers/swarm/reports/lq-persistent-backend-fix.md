# Persistent backend fix

- Armed `TerminalComputeQueue0Retirement` before compute-ring setup and routed setup failure through terminal retirement before resident-memory cleanup.
- Corrected resident prepare and `upload_named` chunk uploads to submit every chunk from the fixed staging GPU VA while advancing only the destination GPU VA.
- Preserved chunk fences, queue setup, error propagation, and byte transport.
- No commands or hardware/tests were run per assignment constraint.
