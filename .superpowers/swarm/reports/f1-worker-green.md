# F1 public worker GREEN implementation

**Target:** `native_r9700/native_worker.py`

## Implemented symbols

- `dispatch_request(request, *, registry)` delegates decoded public requests to the existing `ModelRegistry`, projects the result to the seven-key `r9700_prefill_service_v1` envelope, removes child-only resource fields, and validates the public response through `service_protocol.encode_response`.
- `serve_forever(input_stream, output_stream, *, registry=None, native_runner=None, artifacts_dir="artifacts")` consumes local JSONL frames with `service_protocol.decode_request_frame`, emits bounded public predecode errors, dispatches valid requests through one registry, and closes that registry once on EOF, `KeyboardInterrupt`, or shutdown.
- `main(argv=None)` requires `--native-runner`, constructs exactly one `NativeResourceClient(runner_path=...)` and one `ModelRegistry(resource_client=..., artifact_dir=...)`, then hands the public streams to the persistent loop. Registry construction failure closes the client once.

The public loop does not create child processes, use `subprocess.run`, consult runner environment defaults, expose private pipes, or add a socket/network transport. The existing registry owns the client shutdown ordering (`Release`/teardown before `Shutdown`).

## Supervisor focused command

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_native_worker_evidence.py \
  tests/native_r9700/test_serving.py \
  tests/native_r9700/test_kv_cache.py \
  tests/native_r9700/test_prefill_phase_accounting.py -v
```

No validation command was run by this worker; supervisor owns focused execution.
