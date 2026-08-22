# Task9 kernel-assets FIFO RED contract

## Selector

- `tests/native_r9700/test_kernel_assets.py::test_file_backed_llama_kernel_assets_fail_closed_without_hardware`

## Added behavioral coverage

The C++ probe creates a no-writer FIFO named `fifo.code` as a direct child of the asset root with POSIX `mkfifo`. It then loads the manifest-relative direct-child path through the existing `rejects_without_output_mutation` helper. The helper requires a false result, a nonempty error, and byte-for-byte preservation of the sentinel output descriptor.

This is deliberately a direct child: lexical containment and `O_NOFOLLOW` do not prevent opening a FIFO. A blocking open without `O_NONBLOCK` waits forever for a writer, so the current loader is expected to hang at this probe case rather than return 28. The intended implementation opens the child with `O_NONBLOCK | O_NOFOLLOW`, `fstat`s that descriptor, and rejects the FIFO as non-regular before reading it.

## Supervisor RED command (do not run in this task)

Run the selector under Python's bounded subprocess timeout so a blocking loader is observed as a timeout rather than allowed to hang the supervisor:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -c 'import subprocess; subprocess.run(["${HOME}/.pyenv/versions/3.12.8/bin/python3", "-m", "pytest", "tests/native_r9700/test_kernel_assets.py::test_file_backed_llama_kernel_assets_fail_closed_without_hardware", "-q"], check=True, timeout=30)'
```

A timeout is the intended current RED observation; after the nonblocking descriptor implementation, the probe returns promptly and the selector passes.
