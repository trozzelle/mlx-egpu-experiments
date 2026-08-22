"""native_r9700 - C1 native R9700 producer package.

Lane A owns the native runtime shell (``runtime.h``/``runtime.cpp``); Lane B
owns the weight/config narrow loader (``config.py``/``loader.py``). Python
helpers are imported explicitly (``import native_r9700.loader``) rather than
re-exported here, keeping the C++-first package marker importable without
side effects. No tinygrad dependency in the producer path.
"""
