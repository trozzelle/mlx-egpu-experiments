"""Run-log tests for the Phase 0 harness (no GPU required)."""

import logging
from pathlib import Path

from tinygrad_kv_worker.harness import _close_run_logging, configure_run_logging


def test_close_run_logging_detaches_root_file_handler(tmp_path):
    """Closing a run log must leave no stale root FileHandler behind."""
    root = logging.getLogger()
    log_path = configure_run_logging(str(tmp_path), "unit")
    logging.getLogger("tinygrad_kv_worker.harness").info("unit log line")

    _close_run_logging()

    attached_handlers = [
        handler
        for handler in root.handlers
        if isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == Path(log_path)
    ]
    try:
        assert attached_handlers == []
    finally:
        # If this assertion fails against a broken implementation, keep the
        # global logging state clean for the rest of the test process.
        for handler in attached_handlers:
            root.removeHandler(handler)
            handler.close()

    assert "unit log line" in Path(log_path).read_text()



def test_configure_run_logging_uses_distinct_paths_for_same_second(tmp_path):
    """Each configure call should create a distinct reviewable run-log path."""
    path1 = configure_run_logging(str(tmp_path), "unit")
    _close_run_logging()
    path2 = configure_run_logging(str(tmp_path), "unit")
    _close_run_logging()

    assert path1 != path2