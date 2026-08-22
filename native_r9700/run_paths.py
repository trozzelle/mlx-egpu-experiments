"""Generated-run directory helpers for the native R9700 producer."""

import os
from datetime import datetime, timezone
from pathlib import Path


_DEFAULT_RUN_ROOT = Path("logs/native-r9700-runs")


def run_root() -> Path:
    """Return the configured generated-run root without creating it."""
    return Path(os.environ.get("NATIVE_R9700_RUN_ROOT", _DEFAULT_RUN_ROOT))


def new_run_dir(label: str) -> Path:
    """Create and return a UTC-suffixed generated-run directory for *label*."""
    if "/" in label or "\\" in label:
        raise ValueError("label must not contain a path separator")
    root = run_root()
    try:
        root.resolve().relative_to(Path(__file__).resolve().parent)
    except ValueError:
        pass
    else:
        raise ValueError("run root must not be inside the native_r9700 package directory")


    stem = f"{label}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    for index in range(1_000):
        suffix = "" if index == 0 else f"-{index}"
        directory = root / f"{stem}{suffix}"
        try:
            directory.mkdir(parents=True)
        except FileExistsError:
            continue
        return directory

    raise FileExistsError(
        f"could not create a unique run directory for {label!r} after 1000 attempts"
    )
