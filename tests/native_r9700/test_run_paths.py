"""RED contracts for isolated native R9700 generated-run directories."""

from datetime import datetime, timezone
from pathlib import Path
import re
import shutil

import pytest

from native_r9700 import run_paths



def test_run_root_uses_configured_environment_root(tmp_path, monkeypatch):
    configured_root = tmp_path / "configured-runs"
    monkeypatch.setenv("NATIVE_R9700_RUN_ROOT", str(configured_root))

    assert run_paths.run_root() == configured_root



def test_run_root_defaults_to_native_r9700_logs_directory(monkeypatch):
    monkeypatch.delenv("NATIVE_R9700_RUN_ROOT", raising=False)

    assert run_paths.run_root() == Path("logs/native-r9700-runs")



def test_new_run_dir_creates_a_utc_suffixed_label_under_the_run_root(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv("NATIVE_R9700_RUN_ROOT", str(root))
    before = datetime.now(timezone.utc).replace(microsecond=0)

    created = run_paths.new_run_dir("integration")

    after = datetime.now(timezone.utc).replace(microsecond=0)
    timestamp_match = re.fullmatch(r"integration-(\d{8}T\d{6}Z)", created.name)
    assert timestamp_match is not None
    timestamp = datetime.strptime(timestamp_match.group(1), "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    assert before <= timestamp <= after
    assert created.parent == root
    assert created.is_dir()
    assert list(root.iterdir()) == [created]



def test_new_run_dir_disambiguates_same_label_created_in_the_same_second(
    tmp_path, monkeypatch
):
    root = tmp_path / "runs"
    monkeypatch.setenv("NATIVE_R9700_RUN_ROOT", str(root))
    timestamp = datetime(2026, 8, 21, 12, 34, 56, tzinfo=timezone.utc)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return timestamp

    monkeypatch.setattr(run_paths, "datetime", FrozenDatetime)

    first = run_paths.new_run_dir("integration")
    second = run_paths.new_run_dir("integration")

    assert first != second
    assert first.is_dir()
    assert second.is_dir()
    assert set(root.iterdir()) == {first, second}


def test_new_run_dir_rejects_configured_root_inside_native_r9700(tmp_path, monkeypatch):
    native_r9700_directory = Path(run_paths.__file__).resolve().parent
    configured_root = native_r9700_directory / f".pytest-runs-{tmp_path.name}"
    assert not configured_root.exists()
    monkeypatch.setenv("NATIVE_R9700_RUN_ROOT", str(configured_root))

    try:
        with pytest.raises(ValueError):
            run_paths.new_run_dir("integration")

        assert not configured_root.exists()
    finally:
        shutil.rmtree(configured_root, ignore_errors=True)


@pytest.mark.parametrize("label", ["nested/run", r"nested\run"])
def test_new_run_dir_rejects_path_separator_labels(tmp_path, monkeypatch, label):
    root = tmp_path / "runs"
    monkeypatch.setenv("NATIVE_R9700_RUN_ROOT", str(root))

    with pytest.raises(ValueError, match=r"^label must not contain a path separator$"):
        run_paths.new_run_dir(label)

    assert not root.exists()
