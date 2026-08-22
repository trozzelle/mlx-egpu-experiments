"""C2 serving-wrapper delegation tests for the legacy harness CLI."""

from types import SimpleNamespace

from tinygrad_kv_worker import harness


def _completed(returncode: int = 0):
    return SimpleNamespace(returncode=returncode)


def test_c2_serving_mode_delegates_to_native_r9700_serving_without_path_a_weights(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check=False):
        calls.append((list(cmd), check))
        return _completed(7)

    monkeypatch.setattr(harness, "subprocess", SimpleNamespace(run=fake_run), raising=False)

    rc = harness.main(
        [
            "--c2-serving",
            "--model",
            "consumer-model",
            "--producer-model",
            "producer-model",
            "--fixtures-dir",
            "tests/native_r9700/fixtures",
            "--prompt-name",
            "prompt-0",
            "--threshold-tokens",
            "2",
            "--max-new-tokens",
            "4",
            "--producer-timeout-s",
            "9",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--json",
            str(tmp_path / "result.json"),
            "--log",
            str(tmp_path / "run.log"),
            "--report",
            str(tmp_path / "report.md"),
        ]
    )

    assert rc == 7
    assert len(calls) == 1
    cmd, check = calls[0]
    assert check is False
    assert cmd[:3] == [harness.sys.executable, "-m", "native_r9700.serving"]
    assert "--gguf" not in cmd
    assert "--mlx" not in cmd
    for flag, value in (
        ("--model", "consumer-model"),
        ("--producer-model", "producer-model"),
        ("--fixtures-dir", "tests/native_r9700/fixtures"),
        ("--prompt-name", "prompt-0"),
        ("--threshold-tokens", "2"),
        ("--max-new-tokens", "4"),
        ("--producer-timeout-s", "9"),
        ("--artifacts-dir", str(tmp_path / "artifacts")),
        ("--json", str(tmp_path / "result.json")),
        ("--log", str(tmp_path / "run.log")),
        ("--report", str(tmp_path / "report.md")),
    ):
        assert cmd[cmd.index(flag) + 1] == value


def test_c2_serving_mode_passes_through_r9700_native_fail_closed_request(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check=False):
        calls.append(list(cmd))
        return _completed(2)

    monkeypatch.setattr(harness, "subprocess", SimpleNamespace(run=fake_run), raising=False)

    rc = harness.main(
        [
            "--c2-serving",
            "--model",
            "consumer-model",
            "--token-ids-json",
            "[10, 11, 12, 13]",
            "--producer-kind",
            "r9700_native",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--json",
            str(tmp_path / "result.json"),
            "--log",
            str(tmp_path / "run.log"),
        ]
    )

    assert rc == 2
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:3] == [harness.sys.executable, "-m", "native_r9700.serving"]
    assert cmd[cmd.index("--producer-kind") + 1] == "r9700_native"
    assert cmd[cmd.index("--token-ids-json") + 1] == "[10, 11, 12, 13]"

def test_c2_serving_mode_leaves_wrapper_defaults_when_max_new_tokens_absent(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check=False):
        calls.append(list(cmd))
        return _completed(0)

    monkeypatch.setattr(harness, "subprocess", SimpleNamespace(run=fake_run), raising=False)

    rc = harness.main(
        [
            "--c2-serving",
            "--model",
            "consumer-model",
            "--prompt",
            "hello",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--json",
            str(tmp_path / "result.json"),
            "--log",
            str(tmp_path / "run.log"),
        ]
    )

    assert rc == 0
    assert len(calls) == 1
    assert "--max-new-tokens" not in calls[0]
