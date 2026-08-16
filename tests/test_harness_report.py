"""Validation-report writer tests (no GPU required)."""

from tinygrad_kv_worker.harness import write_validation_report


def test_write_validation_report_records_pass_context(tmp_path):
    out = tmp_path / "report.md"

    write_validation_report(
        str(out),
        {
            "run_log": "logs/runs/pass.log",
            "gguf": "mlx_models/model.F16.gguf",
            "mlx": "mlx_models/model",
            "prompts": [
                {"prompt_name": "prompt-0", "S": 6, "exact_match": True},
                {"prompt_name": "prompt-1", "S": 222, "exact_match": True},
            ],
            "per_layer": [
                {
                    "max_K": 0.004,
                    "mean_K": 0.0002,
                    "max_V": 0.0001,
                    "mean_V": 0.00001,
                    "over_tolerance": True,
                }
            ],
            "flagged_layers": [0],
            "rope_config": {"scaling": {"rope_type": "llama3"}},
        },
    )

    text = out.read_text()
    assert "RUN COMPLETED — GATE PASSED" in text
    assert "logs/runs/pass.log" in text
    assert "official fp16" in text
    assert "`S-1`" in text
    assert "Llama-3 RoPE scaling" in text
    assert "token gate passed" in text
