"""tinygrad_kv_worker — Phase 0 prefill-exporter core.

Pure-CPU bridge that turns per-block tinygrad KV cache tensors into an
mlx-lm prompt-cache ``.safetensors`` file. No tinygrad GPU runtime, no AMD
device, no model evaluation — numpy in, file path out.
"""

from .exporter import export_prompt_cache

__all__ = ["export_prompt_cache"]
