# Task9 live-binding contract correction

- Valid safetensors spans now reject only an offset preceding the payload; equality is valid for the first tensor in a shard.
- Removed the premature `complete-named-llama-assets` probe mode and test. The manifest is intentionally empty until Task9-3 supplies real stage assets; `missing-stage-assets` continues to cover fail-closed behavior.
- No production files, stage assets, or manifest files were changed.

Supervisor verification (not run):

```sh
python -m pytest tests/native_r9700/test_model_weight_binder_contract.py tests/native_r9700/test_layer0_executor_contract.py
```
