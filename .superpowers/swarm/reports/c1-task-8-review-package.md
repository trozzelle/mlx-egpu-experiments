# C1 task 8 review package

## git status
```text
 M .superpowers/swarm/native-r9700-producer-supervisor.md
 M docs/tasks/native-r9700-producer/validation-commands.md
 M native_r9700/prefill.py
 M tests/native_r9700/test_prefill.py
?? .superpowers/swarm/handoff-C1-after-wave2.md
?? .superpowers/swarm/reports/c1-task-8-kv-emitter-red.md
?? .superpowers/swarm/reports/c1-task-8-kv-emitter.md
?? docs.zip
?? docs/research/
?? docs/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-handoff.md
?? native_r9700/kv_cache.py
?? tests/native_r9700/test_kv_cache.py
```

## git diff tracked
```diff
diff --git a/.superpowers/swarm/native-r9700-producer-supervisor.md b/.superpowers/swarm/native-r9700-producer-supervisor.md
index e4f70d1..8fc5022 100644
--- a/.superpowers/swarm/native-r9700-producer-supervisor.md
+++ b/.superpowers/swarm/native-r9700-producer-supervisor.md
@@ -620,3 +620,25 @@ Prove the smallest tinygrad-free macOS R9700 kernel dispatch/readback path on th
 - Review agents: `C1PrefillReview` approved with 0 Critical, 0 Important, 0 Minor.
 - Verification command(s) supervisor ran: RED focused pytest exited 1 with 5 expected missing-module/API failures; focused GREEN `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py -v` exited 0 with 5 passed; CLI `python -m native_r9700.prefill ... --out logs/c1-prefill-prompt0.npz --log logs/c1-prefill-prompt0.log` exited 0 and logged layer0/layer15 deltas; combined `tests/native_r9700 -v` exited 0 with 71 passed; full `tests -v` exited 0 with 111 passed, 2 warnings; `git diff --check` printed no output.
 - Ledger update: C1-7 Done; C1-8 is unblocked and In progress.
+
+## Wave 23: C1 KV interchange emitter
+### Shared context
+- Goal: implement C1 task set 8 by serializing the C1-7 native prefill arrays into an mlx-lm-loadable prompt-cache `.safetensors` file.
+- Constraints: shared work boundary `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every executor/reviewer stays in this cwd/branch. Producer/emitter production path remains tinygrad-free and should not need an MLX runtime import; tests may use mlx-lm to round-trip the emitted file. Do not implement C1 task set 9 parity/decode, C2 integration, Qwen support, or C++ runtime changes in this wave.
+- ABI from `C1EmitterScout`: safetensors tensor keys are `{i}.0` for K and `{i}.1` for V; metadata keys are `0.{i}=""`, `2.{i}="KVCache"`, global `1.offset=str(N)`, `1.num_layers="16"`, `1.n_kv_heads="8"`, `1.head_dim="64"`; `N` is the S-1 prefix length, not full prompt S.
+- TDD policy: supervisor observes focused RED tests before production C1 task-set-8 code. OMP executor agents do not run tests, linters, formatters, package managers, hardware commands, or git commands; supervisor verifies after the wave.
+- Reports: `.superpowers/swarm/reports/c1-task-8-kv-emitter.md` plus RED/review reports and `agent://C1EmitterScout`.
+
+### Agents
+| Agent | Task row | Target | Depends on | Report | Status |
+|---|---|---|---|---|---|
+| C1EmitterScout | C1-8 prep | mlx-lm prompt-cache ABI research | C1-7 API contract | `agent://C1EmitterScout` | Done |
+| C1EmitterRed | C1-8 RED | Focused emitter tests and validation-command row | C1-7 Done | `.superpowers/swarm/reports/c1-task-8-kv-emitter-red.md` | Planned |
+| C1EmitterImpl | C1-8 GREEN | `native_r9700/kv_cache.py` safetensors emitter and CLI | RED observed | `.superpowers/swarm/reports/c1-task-8-kv-emitter.md` | Planned |
+| C1EmitterReview | C1-8 review | Task-scoped correctness/quality review | GREEN verified | `.superpowers/swarm/reports/c1-task-8-kv-emitter-review.md` | Planned |
+
+### Supervisor gates
+- Report checks: scout/RED/implementation/review reports agree on exact mlx-lm safetensors tensor keys, metadata keys, S-1 offset semantics, no casting/repair of bad producer arrays, and no fake Qwen support.
+- Quality bar: correctness via safetensors header checks and mlx-lm load round-trip; maintainability via one narrow emitter module; architectural fit via existing KV interchange contract; simplicity via no duplicate generic exporter framework.
+- Verification command(s) supervisor will run: focused C1-8 RED/GREEN pytest command recorded in `validation-commands.md`, CLI smoke converting a local prefill NPZ into safetensors and loading it, combined `tests/native_r9700 -v`, full `tests -v`, and `git diff --check`.
+- Ledger update: C1-8 In progress while RED/GREEN/review run; dependent C1-9 remains blocked until C1-8 review/verification passes.
diff --git a/docs/tasks/native-r9700-producer/validation-commands.md b/docs/tasks/native-r9700-producer/validation-commands.md
index 9cdf698..fb58fe6 100644
--- a/docs/tasks/native-r9700-producer/validation-commands.md
+++ b/docs/tasks/native-r9700-producer/validation-commands.md
@@ -253,7 +253,7 @@ git diff --check
 | C1 | primitive kernel test commands | `phase-c1-native-producer-parity.md` task set 5 | C0 substrate SELECTED (macOS TinyGPU/AMDev native, C0A25); C1 command discovery now in scope under a C1 contract-freeze plan; not yet executed |
 | C1 | attention/RoPE/KV writer test command | `phase-c1-native-producer-parity.md` task set 6 | Exact focused RED/GREEN command recorded below; supervisor expects RED until `native_r9700.attention` implements the frozen Llama-only API and KV parity contract |
 | C1 | full-stack native prefill smoke command | `phase-c1-native-producer-parity.md` task set 7 | Exact focused RED/GREEN command recorded below; supervisor expects RED until `native_r9700.prefill` implements the full-layer Llama prefix prefill API and CLI |
-| C1 | native KV emitter/load round-trip command | `phase-c1-native-producer-parity.md` task set 8 | C0 substrate SELECTED (macOS TinyGPU/AMDev native, C0A25); C1 command discovery now in scope under a C1 contract-freeze plan; not yet executed |
+| C1 | native KV emitter/load round-trip command | `phase-c1-native-producer-parity.md` task set 8 | Exact focused RED/GREEN command recorded below; supervisor expects RED until `native_r9700.kv_cache` implements the prompt-cache safetensors emitter API/CLI |
 | C1 | native producer parity command | `phase-c1-native-producer-parity.md` task set 9 | C0 substrate SELECTED (macOS TinyGPU/AMDev native, C0A25); C1 command discovery now in scope under a C1 contract-freeze plan; not yet executed |
 | C2 | mlx-lm wrapper focused test command | `phase-c2-serving-integration.md` task set 1 or 2 | Dependency-blocked by C1 parity; not discovered |
 | C2 | fallback/error-state test command | `phase-c2-serving-integration.md` task set 1 or 3 | Dependency-blocked by C1 parity; not discovered |
@@ -452,6 +452,24 @@ Observed: exit `0`; log includes `n_prefix: 5`, `num_layers: 16`,
 `output: logs/c1-prefill-prompt0.npz`, layer0/layer15 K/V max/mean deltas,
 and `exit_status: 0`.
 
+### C1 native KV prompt-cache emitter contract (task set 8)
+
+Focused RED/GREEN contract tests for the future `native_r9700.kv_cache` module.
+The tests lock the mlx-lm prompt-cache safetensors ABI for C1 prefill results:
+16 ordered `KVCache` layers, tensor keys `{i}.0`/`{i}.1`, metadata keys
+`0.{i}`, `2.{i}`, `1.offset`, `1.num_layers`, `1.n_kv_heads`, and
+`1.head_dim`, fixture NPZ conversion, loud validation failures for malformed
+K/V arrays, and CLI conversion/logging from a prefill NPZ. Qwen support,
+decode/parity-harness wiring, C2 integration, and C++ runtime integration
+remain outside this task set.
+
+```sh
+${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_kv_cache.py -v
+```
+
+Expected RED before implementation: collection succeeds and the focused command
+fails with a clear missing `native_r9700.kv_cache` module/API failure.
+
 ### C1 reference fixtures (Lane B2 — task set 3)
 
 The `native_r9700/ref_fixtures.py` module (Lane B2, marker `c1w2-lane-b2`)
diff --git a/native_r9700/prefill.py b/native_r9700/prefill.py
index 2e55bdc..a9b0e79 100644
--- a/native_r9700/prefill.py
+++ b/native_r9700/prefill.py
@@ -358,7 +358,11 @@ def _delta_reports(result: Mapping[str, object], fixture_path: str) -> list[str]
     reports: list[str] = []
     for layer_index in _EXPECTED_DELTA_LAYERS:
         layer = _layer_result(result, layer_index)
-        deltas = compare_layer_kv_to_fixture(layer, fixture_path, layer_index=layer_index)
+        try:
+            deltas = compare_layer_kv_to_fixture(layer, fixture_path, layer_index=layer_index)
+        except ValueError as exc:
+            reports.append(f"layer={layer_index} fixture incompatible: {exc}")
+            continue
         reports.append(format_layer_kv_delta_report(deltas))
     return reports
 
diff --git a/tests/native_r9700/test_prefill.py b/tests/native_r9700/test_prefill.py
index 2e8db35..9269ccc 100644
--- a/tests/native_r9700/test_prefill.py
+++ b/tests/native_r9700/test_prefill.py
@@ -183,6 +183,66 @@ def test_prefill_cli_writes_full_layer_npz_and_review_log(tmp_path):
     assert "exit_status: 0" in log_text
 
 
+def test_prefill_cli_logs_fixture_shape_mismatch_without_failing(tmp_path, monkeypatch):
+    prefill = _prefill_module()
+    fixtures_dir = tmp_path / "fixtures"
+    fixtures_dir.mkdir()
+    (fixtures_dir / "prompts.json").write_text(
+        json.dumps({"prompt-x": {"S": 7, "token_ids": [1, 2, 3, 4, 5, 6, 7]}}),
+        encoding="utf-8",
+    )
+    np.savez(
+        fixtures_dir / "kv_state.npz",
+        layer0_K=np.zeros((1, 8, 5, 64), dtype=np.float16),
+        layer0_V=np.zeros((1, 8, 5, 64), dtype=np.float16),
+        layer15_K=np.zeros((1, 8, 5, 64), dtype=np.float16),
+        layer15_V=np.zeros((1, 8, 5, 64), dtype=np.float16),
+    )
+
+    def fake_prefill_prompt_prefix(model_dir, prefix_token_ids):
+        assert model_dir == "synthetic-model"
+        assert prefix_token_ids == [1, 2, 3, 4, 5, 6]
+        layers = [
+            {
+                "layer": layer_index,
+                "K": np.zeros((1, 8, 6, 64), dtype=np.float16),
+                "V": np.zeros((1, 8, 6, 64), dtype=np.float16),
+            }
+            for layer_index in range(16)
+        ]
+        return {
+            "model": "synthetic-model",
+            "config_path": "synthetic-config.json",
+            "n_prefix": 6,
+            "layers": layers,
+        }
+
+    monkeypatch.setattr(prefill, "prefill_prompt_prefix", fake_prefill_prompt_prefix)
+    out_path = tmp_path / "prefill.npz"
+    log_path = tmp_path / "prefill.log"
+
+    rc = prefill.main(
+        [
+            "--model",
+            "synthetic-model",
+            "--fixtures-dir",
+            str(fixtures_dir),
+            "--prompt-name",
+            "prompt-x",
+            "--out",
+            str(out_path),
+            "--log",
+            str(log_path),
+        ]
+    )
+
+    assert rc == 0
+    assert out_path.is_file()
+    log_text = log_path.read_text(encoding="utf-8")
+    assert "fixture incompatible" in log_text
+    assert "K shape (1, 8, 6, 64) != fixture shape (1, 8, 5, 64)" in log_text
+    assert "exit_status: 0" in log_text
+
 @pytest.mark.parametrize("prefix_token_ids", [[], [128000]])
 def test_prefill_prompt_prefix_rejects_prefixes_shorter_than_two_tokens(prefix_token_ids):
     prefill = _prefill_module()
```

## C1EmitterScout summary

{
  "summary": "C1-8 should emit the existing mlx-lm prompt-cache safetensors tree exactly: per layer `i`, tensor key `\"i.0\"` is K and `\"i.1\"` is V, both fp16 `(1,8,N,64)` for Llama 3.2 1B; metadata key `\"0.i\"` is empty per-layer `meta_state`, `\"2.i\"` is `\"KVCache\"`, and global metadata lives under `\"1.*\"` with at least `offset=str(N)`, `num_layers=\"16\"`, `n_kv_heads=\"8\"`, `head_dim=\"64\"`. Map C1-7 `prefill_prompt_prefix(...)` output directly: ordered layer dicts become layers `0..15`, `N = result[\"n_prefix\"] = S-1`, and `offset` must be `N`, not full prompt `S`. Do not use fixture-style `layer0_K` keys in the safetensors file.",
  "files": [
    {
      "path": "${HOME}/.pyenv/versions/3.12.8/lib/python3.12/site-packages/mlx_lm/models/cache.py",
      "description": "Primary upstream ABI: `make_prompt_cache` default list of `KVCache` (`:14-43`); `save_prompt_cache`/`load_prompt_cache` flatten and rebuild cache arrays/classes/metadata (`:42-93`); `_BaseCache.meta_state` empty-string behavior and truthy-value rejection (`:137-157`); `KVCache` state/offset/update behavior (`:325-381`)."
    },
    {
      "path": "${HOME}/.pyenv/versions/3.12.8/lib/python3.12/site-packages/mlx/utils.py",
      "description": "Explains exact flattened key naming for lists/tuples/dicts (`0.0`, `1.offset`, etc.) and unflattening (`:117-241`). This is what turns `[c.state for c in cache]` into safetensors tensor keys."
    },
    {
      "path": "${HOME}/.pyenv/versions/3.12.8/lib/python3.12/site-packages/mlx_lm/generate.py",
      "description": "`generate_step` always processes supplied prompt and mutates a provided prompt cache (`:306-423`, `:423-453`), proving the S-1 cache + final-token suffix contract. CLI prompt-cache mode additionally expects `model` and `tokenizer_config` metadata (`:1969-2009`)."
    },
    {
      "path": "${HOME}/.pyenv/versions/3.12.8/lib/python3.12/site-packages/mlx_lm/cache_prompt.py",
      "description": "Upstream CLI cache writer adds optional global `model` and `tokenizer_config` metadata before `save_prompt_cache` (`:110-148`)."
    },
    {
      "path": "docs/DESIGN.md",
      "description": "Project contract for KV interchange: `.safetensors`, per-layer `KVCache`, empty per-layer `meta_state`, global `offset`, Llama shape `(1,8,N,64)`, fp16, temporal/RoPE semantics, exporter steps (`:41-79`)."
    },
    {
      "path": "docs/pinned-upstream-interfaces.md",
      "description": "Pinned mlx-lm ABI summary: KVCache shapes, empty meta_state requirement, global metadata fields, `load_prompt_cache` rebuild, and S-1 `generate_step` seam (`:49-82`)."
    },
    {
      "path": "docs/tasks/native-r9700-producer/phase-c1-native-producer-parity.md",
      "description": "C1 frozen output contract and task set 8 acceptance: native output prompt cache, metadata fields, load with `load_prompt_cache`, preserve `N == S-1`, fail before partial output (`:67-83`, `:308-342`)."
    },
    {
      "path": "tinygrad_kv_worker/exporter.py",
      "description": "Existing round-trippable exporter implementation: validates tinygrad block cache count/shape/dtype/S, slices/splits/casts fp16, builds `KVCache` via state setter, writes metadata and atomic temp file (`:1-40`, `:58-188`, `:194-257`)."
    },
    {
      "path": "tests/test_exporter.py",
      "description": "Existing behavioral tests for the schema: round-trip through `load_prompt_cache`, KVCache class, shapes/dtypes/offset, global metadata, and fail-loud/no-output cases (`:1-178`, `:174-184`)."
    },
    {
      "path": "tests/native_r9700/test_prefill.py",
      "description": "C1-7 native prefill result contract: top-level `model`, `n_prefix`, ordered `layers`; each layer has `layer`, fp16 K/V arrays shaped `(1,8,N,64)`; CLI currently emits fixture-style NPZ (`:1-153`, `:135-192`)."
    },
    {
      "path": "tests/native_r9700/test_ref_fixtures.py",
      "description": "Committed fixture schema for prompt-0 S-1 prefix: `n_prefix=5`, 16 layers, fixture NPZ keys `layer{i}_K/V`, each `(1,8,5,64)` fp16 (`:133-156`)."
    },
    {
      "path": "native_r9700/ref_fixtures.py",
      "description": "Shows fixture generation source: native mlx-lm baseline cache is sliced to `S-1`, stored fp16 with shape `(1,8,n_prefix,64)`, final token excluded (`:180-273`)."
    },
    {
      "path": "native_r9700/config.py",
      "description": "Frozen C1 Llama-only geometry constants and unsupported-model policy: 16 layers, 8 KV heads, head_dim 64, hidden 2048, Llama-3 RoPE sidecar (`:27-44`)."
    }
  ],
  "architecture": "Implementation handoff for C1-8 KV emitter\n\n1. Exact safetensors tensor keys\n\n`mlx_lm.models.cache.save_prompt_cache` builds `cache_data = [c.state for c in cache]`, then `dict(tree_flatten(cache_data))`, then writes that dict with `mx.save_safetensors` (`cache.py:43-59`). `tree_flatten` uses numeric dot notation for lists/tuples; its own example says `[[[0]]] -> \"0.0.0\"` (`mlx/utils.py:118-190`). Therefore the mlx-lm-loadable tensor keys for 16 standard layers are:\n\n- Layer `i` K: tensor key `\"{i}.0\"`.\n- Layer `i` V: tensor key `\"{i}.1\"`.\n- For C1 Llama 3.2 1B: each value is fp16/F16 shape `(1, 8, N, 64)`, where `N = n_prefix = S-1` for `generate_step` injection.\n- Examples: `\"0.0\"` = layer 0 K, `\"0.1\"` = layer 0 V, `\"15.0\"` = layer 15 K, `\"15.1\"` = layer 15 V.\n\nDo not put `layer0_K`, `layer0_V`, etc. in the safetensors prompt cache. Those names are the committed fixture NPZ schema only (`tests/native_r9700/test_ref_fixtures.py:133-156`; `native_r9700/ref_fixtures.py:257-262`).\n\n2. Exact safetensors metadata keys and values\n\n`save_prompt_cache` builds metadata as `[cache_info, metadata, cache_classes]`, where `cache_info = [c.meta_state]`, `metadata` is caller global metadata, and `cache_classes = [type(c).__name__]`; it then flattens that tree (`cache.py:53-59`). Thus manual safetensors writers must reproduce these flattened metadata keys:\n\n- Per layer `i`: `\"0.{i}\" = \"\"` (empty per-layer `meta_state`). This must not be `str(N)`. Standard `KVCache` inherits `_BaseCache.meta_state`, which returns `\"\"` and whose setter raises on any truthy value (`cache.py:137-157`); `load_prompt_cache` calls `KVCache.from_state(state, meta_state)` (`cache.py:62-85`), so non-empty per-layer meta_state can break load.\n- Per layer `i`: `\"2.{i}\" = \"KVCache\"`.\n- Required C1 global shape/offset metadata, flattened under outer metadata index 1: `\"1.offset\" = str(N)`, `\"1.num_layers\" = \"16\"`, `\"1.n_kv_heads\" = \"8\"`, `\"1.head_dim\" = \"64\"`. When read with `load_prompt_cache(path, return_metadata=True)`, these come back unflattened as `metadata[\"offset\"]`, etc. (`cache.py:62-85`; `docs/pinned-upstream-interfaces.md:69-75`). All safetensors metadata values should be strings.\n- Model metadata is not required by `load_prompt_cache` itself. Upstream `mlx_lm.cache_prompt` adds `metadata[\"model\"]` and `metadata[\"tokenizer_config\"]` (`cache_prompt.py:142-145`), and the mlx-lm generate CLI assumes those keys when `--prompt-cache-file` is used (`generate.py:1991-2004`). The existing project exporter does not write them; it writes only `offset`, `num_layers`, `n_kv_heads`, and `head_dim` (`tinygrad_kv_worker/exporter.py:174-184`). For C1-8, require the shape/offset fields; add `model`/`config_path`/`tokenizer_config` only if the emitter explicitly claims direct mlx-lm CLI `--prompt-cache-file` compatibility.\n\n3. Native prefill result mapping\n\nC1-7 is expected to return `prefill_prompt_prefix(model_dir, prefix_token_ids)` as `{\"model\": ..., \"n_prefix\": N, \"layers\": [...]}` with 16 ordered layer dicts. The current RED tests lock that each layer has `layer == index` and K/V arrays are fp16 `(1,8,N,64)` (`tests/native_r9700/test_prefill.py:87-102`). Map it as:\n\n```text\nN = result[\"n_prefix\"]\nassert N > 0\nassert len(result[\"layers\"]) == 16\nfor i, layer in enumerate(result[\"layers\"]):\n    assert layer[\"layer\"] == i\n    K = np.asarray(layer[\"K\"]); V = np.asarray(layer[\"V\"])\n    assert K.dtype == V.dtype == np.float16\n    assert K.shape == V.shape == (1, 8, N, 64)\n    tensors[f\"{i}.0\"] = contiguous K\n    tensors[f\"{i}.1\"] = contiguous V\n    metadata[f\"0.{i}\"] = \"\"\n    metadata[f\"2.{i}\"] = \"KVCache\"\nmetadata[\"1.offset\"] = str(N)\nmetadata[\"1.num_layers\"] = \"16\"\nmetadata[\"1.n_kv_heads\"] = \"8\"\nmetadata[\"1.head_dim\"] = \"64\"\n```\n\nIf the implementation chooses to use mlx-lm writer instead of manual `safetensors.numpy.save_file`, construct `KVCache()` per layer and assign `layer_cache.state = (mx.array(K, dtype=mx.float16), mx.array(V, dtype=mx.float16))`; the state setter reconstructs `offset = keys.shape[2]` (`cache.py:363-374`) and `save_prompt_cache` writes the exact flattened schema above. Existing Path A does this in `_build_kv_cache_round_trippable` (`tinygrad_kv_worker/exporter.py:202-223`). If avoiding MLX runtime in the native producer package, manual `safetensors` writing is fine as long as the flattened keys match.\n\n4. S/N injection semantics\n\nFor C1-8, `N` must be `S-1`, not full prompt `S`, when the file is intended for mlx-lm `generate_step`. The design says the imported cache contains the prompt prefix and the final prompt token is passed as the one-token suffix (`docs/DESIGN.md:47-54`), task set 8 explicitly says preserve `N == S-1` (`phase-c1-native-producer-parity.md:327-332`), and `generate_step` always processes the supplied prompt: it pre-fills `prompt[:-1]` then steps `prompt[-1]` (`generate.py:423-453`). The Phase 0 harness has the same guard and documents that full `S` cache plus full prompt duplicates the prompt (`tinygrad_kv_worker/harness.py:584-625`).\n\n5. Existing exporter behavior to mirror or adapt\n\n`tinygrad_kv_worker.exporter.export_prompt_cache` is the reference for round-trippable files, but its input is Path A-specific: `[2,B,n_kv_heads,max_context,head_dim]` fp32 block caches. It validates positive `S`, geometry, layer count, rank, axis-0 K/V size, dtype, batch consistency; slices `[..., :S, :]`; splits K/V; casts to fp16; builds `KVCache`; records global metadata; and writes via temp file then `os.replace` (`tinygrad_kv_worker/exporter.py:58-188`, `:194-257`). For native C1-8, do not inherit the tinygrad stacked fp32 input contract; consume the C1-7 native layer dicts already shaped `(1,8,N,64)` fp16. Existing tests prove the exporter round-trips and checks metadata/failure behavior (`tests/test_exporter.py:89-178`).\n\n6. Required failure modes for C1-8\n\nFail loudly before final output exists, and never silently repair an accepted producer cache:\n\n- Wrong dtype: any K/V not `np.float16` should raise. Do not silently cast native fp32/fp64 to fp16; C1-7 already promises fp16 (`tests/native_r9700/test_prefill.py:96-100`). Path A casts because tinygrad source is fp32 by contract; native emitter should catch drift.\n- Wrong shape/rank: reject anything except K and V both `(1,8,N,64)` for C1 Llama; reject mismatched K/V shapes, wrong batch, wrong heads, wrong head_dim, rank != 4, or `N <= 0`.\n- Wrong layers/order: reject `len(layers) != 16`, duplicate/missing layer indices, out-of-order `layer` fields, or layer values outside `0..15`.\n- Wrong offset/S: reject top-level `n_prefix` not equal to array temporal dimension, per-layer `n_prefix` if present and inconsistent, and metadata offset not equal to `N`. For injection, reject attempts to label the output as full prompt `S` when it only contains/should contain `S-1`.\n- Unsupported model/geometry: remain Llama-only in C1; do not broaden to Qwen3.8-27B. Task 6/7 docs explicitly defer Qwen (`validation-commands.md:386-431`; `test_attention_kv.py:7-10`; `test_prefill.py:7-10`), and `native_r9700/config.py:27-44` freezes 16/8/64 Llama geometry.\n- Bad writer/partial output: validate all layer arrays and metadata before writing; write to a temp sibling and atomically replace final path; cleanup temp on error, matching existing exporter (`tinygrad_kv_worker/exporter.py:231-257`).\n\n7. Recommended RED tests and round-trip command\n\nCreate focused tests under `tests/native_r9700/test_kv_cache.py` (or `test_emitter.py`) and record this exact command in `docs/tasks/native-r9700-producer/validation-commands.md` when implementing:\n\n```sh\ncd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_kv_cache.py -v\n```\n\nExpected RED before implementation: collection succeeds and fails with a clear missing `native_r9700.kv_cache` module/API or missing emitter function. Expected GREEN: all synthetic/fixture-backed tests pass without hardware. Model-backed integration may skip only if the local MLX model path is absent, following the task 6/7 pattern (`test_attention_kv.py:218-268`; `test_prefill.py:135-192`).\n\nSuggested tests:\n\n- `test_emits_mlx_loadable_prompt_cache_from_native_prefill_result`: build a small deterministic C1-7-shaped result with 16 layers, `N=5`, K/V fp16 `(1,8,5,64)`, call emitter, then `load_prompt_cache(out, return_metadata=True)`. Assert 16 `KVCache` instances, each `offset == 5`, K/V dtype `mx.float16`, shape `(1,8,5,64)`, and array equality to the input. Assert returned metadata `offset`, `num_layers`, `n_kv_heads`, `head_dim`.\n- `test_safetensors_header_uses_mlx_flattened_keys`: open with `safetensors.safe_open` and assert tensor keys are exactly `{f\"{i}.0\", f\"{i}.1\" for i in 0..15}`; assert metadata has `0.i == \"\"`, `2.i == \"KVCache\"`, and `1.offset/1.num_layers/1.n_kv_heads/1.head_dim`. This catches accidental `layer{i}_K` schema drift.\n- `test_fixture_npz_layers_round_trip`: read `tests/native_r9700/fixtures/kv_state.npz`, adapt its `layer{i}_K/V` arrays into a C1-7 result, emit safetensors, load with mlx-lm, assert exact equality. This uses committed fixture data and no model/hardware (`tests/native_r9700/test_ref_fixtures.py:133-156`).\n- `test_rejects_wrong_dtype_without_casting`: mutate one K or V to fp32/float64; expect `ValueError` and no final `.safetensors`.\n- `test_rejects_wrong_shape`: parameterize rank drop, wrong heads `(1,7,N,64)`, wrong head_dim `(1,8,N,128)`, mismatched K/V temporal dimension, batch `2`; expect failure and no output.\n- `test_rejects_wrong_layer_count_or_order`: 15 layers, duplicate index, layer field out of order, missing layer 15; expect failure and no output.\n- `test_rejects_n_prefix_offset_mismatch`: top-level `n_prefix` differs from `K.shape[2]`, `n_prefix == 0`, or per-layer `n_prefix` mismatch if supported; expect failure and no output.\n- `test_writer_failure_leaves_no_partial_artifact`: monkeypatch the low-level save function to raise after validation; final path must not exist and temp should be cleaned best-effort.\n\nOptional CLI smoke if task 8 exposes a CLI: consume the task-7 NPZ output shape and write `.safetensors`, then load it. Keep it separate from the pure function tests so C1-8 remains hardware/model-light."
}

## File: `native_r9700/kv_cache.py`

```text
"""Emit C1 Llama prompt-prefill K/V state as an mlx-lm prompt cache.

This module is deliberately narrow: it accepts the C1 Llama-3.2-1B prefill
result shape produced by :mod:`native_r9700.prefill` and writes the safetensors
prompt-cache ABI consumed by ``mlx-lm``.  It has no tinygrad or MLX runtime
production dependency.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file

_NUM_LAYERS = 16
_N_KV_HEADS = 8
_HEAD_DIM = 64
_BATCH = 1
_EXPECTED_SUFFIX = ".safetensors"


class KVCacheError(ValueError):
    """Raised when a C1 prompt cache cannot be validated or written."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise KVCacheError(f"{name} must be a positive int scalar")
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise KVCacheError(f"{name} must be a positive int")
    coerced = int(value)
    if coerced <= 0:
        raise KVCacheError(f"{name} must be a positive int")
    return coerced

def _layer_index(value: Any, name: str) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise KVCacheError(f"{name} must be an int scalar")
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise KVCacheError(f"{name} must be an int layer index")
    coerced = int(value)
    if coerced < 0:
        raise KVCacheError(f"{name} must be a non-negative layer index")
    return coerced



def _metadata(n_prefix: int) -> dict[str, str]:
    metadata = {f"0.{layer_index}": "" for layer_index in range(_NUM_LAYERS)}
    metadata.update({f"2.{layer_index}": "KVCache" for layer_index in range(_NUM_LAYERS)})
    metadata.update(
        {
            "1.offset": str(n_prefix),
            "1.num_layers": str(_NUM_LAYERS),
            "1.n_kv_heads": str(_N_KV_HEADS),
            "1.head_dim": str(_HEAD_DIM),
        }
    )
    return metadata


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KVCacheError(f"{name} must be a mapping")
    return value


def _require_layer_array(layer_index: int, name: str, value: Any, n_prefix: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise KVCacheError(f"layer {layer_index} {name} must be a numpy array")
    if value.dtype != np.float16:
        raise KVCacheError(
            f"layer {layer_index} {name} dtype must be float16/fp16, got {value.dtype}"
        )
    expected_shape = (_BATCH, _N_KV_HEADS, n_prefix, _HEAD_DIM)
    if value.ndim != 4:
        raise KVCacheError(
            f"layer {layer_index} {name} shape must be rank 4 {expected_shape}, got {value.shape}"
        )
    if value.shape != expected_shape:
        if value.shape[2] != n_prefix:
            raise KVCacheError(
                f"n_prefix {n_prefix} does not match layer {layer_index} {name} temporal length {value.shape[2]}"
            )
        raise KVCacheError(
            f"layer {layer_index} {name} shape must be {expected_shape} with 8 KV heads and head_dim 64, got {value.shape}"
        )
    return value


def _validated_payload(prefill_result: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, str], int]:
    result = _require_mapping(prefill_result, "prefill_result")
    if "n_prefix" not in result:
        raise KVCacheError("prefill_result must contain positive int n_prefix")
    if "layers" not in result:
        raise KVCacheError("prefill_result must contain layers")

    n_prefix = _positive_int(result["n_prefix"], "n_prefix")
    layers = result["layers"]
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes, bytearray)):
        raise KVCacheError("layers must be an ordered sequence")
    if len(layers) != _NUM_LAYERS:
        raise KVCacheError(f"layer count/num_layers must be {_NUM_LAYERS}, got {len(layers)}")

    tensors: dict[str, np.ndarray] = {}
    for expected_index, layer_value in enumerate(layers):
        layer = _require_mapping(layer_value, f"layers[{expected_index}]")
        if "layer" not in layer:
            raise KVCacheError(f"layer order invalid: layers[{expected_index}] has no layer index")
        layer_index = _layer_index(layer["layer"], f"layers[{expected_index}].layer")
        if layer_index != expected_index:
            raise KVCacheError(
                f"layer order invalid: expected layer {expected_index}, got {layer_index}"
            )
        if "K" not in layer or "V" not in layer:
            raise KVCacheError(f"layer {expected_index} must contain K and V arrays")
        key = _require_layer_array(expected_index, "K", layer["K"], n_prefix)
        value = _require_layer_array(expected_index, "V", layer["V"], n_prefix)
        if key.shape != value.shape:
            raise KVCacheError(
                f"layer {expected_index} K/V shapes must match, got {key.shape} and {value.shape}"
            )
        tensors[f"{expected_index}.0"] = key.copy(order="C")
        tensors[f"{expected_index}.1"] = value.copy(order="C")

    return tensors, _metadata(n_prefix), n_prefix


def _validate_out_path(out_path: os.PathLike[str] | str) -> Path:
    path = Path(out_path)
    if path.suffix != _EXPECTED_SUFFIX:
        raise KVCacheError(f"output path must end with {_EXPECTED_SUFFIX}")
    if not path.parent.exists() or not path.parent.is_dir():
        raise KVCacheError(
            f"output path parent directory is not writable/missing for write: {path.parent}"
        )
    return path


def emit_prompt_cache(prefill_result: Mapping[str, Any], out_path: os.PathLike[str] | str) -> None:
    """Write a C1 Llama prefill result as an mlx-lm prompt-cache safetensors file.

    The accepted schema is ``{"n_prefix": int, "layers": [...]}`` with exactly
    16 ordered layers.  Each layer must be a mapping containing ``layer`` equal
    to its zero-based order and fp16 numpy ``K``/``V`` arrays shaped
    ``(1, 8, n_prefix, 64)``.  Malformed dtype, shape, order, or output paths
    raise :class:`KVCacheError` before the final output path is installed.
    """

    tensors, metadata, _n_prefix = _validated_payload(prefill_result)
    path = _validate_out_path(out_path)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}{_EXPECTED_SUFFIX}")
    try:
        save_file(tensors, str(tmp_path), metadata=metadata)
        os.replace(tmp_path, path)
    except Exception as exc:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        if isinstance(exc, KVCacheError):
            raise
        raise KVCacheError(f"failed to write output path {path}: {exc}") from exc


def _npz_array(npz: Mapping[str, Any], key: str, path: Path) -> np.ndarray:
    if key not in npz:
        raise KVCacheError(f"required prefill npz key {key!r} missing from {path}")
    value = npz[key]
    if not isinstance(value, np.ndarray):
        raise KVCacheError(f"prefill npz key {key!r} did not load as a numpy array")
    return value


def prefill_result_from_npz(path: os.PathLike[str] | str, *, model: str | None = None) -> dict[str, Any]:
    """Load fixture/C1-7 ``layer{i}_K``/``layer{i}_V`` arrays into emitter input.

    Arrays are preserved as loaded; dtype and full shape validation is performed
    by :func:`emit_prompt_cache` so malformed fixtures are not silently cast.
    """

    npz_path = Path(path)
    try:
        with np.load(npz_path) as npz:
            layers: list[dict[str, Any]] = []
            for layer_index in range(_NUM_LAYERS):
                layers.append(
                    {
                        "layer": layer_index,
                        "K": _npz_array(npz, f"layer{layer_index}_K", npz_path),
                        "V": _npz_array(npz, f"layer{layer_index}_V", npz_path),
                    }
                )
            layer0_k = layers[0]["K"]
            if layer0_k.ndim < 3:
                raise KVCacheError(
                    f"layer0_K shape {layer0_k.shape} cannot infer n_prefix temporal dimension"
                )
            inferred_n_prefix = int(layer0_k.shape[2])
            if "n_prefix" in npz:
                n_prefix = _positive_int(npz["n_prefix"], "n_prefix")
                if n_prefix != inferred_n_prefix:
                    raise KVCacheError(
                        f"n_prefix {n_prefix} does not match layer0_K temporal length {inferred_n_prefix}"
                    )
            else:
                n_prefix = inferred_n_prefix
    except KVCacheError:
        raise
    except Exception as exc:
        raise KVCacheError(f"failed to load prefill npz {npz_path}: {exc}") from exc

    return {"model": model, "n_prefix": n_prefix, "layers": layers}


def _write_log(log_path: os.PathLike[str] | str, lines: Sequence[tuple[str, Any]]) -> None:
    text = "".join(f"{key}: {value}\n" for key, value in lines)
    Path(log_path).write_text(text, encoding="utf-8")


def _command_line(argv: Sequence[str]) -> str:
    return shlex.join([sys.executable, "-m", "native_r9700.kv_cache", *argv])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit an mlx-lm prompt-cache safetensors file from a C1 prefill NPZ")
    parser.add_argument("--prefill-npz", required=True, help="input NPZ with layer{i}_K/layer{i}_V arrays")
    parser.add_argument("--out", required=True, help="output .safetensors prompt-cache path")
    parser.add_argument("--log", required=True, help="path for a compact conversion log")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    command = _command_line(actual_argv)
    try:
        result = prefill_result_from_npz(args.prefill_npz)
        emit_prompt_cache(result, args.out)
        _write_log(
            args.log,
            (
                ("command", command),
                ("prefill_npz", args.prefill_npz),
                ("output", args.out),
                ("n_prefix", result["n_prefix"]),
                ("num_layers", len(result["layers"])),
                ("exit_status", 0),
            ),
        )
        print(
            f"wrote prompt cache {args.out} "
            f"(n_prefix={result['n_prefix']}, num_layers={len(result['layers'])})"
        )
        return 0
    except Exception as exc:
        message = str(exc)
        try:
            _write_log(
                args.log,
                (
                    ("command", command),
                    ("prefill_npz", args.prefill_npz),
                    ("output", args.out),
                    ("exit_status", 1),
                    ("stderr", message),
                ),
            )
        except Exception:
            pass
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised by focused CLI tests.
    raise SystemExit(main())
```

## File: `native_r9700/prefill.py`

```text
"""C1 Llama-3.2-1B full-layer prefix prefill producer.

Narrow first-parity path: MLX safetensors model directory + config sidecar in,
S-1 prefix token ids in, all 16 layer fp16 K/V tensors out.  The producer path
is stdlib + numpy + safetensors only; MLX/tinygrad remain outside production.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from safetensors import safe_open

from . import primitives
from .attention import (
    apply_rope_split_half,
    compare_layer_kv_to_fixture,
    format_layer_kv_delta_report,
    llama3_rope_frequencies,
    split_prompt_tokens_for_cache,
)
from .config import load_config_from_json


class PrefillError(ValueError):
    """Base class for narrow prefill producer misuse."""


@dataclass(frozen=True)
class _LayerWeights:
    input_norm: np.ndarray
    post_norm: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray


_LAYER_TENSOR_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


_EXPECTED_DELTA_LAYERS = (0, 15)


def _tensor_name(layer_index: int, suffix: str) -> str:
    return f"model.layers.{layer_index}.{suffix}"


def _required_tensor_names(num_layers: int) -> list[str]:
    names = ["model.embed_tokens.weight"]
    for layer_index in range(num_layers):
        for suffix in _LAYER_TENSOR_SUFFIXES:
            names.append(_tensor_name(layer_index, suffix))
    return names


def _weight_index_path(model_dir: str) -> Optional[str]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        return index_path
    return None


def _tensor_shards(model_dir: str, tensor_names: Sequence[str]) -> Dict[str, str]:
    index_path = _weight_index_path(model_dir)
    if index_path is None:
        single = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(single):
            raise PrefillError(
                f"no model.safetensors or model.safetensors.index.json found in {model_dir!r}"
            )
        return {name: single for name in tensor_names}

    try:
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise PrefillError(f"failed to parse safetensors index {index_path!r}: {exc}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise PrefillError(f"safetensors index {index_path!r} has no weight_map object")

    shards: Dict[str, str] = {}
    for name in tensor_names:
        shard_name = weight_map.get(name)
        if not shard_name:
            raise PrefillError(f"required tensor {name!r} missing from safetensors index {index_path!r}")
        shards[name] = os.path.join(model_dir, str(shard_name))
    return shards


def _load_tensor(tensor_name: str, shard_path: str) -> np.ndarray:
    if not os.path.exists(shard_path):
        raise PrefillError(f"required tensor shard missing for {tensor_name!r}: {shard_path!r}")
    try:
        with safe_open(shard_path, framework="np") as fh:
            if tensor_name not in fh.keys():
                raise PrefillError(f"required tensor {tensor_name!r} missing from {shard_path!r}")
            tensor = fh.get_tensor(tensor_name)
    except PrefillError:
        raise
    except Exception as exc:  # safetensors raises its own exception hierarchy.
        raise PrefillError(f"failed to load tensor {tensor_name!r} from {shard_path!r}: {exc}") from exc

    arr = np.asarray(tensor)
    if arr.dtype != np.float16:
        raise PrefillError(f"required tensor {tensor_name!r} must be fp16, got {arr.dtype}")
    return arr


def _load_and_validate_tensor(
    tensor_name: str, shard_path: str, expected_shape: tuple[int, ...]
) -> np.ndarray:
    arr = _load_tensor(tensor_name, shard_path)
    if arr.shape != expected_shape:
        raise PrefillError(
            f"required tensor {tensor_name!r} shape {arr.shape} != expected {expected_shape}"
        )
    return arr


def _coerce_prefix_token_ids(prefix_token_ids: Sequence[int]) -> list[int]:
    try:
        token_ids = [int(token_id) for token_id in prefix_token_ids]
    except (TypeError, ValueError) as exc:
        raise PrefillError("prefix_token_ids must be a sequence of integer token ids") from exc
    if len(token_ids) < 2:
        raise PrefillError(
            "prefix_token_ids must contain at least 2 prompt tokens for full-layer prefix prefill"
        )
    return token_ids


def _validate_token_ids_in_vocab(token_ids: Sequence[int], vocab_size: int) -> None:
    if min(token_ids) < 0 or max(token_ids) >= vocab_size:
        raise PrefillError(f"prefix_token_ids must be within [0, {vocab_size})")


def _load_embedding(shards: Mapping[str, str], cfg: Any) -> np.ndarray:
    return _load_and_validate_tensor(
        "model.embed_tokens.weight",
        shards["model.embed_tokens.weight"],
        (cfg.vocab_size, cfg.hidden_size),
    )


def _load_layer_weights(shards: Mapping[str, str], cfg: Any, layer_index: int) -> _LayerWeights:
    hidden = cfg.hidden_size
    kv_hidden = cfg.n_kv_heads * cfg.head_dim
    intermediate = cfg.intermediate_size

    def load(suffix: str, shape: tuple[int, ...]) -> np.ndarray:
        name = _tensor_name(layer_index, suffix)
        return _load_and_validate_tensor(name, shards[name], shape)

    return _LayerWeights(
        input_norm=load("input_layernorm.weight", (hidden,)),
        post_norm=load("post_attention_layernorm.weight", (hidden,)),
        q_proj=load("self_attn.q_proj.weight", (hidden, hidden)),
        k_proj=load("self_attn.k_proj.weight", (kv_hidden, hidden)),
        v_proj=load("self_attn.v_proj.weight", (kv_hidden, hidden)),
        o_proj=load("self_attn.o_proj.weight", (hidden, hidden)),
        gate_proj=load("mlp.gate_proj.weight", (intermediate, hidden)),
        up_proj=load("mlp.up_proj.weight", (intermediate, hidden)),
        down_proj=load("mlp.down_proj.weight", (hidden, intermediate)),
    )


def _project_heads(normed: np.ndarray, weight: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
    projected = primitives.matmul(normed, weight.T)
    expected = num_heads * head_dim
    if projected.shape != (normed.shape[0], expected):
        raise PrefillError(
            f"projection shape {projected.shape} != expected {(normed.shape[0], expected)}"
        )
    return projected.reshape(1, normed.shape[0], num_heads, head_dim).transpose(0, 2, 1, 3)


def _residual_add(x: np.ndarray, update: np.ndarray, name: str) -> np.ndarray:
    lhs = np.asarray(x)
    rhs = np.asarray(update)
    if lhs.dtype != np.float16 or rhs.dtype != np.float16:
        raise PrefillError(f"{name} residual add requires fp16 operands, got {lhs.dtype} and {rhs.dtype}")
    if lhs.shape != rhs.shape:
        raise PrefillError(f"{name} residual add shape mismatch: {lhs.shape} != {rhs.shape}")
    return (lhs + rhs).astype(np.float16, copy=False)


def _causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, cfg: Any) -> np.ndarray:
    if q.shape != (1, cfg.num_heads, q.shape[2], cfg.head_dim):
        raise PrefillError(f"Q shape {q.shape} does not match supported attention geometry")
    if k.shape != (1, cfg.n_kv_heads, q.shape[2], cfg.head_dim):
        raise PrefillError(f"K shape {k.shape} does not match supported attention geometry")
    if v.shape != k.shape:
        raise PrefillError(f"V shape {v.shape} != K shape {k.shape}")

    repeats = cfg.num_heads // cfg.n_kv_heads
    if repeats * cfg.n_kv_heads != cfg.num_heads:
        raise PrefillError(
            f"num_heads {cfg.num_heads} must be a multiple of n_kv_heads {cfg.n_kv_heads}"
        )
    k_heads = np.repeat(k, repeats, axis=1).astype(np.float32)
    v_heads = np.repeat(v, repeats, axis=1).astype(np.float32)
    q_heads = q.astype(np.float32)

    scores = np.matmul(q_heads, k_heads.transpose(0, 1, 3, 2))
    scores *= np.float32(1.0 / np.sqrt(np.float32(cfg.head_dim)))
    n_tokens = q.shape[2]
    mask = np.triu(np.ones((n_tokens, n_tokens), dtype=bool), k=1)
    scores = np.where(mask[np.newaxis, np.newaxis, :, :], -np.inf, scores)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores, dtype=np.float32)
    probs /= np.sum(probs, axis=-1, keepdims=True)

    context = np.matmul(probs, v_heads).astype(np.float16)
    return context.transpose(0, 2, 1, 3).reshape(n_tokens, cfg.hidden_size)


def _run_layer(x: np.ndarray, weights: _LayerWeights, cfg: Any, positions: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normed = primitives.rms_norm(x, weights.input_norm, cfg.rms_norm_eps)
    q = _project_heads(normed, weights.q_proj, cfg.num_heads, cfg.head_dim)
    k = _project_heads(normed, weights.k_proj, cfg.n_kv_heads, cfg.head_dim)
    v = _project_heads(normed, weights.v_proj, cfg.n_kv_heads, cfg.head_dim)

    q = apply_rope_split_half(q, positions, freqs)
    k = apply_rope_split_half(k, positions, freqs)

    attention_out = _causal_attention(q, k, v, cfg)
    projected = primitives.matmul(attention_out, weights.o_proj.T)
    x = _residual_add(x, projected, "attention")

    post_normed = primitives.rms_norm(x, weights.post_norm, cfg.rms_norm_eps)
    gate = primitives.matmul(post_normed, weights.gate_proj.T)
    up = primitives.matmul(post_normed, weights.up_proj.T)
    gated = (primitives.silu(gate) * up).astype(np.float16, copy=False)
    mlp_out = primitives.matmul(gated, weights.down_proj.T)
    x = _residual_add(x, mlp_out, "mlp")
    return x, k, v


def prefill_prompt_prefix(model_dir: str, prefix_token_ids: Sequence[int]) -> Mapping[str, object]:
    """Run all 16 Llama-3.2-1B decoder layers for an S-1 prefix prompt.

    Returns an ordered mapping with model/config provenance, prefix length, and
    one layer dict per decoder layer.  Each layer dict carries fp16 K/V cache
    arrays shaped ``(1, 8, N, 64)`` in temporal order.
    """

    token_ids = _coerce_prefix_token_ids(prefix_token_ids)
    cfg = load_config_from_json(model_dir)
    _validate_token_ids_in_vocab(token_ids, cfg.vocab_size)
    config_path = os.path.join(model_dir, "config.json")
    shards = _tensor_shards(model_dir, _required_tensor_names(cfg.num_layers))
    freqs = llama3_rope_frequencies(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)
    positions = np.arange(len(token_ids), dtype=np.int64)

    embed_weight = _load_embedding(shards, cfg)
    x = embed_weight[np.asarray(token_ids, dtype=np.int64)]
    if x.shape != (len(token_ids), cfg.hidden_size) or x.dtype != np.float16:
        raise PrefillError(
            f"embedding lookup produced {x.dtype} {x.shape}, expected fp16 {(len(token_ids), cfg.hidden_size)}"
        )

    layers: list[dict[str, object]] = []
    for layer_index in range(cfg.num_layers):
        weights = _load_layer_weights(shards, cfg, layer_index)
        x, k, v = _run_layer(x, weights, cfg, positions, freqs)
        expected_kv_shape = (1, cfg.n_kv_heads, len(token_ids), cfg.head_dim)
        if k.dtype != np.float16 or k.shape != expected_kv_shape:
            raise PrefillError(f"layer {layer_index} K produced {k.dtype} {k.shape}, expected fp16 {expected_kv_shape}")
        if v.dtype != np.float16 or v.shape != expected_kv_shape:
            raise PrefillError(f"layer {layer_index} V produced {v.dtype} {v.shape}, expected fp16 {expected_kv_shape}")
        layers.append({"layer": layer_index, "K": k, "V": v})

    return {
        "model": model_dir,
        "config_path": config_path,
        "n_prefix": len(token_ids),
        "layers": layers,
    }


def write_prefill_npz(result: Mapping[str, object], out_path: os.PathLike[str] | str) -> None:
    """Write all layer K/V arrays plus scalar metadata to a NumPy NPZ file."""

    layers = [dict(layer) for layer in result["layers"]]  # type: ignore[index,arg-type]
    arrays: dict[str, np.ndarray] = {
        "n_prefix": np.asarray(int(result["n_prefix"]), dtype=np.int64),
        "num_layers": np.asarray(len(layers), dtype=np.int64),
    }
    for layer_map in layers:
        layer_index = int(layer_map["layer"])
        arrays[f"layer{layer_index}_K"] = np.asarray(layer_map["K"], dtype=np.float16)
        arrays[f"layer{layer_index}_V"] = np.asarray(layer_map["V"], dtype=np.float16)

    out_str = os.fspath(out_path)
    parent = os.path.dirname(out_str)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(out_str, **arrays)


def _load_prompt_tokens(fixtures_dir: str, prompt_name: str) -> list[int]:
    prompts_path = os.path.join(fixtures_dir, "prompts.json")
    try:
        with open(prompts_path, encoding="utf-8") as fh:
            prompts = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise PrefillError(f"failed to load prompts fixture {prompts_path!r}: {exc}") from exc
    try:
        token_ids = prompts[prompt_name]["token_ids"]
    except (KeyError, TypeError) as exc:
        raise PrefillError(f"prompt {prompt_name!r} missing token_ids in {prompts_path!r}") from exc
    return [int(token_id) for token_id in token_ids]


def _write_log(path: Optional[str], lines: Iterable[str]) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _layer_result(result: Mapping[str, object], layer_index: int) -> Mapping[str, object]:
    for layer in result["layers"]:  # type: ignore[index]
        layer_map = dict(layer)  # type: ignore[arg-type]
        if int(layer_map["layer"]) == layer_index:
            return layer_map
    raise PrefillError(f"prefill result missing layer {layer_index}")


def _delta_reports(result: Mapping[str, object], fixture_path: str) -> list[str]:
    if not os.path.exists(fixture_path):
        return [f"fixture missing: {fixture_path}"]
    reports: list[str] = []
    for layer_index in _EXPECTED_DELTA_LAYERS:
        layer = _layer_result(result, layer_index)
        try:
            deltas = compare_layer_kv_to_fixture(layer, fixture_path, layer_index=layer_index)
        except ValueError as exc:
            reports.append(f"layer={layer_index} fixture incompatible: {exc}")
            continue
        reports.append(format_layer_kv_delta_report(deltas))
    return reports


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m native_r9700.prefill",
        description="Produce and compare Llama-3.2-1B full-layer S-1 prefix K/V tensors.",
    )
    parser.add_argument("--model", required=True, help="MLX safetensors model directory")
    parser.add_argument("--fixtures-dir", required=True, help="Directory containing prompts.json and kv_state.npz")
    parser.add_argument("--prompt-name", required=True, help="Prompt fixture name, e.g. prompt-0")
    parser.add_argument("--out", required=True, help="Path to write the full-layer prefix NPZ")
    parser.add_argument("--log", help="Path to write the prefill delta log")
    args = parser.parse_args(argv)

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    command = shlex.join([sys.executable, "-m", "native_r9700.prefill", *raw_argv])
    try:
        token_ids = _load_prompt_tokens(args.fixtures_dir, args.prompt_name)
        prefix_token_ids, final_token_id = split_prompt_tokens_for_cache(token_ids)
        result = prefill_prompt_prefix(args.model, prefix_token_ids)
        write_prefill_npz(result, args.out)
        reports = _delta_reports(result, os.path.join(args.fixtures_dir, "kv_state.npz"))
        num_layers = len(list(result["layers"]))  # type: ignore[arg-type,index]
        summary = f"prefill n_prefix={result['n_prefix']} num_layers={num_layers} output={args.out}"
        _write_log(
            args.log,
            (
                f"command: {command}",
                f"model: {args.model}",
                f"config: {result['config_path']}",
                f"prompt: {args.prompt_name}",
                f"final_token_id: {final_token_id}",
                f"n_prefix: {result['n_prefix']}",
                f"num_layers: {num_layers}",
                f"output: {args.out}",
                "deltas:",
                *reports,
                "exit_status: 0",
            ),
        )
        print(summary)
        for report in reports:
            print(report)
        return 0
    except Exception as exc:
        _write_log(
            args.log,
            (
                f"command: {command}",
                f"model: {args.model}",
                f"prompt: {args.prompt_name}",
                f"output: {args.out}",
                f"error: {exc}",
                "exit_status: 1",
            ),
        )
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

## File: `tests/native_r9700/test_kv_cache.py`

```text
"""C1 task set 8 RED contract for mlx-lm prompt-cache safetensors emission.

These tests define the future ``native_r9700.kv_cache`` API before production
code lands. The module is imported lazily so pytest collection succeeds; the
current RED should be a clear missing module/API failure, not a syntax error.

Contract: take the C1 task set 7 prefill result shape (16 ordered layers, fp16
K/V arrays shaped ``(1, 8, N, 64)``), emit the mlx-lm prompt-cache safetensors
ABI, and keep Qwen/decode/parity-harness/native-runtime integration outside
this C1 RED gate.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "native_r9700" / "fixtures"
_KV_FIXTURE_NPZ = _FIXTURE_DIR / "kv_state.npz"
_PYTHON = "${HOME}/.pyenv/versions/3.12.8/bin/python3"

_EXPECTED_NUM_LAYERS = 16
_EXPECTED_N_PREFIX = 5
_EXPECTED_N_KV_HEADS = 8
_EXPECTED_HEAD_DIM = 64
_EXPECTED_KV_SHAPE = (
    1,
    _EXPECTED_N_KV_HEADS,
    _EXPECTED_N_PREFIX,
    _EXPECTED_HEAD_DIM,
)
_EXPECTED_METADATA = {
    **{f"0.{layer_index}": "" for layer_index in range(_EXPECTED_NUM_LAYERS)},
    **{
        f"2.{layer_index}": "KVCache"
        for layer_index in range(_EXPECTED_NUM_LAYERS)
    },
    "1.offset": str(_EXPECTED_N_PREFIX),
    "1.num_layers": str(_EXPECTED_NUM_LAYERS),
    "1.n_kv_heads": str(_EXPECTED_N_KV_HEADS),
    "1.head_dim": str(_EXPECTED_HEAD_DIM),
}


# Production mutation caught: deleting/renaming the task set 8 public module or
# entry points should fail here before any safetensors behavior is exercised.
def _kv_cache_module():
    try:
        module = importlib.import_module("native_r9700.kv_cache")
    except ModuleNotFoundError as exc:
        if exc.name == "native_r9700.kv_cache":
            pytest.fail(
                "native_r9700.kv_cache module missing; implement the C1 task "
                "set 8 prompt-cache safetensors emitter API"
            )
        raise

    for api_name in ("emit_prompt_cache", "prefill_result_from_npz"):
        assert hasattr(module, api_name), (
            f"native_r9700.kv_cache missing public API: {api_name}"
        )
        assert callable(getattr(module, api_name)), (
            f"native_r9700.kv_cache.{api_name} must be callable"
        )
    return module


def _synthetic_prefill_result():
    base = (
        np.arange(np.prod(_EXPECTED_KV_SHAPE), dtype=np.float32).reshape(
            _EXPECTED_KV_SHAPE
        )
        / np.float32(2048.0)
    )
    layers = []
    for layer_index in range(_EXPECTED_NUM_LAYERS):
        layers.append(
            {
                "layer": layer_index,
                "K": (base + np.float32(layer_index)).astype(np.float16),
                "V": (base + np.float32(50 + layer_index)).astype(np.float16),
            }
        )
    return {
        "model": "synthetic-llama",
        "n_prefix": _EXPECTED_N_PREFIX,
        "layers": layers,
    }


def _write_synthetic_npz(path: Path):
    result = _synthetic_prefill_result()
    arrays = {}
    for layer in result["layers"]:
        layer_index = layer["layer"]
        arrays[f"layer{layer_index}_K"] = layer["K"]
        arrays[f"layer{layer_index}_V"] = layer["V"]
    np.savez(path, **arrays)
    return result


def _safe_open_header(path: Path):
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - dependency is expected here.
        pytest.fail(f"safetensors is required for prompt-cache header checks: {exc}")

    with safe_open(str(path), framework="np") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key) for key in keys}
    return keys, metadata, tensors


def _assert_prompt_cache_header(path: Path, result):
    keys, metadata, tensors = _safe_open_header(path)

    assert keys == {
        tensor_key
        for layer_index in range(_EXPECTED_NUM_LAYERS)
        for tensor_key in (f"{layer_index}.0", f"{layer_index}.1")
    }
    for key, expected_value in _EXPECTED_METADATA.items():
        assert metadata[key] == expected_value

    for layer in result["layers"]:
        layer_index = layer["layer"]
        np.testing.assert_array_equal(tensors[f"{layer_index}.0"], layer["K"])
        np.testing.assert_array_equal(tensors[f"{layer_index}.1"], layer["V"])


def _load_prompt_cache_or_skip(path: Path):
    try:
        from mlx_lm.models.cache import load_prompt_cache
    except ImportError as exc:
        pytest.skip(f"mlx_lm prompt-cache round-trip unavailable: {exc}")

    return load_prompt_cache(str(path), return_metadata=True)


def _assert_mlx_round_trip(path: Path, result):
    cache, metadata = _load_prompt_cache_or_skip(path)

    assert metadata["offset"] == str(_EXPECTED_N_PREFIX)
    assert metadata["num_layers"] == str(_EXPECTED_NUM_LAYERS)
    assert metadata["n_kv_heads"] == str(_EXPECTED_N_KV_HEADS)
    assert metadata["head_dim"] == str(_EXPECTED_HEAD_DIM)
    assert len(cache) == _EXPECTED_NUM_LAYERS

    for expected_layer_index, (cache_layer, input_layer) in enumerate(
        zip(cache, result["layers"], strict=True)
    ):
        assert type(cache_layer).__name__ == "KVCache"
        assert cache_layer.offset == _EXPECTED_N_PREFIX
        assert cache_layer.size() == _EXPECTED_N_PREFIX
        assert input_layer["layer"] == expected_layer_index
        np.testing.assert_array_equal(np.asarray(cache_layer.keys), input_layer["K"])
        np.testing.assert_array_equal(np.asarray(cache_layer.values), input_layer["V"])


def _assert_prefill_result_shape(result):
    assert result["n_prefix"] == _EXPECTED_N_PREFIX
    assert len(result["layers"]) == _EXPECTED_NUM_LAYERS
    for layer_index in (0, 15):
        layer = result["layers"][layer_index]
        assert layer["layer"] == layer_index
        assert layer["K"].dtype == np.float16
        assert layer["V"].dtype == np.float16
        assert layer["K"].shape == _EXPECTED_KV_SHAPE
        assert layer["V"].shape == _EXPECTED_KV_SHAPE


def test_kv_cache_module_exports_public_api():
    kv_cache = _kv_cache_module()

    assert callable(kv_cache.emit_prompt_cache)
    assert callable(kv_cache.prefill_result_from_npz)


def test_emit_prompt_cache_writes_mlx_lm_safetensors_header(tmp_path):
    kv_cache = _kv_cache_module()
    result = _synthetic_prefill_result()
    out_path = tmp_path / "synthetic-prompt-cache.safetensors"

    kv_cache.emit_prompt_cache(result, out_path)

    assert out_path.is_file()
    _assert_prompt_cache_header(out_path, result)


def test_emit_prompt_cache_round_trips_through_mlx_lm_when_available(tmp_path):
    kv_cache = _kv_cache_module()
    result = _synthetic_prefill_result()
    out_path = tmp_path / "synthetic-prompt-cache.safetensors"

    kv_cache.emit_prompt_cache(result, out_path)

    _assert_mlx_round_trip(out_path, result)


def test_prefill_result_from_npz_fixture_converts_and_emits_header(tmp_path):
    if not _KV_FIXTURE_NPZ.is_file():
        pytest.skip(f"missing committed KV fixture {_KV_FIXTURE_NPZ}")
    kv_cache = _kv_cache_module()
    out_path = tmp_path / "fixture-prompt-cache.safetensors"

    result = kv_cache.prefill_result_from_npz(_KV_FIXTURE_NPZ, model="fixture-model")
    _assert_prefill_result_shape(result)
    kv_cache.emit_prompt_cache(result, out_path)

    assert out_path.is_file()
    _assert_prompt_cache_header(out_path, result)


def test_prefill_result_from_npz_fixture_round_trips_when_mlx_lm_available(tmp_path):
    if not _KV_FIXTURE_NPZ.is_file():
        pytest.skip(f"missing committed KV fixture {_KV_FIXTURE_NPZ}")
    kv_cache = _kv_cache_module()
    out_path = tmp_path / "fixture-prompt-cache.safetensors"

    result = kv_cache.prefill_result_from_npz(_KV_FIXTURE_NPZ, model="fixture-model")
    kv_cache.emit_prompt_cache(result, out_path)

    _assert_mlx_round_trip(out_path, result)


@pytest.mark.parametrize(
    ("mutation", "out_path_factory", "match"),
    [
        pytest.param(
            lambda result: result["layers"][0].__setitem__(
                "K", result["layers"][0]["K"].astype(np.float32)
            ),
            lambda tmp_path: tmp_path / "wrong-dtype.safetensors",
            "(?i)dtype|float16|fp16",
            id="wrong-dtype-fp32",
        ),
        pytest.param(
            lambda result: result["layers"][0].__setitem__(
                "K", result["layers"][0]["K"][:, :7, :, :]
            ),
            lambda tmp_path: tmp_path / "wrong-head-count.safetensors",
            "(?i)shape|head|8",
            id="wrong-shape-head-count",
        ),
        pytest.param(
            lambda result: result["layers"].pop(),
            lambda tmp_path: tmp_path / "wrong-layer-count.safetensors",
            "(?i)layer|count|num_layers|16",
            id="wrong-layer-count",
        ),
        pytest.param(
            lambda result: result["layers"].__setitem__(
                slice(1, 3), [result["layers"][2], result["layers"][1]]
            ),
            lambda tmp_path: tmp_path / "wrong-layer-order.safetensors",
            "(?i)layer|order",
            id="wrong-layer-order",
        ),
        pytest.param(
            lambda result: result.__setitem__("n_prefix", _EXPECTED_N_PREFIX - 1),
            lambda tmp_path: tmp_path / "wrong-prefix.safetensors",
            "(?i)n_prefix|offset|length|5",
            id="n-prefix-mismatch",
        ),
        pytest.param(
            lambda result: None,
            lambda tmp_path: tmp_path / "missing-parent" / "invalid.safetensors",
            "(?i)output|path|parent|write",
            id="invalid-output-path",
        ),
    ],
)
def test_emit_prompt_cache_rejects_invalid_input_without_final_file(
    tmp_path, mutation, out_path_factory, match
):
    kv_cache = _kv_cache_module()
    result = _synthetic_prefill_result()
    out_path = out_path_factory(tmp_path)
    mutation(result)

    with pytest.raises(ValueError, match=match):
        kv_cache.emit_prompt_cache(result, out_path)

    assert not out_path.exists()


def test_kv_cache_cli_converts_prefill_npz_and_writes_log(tmp_path):
    prefill_npz = tmp_path / "synthetic-prefill.npz"
    out_path = tmp_path / "cli-prompt-cache.safetensors"
    log_path = tmp_path / "kv-cache.log"
    expected_result = _write_synthetic_npz(prefill_npz)

    completed = subprocess.run(
        [
            _PYTHON,
            "-m",
            "native_r9700.kv_cache",
            "--prefill-npz",
            str(prefill_npz),
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert out_path.is_file(), completed.stdout + completed.stderr
    assert log_path.is_file(), completed.stdout + completed.stderr
    _assert_prompt_cache_header(out_path, expected_result)

    log_text = log_path.read_text(encoding="utf-8")
    assert "prefill_npz" in log_text
    assert str(prefill_npz) in log_text
    assert "output" in log_text
    assert str(out_path) in log_text
    assert "n_prefix: 5" in log_text
    assert "num_layers: 16" in log_text
    assert "exit_status: 0" in log_text
```

## File: `tests/native_r9700/test_prefill.py`

```text
"""C1 task set 7 RED contract for full-layer Llama prefix prefill.

These tests define the future ``native_r9700.prefill`` API before production
code lands. The module is imported lazily so pytest collection succeeds; the
current RED should be a clear missing module/API failure, not a syntax error.

Contract: Llama-3.2-1B-Instruct MLX model dir, prompt-0 S-1 prefix tokens from
``prompts.json``, all 16 layers of fp16 K/V shaped ``(1, 8, N, 64)`` in layer
and temporal order, and no Qwen/partial-layer broadening in this C1 ladder.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "native_r9700" / "fixtures"
_PROMPTS_JSON = _FIXTURE_DIR / "prompts.json"
_KV_FIXTURE_NPZ = _FIXTURE_DIR / "kv_state.npz"
_PYTHON = "${HOME}/.pyenv/versions/3.12.8/bin/python3"
_LLAMA_MLX_MODEL_DIR = (
    _REPO_ROOT
    / ".."
    / "tinygrad-kv-worker-phase0"
    / "mlx_models"
    / "meta-Llama-3.2-1B-Instruct"
).resolve()

_EXPECTED_NUM_LAYERS = 16
_EXPECTED_N_PREFIX = 5
_EXPECTED_KV_SHAPE = (1, 8, _EXPECTED_N_PREFIX, 64)


# Production mutation caught: deleting/renaming the task set 7 public module or
# entry point should fail here before any fixture-backed behavior is exercised.
def _prefill_module():
    try:
        module = importlib.import_module("native_r9700.prefill")
    except ModuleNotFoundError as exc:
        if exc.name == "native_r9700.prefill":
            pytest.fail(
                "native_r9700.prefill module missing; implement the C1 task "
                "set 7 full-layer prefix prefill API"
            )
        raise

    assert hasattr(module, "prefill_prompt_prefix"), (
        "native_r9700.prefill missing public API: prefill_prompt_prefix"
    )
    assert callable(module.prefill_prompt_prefix), (
        "native_r9700.prefill.prefill_prompt_prefix must be callable"
    )
    return module


def _prompt0_prefix_token_ids():
    with _PROMPTS_JSON.open(encoding="utf-8") as fh:
        prompt0 = json.load(fh)["prompt-0"]

    assert prompt0["S"] == 6
    token_ids = prompt0["token_ids"]
    assert token_ids == [128000, 791, 6864, 315, 9822, 374]
    return token_ids[: prompt0["S"] - 1]


def _require_model_dir():
    if not _LLAMA_MLX_MODEL_DIR.is_dir():
        pytest.skip(f"missing local Llama MLX model {_LLAMA_MLX_MODEL_DIR}")


def _require_prefill_inputs():
    missing = []
    if not _LLAMA_MLX_MODEL_DIR.is_dir():
        missing.append(f"local Llama MLX model {_LLAMA_MLX_MODEL_DIR}")
    if not _KV_FIXTURE_NPZ.is_file():
        missing.append(f"committed KV fixture {_KV_FIXTURE_NPZ}")
    if missing:
        pytest.skip("missing " + " and ".join(missing))


def _assert_full_layer_prefill_result(result):
    assert set(result) >= {"model", "n_prefix", "layers"}
    assert result["model"] is not None
    assert result["n_prefix"] == _EXPECTED_N_PREFIX

    layers = list(result["layers"])
    assert len(layers) == _EXPECTED_NUM_LAYERS
    for expected_layer_index, layer in enumerate(layers):
        assert layer["layer"] == expected_layer_index
        for name in ("K", "V"):
            arr = np.asarray(layer[name])
            assert arr.dtype == np.float16, f"layer {expected_layer_index} {name} dtype"
            assert arr.shape == _EXPECTED_KV_SHAPE, (
                f"layer {expected_layer_index} {name} shape"
            )
    return layers


def _assert_layer_deltas_within_probe_bounds(layers):
    fixture = np.load(_KV_FIXTURE_NPZ)
    for layer_index in (0, 15):
        layer = layers[layer_index]
        for name, max_bound in (("K", 0.025), ("V", 0.012)):
            actual = np.asarray(layer[name]).astype(np.float32)
            expected = fixture[f"layer{layer_index}_{name}"].astype(np.float32)
            delta = np.abs(actual - expected)
            assert float(delta.max()) <= max_bound, f"layer {layer_index} {name} max"
            assert float(delta.mean()) <= 0.003, f"layer {layer_index} {name} mean"


def test_prefill_module_exports_prompt_prefix_api():
    prefill = _prefill_module()

    assert callable(prefill.prefill_prompt_prefix)


def test_prefill_prompt_prefix_emits_all_prompt0_layers_in_order_with_bounded_deltas():
    _require_prefill_inputs()
    prefill = _prefill_module()
    prefix_token_ids = _prompt0_prefix_token_ids()

    result = prefill.prefill_prompt_prefix(str(_LLAMA_MLX_MODEL_DIR), prefix_token_ids)

    layers = _assert_full_layer_prefill_result(result)
    _assert_layer_deltas_within_probe_bounds(layers)


def test_prefill_cli_writes_full_layer_npz_and_review_log(tmp_path):
    _require_prefill_inputs()
    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "prefill.log"

    completed = subprocess.run(
        [
            _PYTHON,
            "-m",
            "native_r9700.prefill",
            "--model",
            str(_LLAMA_MLX_MODEL_DIR),
            "--fixtures-dir",
            "tests/native_r9700/fixtures",
            "--prompt-name",
            "prompt-0",
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert out_path.is_file(), completed.stdout + completed.stderr
    assert log_path.is_file(), completed.stdout + completed.stderr

    candidate = np.load(out_path)
    for layer_index in range(_EXPECTED_NUM_LAYERS):
        for name in ("K", "V"):
            key = f"layer{layer_index}_{name}"
            assert key in candidate.files
            assert candidate[key].dtype == np.float16
            assert candidate[key].shape == _EXPECTED_KV_SHAPE
    for key in ("layer0_K", "layer0_V", "layer15_K", "layer15_V"):
        assert key in candidate.files

    log_text = log_path.read_text(encoding="utf-8")
    assert "command:" in log_text
    assert "model:" in log_text
    assert str(_LLAMA_MLX_MODEL_DIR) in log_text
    assert "prompt: prompt-0" in log_text
    assert "n_prefix: 5" in log_text
    assert "num_layers: 16" in log_text
    assert str(out_path) in log_text
    assert "exit_status: 0" in log_text


def test_prefill_cli_logs_fixture_shape_mismatch_without_failing(tmp_path, monkeypatch):
    prefill = _prefill_module()
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "prompts.json").write_text(
        json.dumps({"prompt-x": {"S": 7, "token_ids": [1, 2, 3, 4, 5, 6, 7]}}),
        encoding="utf-8",
    )
    np.savez(
        fixtures_dir / "kv_state.npz",
        layer0_K=np.zeros((1, 8, 5, 64), dtype=np.float16),
        layer0_V=np.zeros((1, 8, 5, 64), dtype=np.float16),
        layer15_K=np.zeros((1, 8, 5, 64), dtype=np.float16),
        layer15_V=np.zeros((1, 8, 5, 64), dtype=np.float16),
    )

    def fake_prefill_prompt_prefix(model_dir, prefix_token_ids):
        assert model_dir == "synthetic-model"
        assert prefix_token_ids == [1, 2, 3, 4, 5, 6]
        layers = [
            {
                "layer": layer_index,
                "K": np.zeros((1, 8, 6, 64), dtype=np.float16),
                "V": np.zeros((1, 8, 6, 64), dtype=np.float16),
            }
            for layer_index in range(16)
        ]
        return {
            "model": "synthetic-model",
            "config_path": "synthetic-config.json",
            "n_prefix": 6,
            "layers": layers,
        }

    monkeypatch.setattr(prefill, "prefill_prompt_prefix", fake_prefill_prompt_prefix)
    out_path = tmp_path / "prefill.npz"
    log_path = tmp_path / "prefill.log"

    rc = prefill.main(
        [
            "--model",
            "synthetic-model",
            "--fixtures-dir",
            str(fixtures_dir),
            "--prompt-name",
            "prompt-x",
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 0
    assert out_path.is_file()
    log_text = log_path.read_text(encoding="utf-8")
    assert "fixture incompatible" in log_text
    assert "K shape (1, 8, 6, 64) != fixture shape (1, 8, 5, 64)" in log_text
    assert "exit_status: 0" in log_text

@pytest.mark.parametrize("prefix_token_ids", [[], [128000]])
def test_prefill_prompt_prefix_rejects_prefixes_shorter_than_two_tokens(prefix_token_ids):
    prefill = _prefill_module()
    _require_model_dir()

    with pytest.raises(ValueError, match="(?i)prompt|prefix|at least 2|short"):
        prefill.prefill_prompt_prefix(str(_LLAMA_MLX_MODEL_DIR), prefix_token_ids)
```

## File: `.superpowers/swarm/reports/c1-task-8-kv-emitter-red.md`

```text
# C1 task set 8 — KV prompt-cache emitter RED contract

## Files changed

- `tests/native_r9700/test_kv_cache.py` — new focused RED tests for the future `native_r9700.kv_cache` API and CLI.
- `docs/tasks/native-r9700-producer/validation-commands.md` — added the exact focused task set 8 RED/GREEN command and updated the discovery row.
- `.superpowers/swarm/reports/c1-task-8-kv-emitter-red.md` — this handoff report.

## Command added

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_kv_cache.py -v
```

## Expected RED reason

Expected RED before production implementation: pytest collection succeeds, then the focused tests fail with a clear missing `native_r9700.kv_cache` module/API failure.

## Contract covered

- Lazy import helper freezes `native_r9700.kv_cache.emit_prompt_cache` and `prefill_result_from_npz` without causing collection-time import errors.
- Synthetic prefill helper builds 16 ordered layers with `n_prefix=5`; every K/V array is fp16 and shaped `(1, 8, 5, 64)` with deterministic distinct data.
- `emit_prompt_cache` must write a `.safetensors` prompt cache with tensor keys `{i}.0`/`{i}.1` for layers 0 through 15 and mlx-lm metadata keys `0.{i}`, `2.{i}`, `1.offset`, `1.num_layers`, `1.n_kv_heads`, and `1.head_dim`.
- mlx-lm round-trip checks use `mlx_lm.models.cache.load_prompt_cache(..., return_metadata=True)` when available; only that round-trip part skips if mlx-lm is unavailable, while safetensors header checks remain active.
- `prefill_result_from_npz` must consume committed fixture-style NPZ files with `layer{i}_K`/`layer{i}_V` arrays, including `tests/native_r9700/fixtures/kv_state.npz` when present.
- Failure tests require `ValueError` and no final output file for fp32 K/V, wrong head-count shape, wrong layer count, wrong layer order, `n_prefix` mismatch, and invalid output path.
- CLI must run as `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.kv_cache --prefill-npz <tmp.npz> --out <tmp.safetensors> --log <tmp.log>`, exit 0, write a valid safetensors header, and log `prefill_npz`, `output`, `n_prefix: 5`, `num_layers: 16`, and `exit_status: 0`.
- Production `native_r9700.kv_cache`, parity harness/decode, C2 integration, Qwen support, and C++ runtime remain non-goals for this RED gate.

Validation was not run, per the task constraint that the supervisor owns RED/GREEN validation.
```

## File: `.superpowers/swarm/reports/c1-task-8-kv-emitter.md`

```text
# C1 task set 8 — KV prompt-cache emitter implementation

## Files changed

- `native_r9700/kv_cache.py` — added the C1 Llama prompt-cache safetensors emitter, NPZ adapter, validation error type, atomic write path, and CLI.
- `.superpowers/swarm/reports/c1-task-8-kv-emitter.md` — this implementation report.

## API summary

- `KVCacheError(ValueError)` is the public validation/write failure type.
- `emit_prompt_cache(prefill_result, out_path)` validates a narrow C1 Llama prefill result, assembles mlx-lm prompt-cache tensors/metadata, writes a sibling temp file, then installs it with `os.replace`.
- `prefill_result_from_npz(path, *, model=None)` loads fixture/C1-7 NPZ keys `layer{i}_K` and `layer{i}_V` for layers `0..15`, preserves the loaded numpy arrays as-is, infers `n_prefix` from `layer0_K.shape[2]` unless a scalar `n_prefix` key is present and consistent, and returns the emitter input mapping.
- CLI: `python -m native_r9700.kv_cache --prefill-npz <path.npz> --out <path.safetensors> --log <path.log>` converts NPZ input, writes the prompt cache, logs command/input/output/counts/status, and prints a compact success line.

## Exact accepted prefill schema

```python
{
    "model": object | None,          # passed through by prefill_result_from_npz; not used by emitter
    "n_prefix": positive int,
    "layers": [
        {
            "layer": 0,             # must equal list order
            "K": np.ndarray,        # dtype np.float16, shape (1, 8, n_prefix, 64)
            "V": np.ndarray,        # dtype np.float16, shape (1, 8, n_prefix, 64)
        },
        ...                          # exactly 16 ordered layers, through layer 15
    ],
}
```

## Safetensors schema emitted

- Tensor keys: `"{i}.0"` for K and `"{i}.1"` for V, for every layer `i in range(16)`.
- Tensor values: contiguous copies of the validated fp16 K/V arrays.
- Metadata keys and string values:
  - `0.{i}: ""` for `i=0..15`
  - `2.{i}: "KVCache"` for `i=0..15`
  - `1.offset: str(n_prefix)`
  - `1.num_layers: "16"`
  - `1.n_kv_heads: "8"`
  - `1.head_dim: "64"`

## Failure behavior

- Malformed top-level input, missing/invalid `n_prefix`, wrong layer count/order, non-numpy K/V values, non-fp16 dtype, rank/shape mismatch, K/V shape mismatch, and temporal length mismatch raise `KVCacheError`/`ValueError` without installing the final output file.
- `out_path` must end in `.safetensors`.
- Missing or non-directory output parent raises `KVCacheError`/`ValueError` mentioning output/path/parent/write.
- Writes use a temp sibling named `.<output-name>.tmp.<pid>.safetensors`; temp files are cleaned up on save/replace errors before re-raising `KVCacheError`.
- CLI returns `0` and logs `exit_status: 0` on success; on errors it returns `1`, writes `exit_status: 1` plus `stderr` when possible, and prints the error to stderr.

## Qwen decision

Qwen support remains unsupported/deferred for this C1-8 gate. The implementation is intentionally fixed to the C1 Llama geometry: 16 layers, 8 KV heads, head dim 64, and `(1, 8, N, 64)` fp16 K/V tensors.

## Verification run here

- `python3 -m py_compile native_r9700/kv_cache.py`
- Tiny non-pytest smoke: imported `native_r9700.kv_cache`, emitted a synthetic safetensors prompt cache, checked keys/metadata/data with `safetensors.safe_open`, checked missing-parent rejection, loaded a synthetic NPZ with `prefill_result_from_npz`, and exercised the CLI path successfully.

No pytest, formatter, linter, package-manager, hardware, or git commands were run by this executor.

## Supervisor validation commands to run

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_kv_cache.py -v
```
```

## File: `docs/tasks/native-r9700-producer/validation-commands.md`

```text
# Native R9700 Producer — Validation Commands

This file is the shared command ledger for `docs/tasks/native-r9700-producer/`. Agents must add exact commands here when a phase task discovers them. Do not write placeholder commands; if a command is not knowable before implementation, name the task set that must discover it.

## Fixed environment

Use this Python for Python-side validation in this repo:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3
```

Do not rely on `python3` from `PATH`.

For AMD eGPU/tinygrad comparison runs that intentionally use tinygrad:

```sh
DEV=AMD
JITBEAM=2
HF_HOME=${HOME}/Development/ml/models
```

Path C native producer commands must not import or call tinygrad unless explicitly running a comparison/control command outside the producer path.

## Exact commands known now

### Existing Python regression suite

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v
```

Expected last known result from Phase 0 handoff:

```text
17 passed, 2 warnings
```

### Existing harness syntax check

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m py_compile tinygrad_kv_worker/harness.py
```

Expected last known result from Phase 0 handoff: exit 0.

### Existing Phase 0 GPU parity command

This is a regression/control command for the validated tinygrad producer path, not a Path C native command:

```sh
DEV=AMD JITBEAM=2 HF_HOME=${HOME}/Development/ml/models \
  ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m tinygrad_kv_worker.harness \
  --gguf mlx_models/meta-Llama-3.2-1B-Instruct.F16.gguf \
  --mlx mlx_models/meta-Llama-3.2-1B-Instruct \
  --out docs/path-a-validation-results.md \
  --run-tag meta-f16-final
```

Expected last known result from Phase 0 handoff:

```text
Gate PASS; report written to docs/path-a-validation-results.md
```

### C0A macOS TinyGPU.app / IOKit PCI discovery probe

This is the correct macOS visibility check for the existing tinygrad R9700 path. It is a reference/discovery command, not a Path C native producer command: it imports tinygrad to prove the substrate used by the working Phase 0 path.

```sh
JITBEAM=2 DEV=AMD PYTHONPATH=${HOME}/Development/ml/tools/tinygrad \
  ${HOME}/.pyenv/versions/3.12.8/bin/python3 -c "from tinygrad.runtime.support.system import System; from tinygrad import Device; devs=System.list_devices(0x1002, ((0xffff,(0x74a1,0x744c,0x7480,0x7550,0x7551,0x7590,0x75a0)),), None); print('amd_pci_devices', devs); d=Device['AMD']; print('iface', type(d.iface).__name__); print('arch', d.arch); print('pcibus', getattr(d.iface.pci_dev, 'pcibus', None)); print('pci_dev_class', type(d.iface.pci_dev).__name__)"
```

Observed supervisor result:

```text
amd_pci_devices [(<class 'tinygrad.runtime.support.system.APLRemotePCIDevice'>, '1002:7551')]
iface PCIIface
arch gfx1201
pcibus usb4
pci_dev_class APLRemotePCIDevice
```

User-provided working model/server command for the same substrate:

```sh
JITBEAM=2 DEV=AMD python3 -m tinygrad.llm
```

Task set 2 pinned the client-side native contract from the TinyGPU.app/APLRemotePCIDevice/PCIIface path, not from the stale libusb-only `USBIface` probe below.
Task set 3 is now satisfied by the C0B native AMDev/SDMA transfer proof below. The visible TinyGPU.app client ABI exposes primitive `RemoteCmd` PCI/sysmem/BAR/MMIO operations, so the passing proof implements the necessary tinygrad-free native AMD bring-up locally: BAR0/BAR2/BAR5 mapping, IP discovery, fixed gfx12 page tables, MMHUB VMID0/TLB setup, source-grounded SDMA0 7.0.1 queue0 reset/programming, BAR2 doorbell submission, fence polling, and CPU byte comparison. The latest supervisor log is `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z` with all host-device transfer pass tokens.

### C0A task set 3 host-device transfer proof evidence

The TinyGPU.app/APLRemotePCIDevice/PCIIface host↔device transfer command is the C0B native AMDev/SDMA transfer proof command in this file. It is accepted for C0A task set 3 only when the log contains:

- `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`
- `pci_id: 1002:7551`
- `arch: gfx1201`
- `transfer_byte_count: 32`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `failure_stage: none`
- `exit_status: 0`
- `wrapper_exit_status: 0`

The stale libusb-only command below remains a negative control and must not be used for transfer acceptance.

### C0B native AMDev/SDMA transfer contract tests

This is the no-hardware RED/GREEN contract for `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`. Task set 1 expects RED before production source exists; task set 2 and later must make it green without importing tinygrad or using libusb as the acceptance path.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected RED result before C0B task set 2:

```text
AssertionError: native transfer probe source missing
```

### C0B TinyGPU.app discovery smoke

This is the hardware discovery smoke for task set 3. It builds the native probe, runs `--discovery-smoke`, and writes `logs/c0b-discovery-smoke.log`.

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-discovery-smoke.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --discovery-smoke"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --discovery-smoke; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

OMP task executors record this command for the supervisor; they do not run it in task mode.

### C0B native AMDev/SDMA transfer proof

This is the task set 5 hardware transfer proof command. It builds the tinygrad-free native probe, runs `--transfer-proof`, and writes `logs/c0b-native-amdev-sdma-transfer.log`. Success requires `host_device_transfer_status: pass`, `transfer_byte_count: 32`, `cpu_comparison_status: pass`, and both `exit_status: 0` and `wrapper_exit_status: 0`. A precise blocker is acceptable only with nonzero exit and `failure_stage: vm_mapping`, `sdma_ring_setup`, `sdma_submit`, `timeline_timeout`, or `readback_mismatch`.

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-native-amdev-sdma-transfer.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

OMP task executors record this command for the supervisor; they do not run it in task mode.

### C0A task set 4 native AMDev kernel proof / precise blocker

This is the C0A task set 4 supervisor hardware command. It builds the tinygrad-free native probe, runs `--kernel-proof`, and writes `logs/c0-macos-egpu-minimal-runtime.log`. A future pass requires `kernel_launch_status: pass`, `kernel_elapsed_usec`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0` after CPU readback equals expected `2,3,4,5,6,7,8,9`. The current reviewed outcome (C0A22, `logs/c0l-native-amdev-mec-rs64-pipe-activation.log`) is a launch-eliminated progress: hardware reaches `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `sdma_h2d_status: pass`, `compute_ring_setup_status: pass`, and `compute_hqd_active_status: pass`, and now also `kernel_launch_status: pass` with `compute_doorbell_probe_post doorbell_hit=1` (doorbell consumed) after the MEC RS64 pipe-activation replay into `regCP_MEC_RS64_CNTL` (`mec_rs64_cntl_readback: 0x04000000`). The failure stage advanced to `readback_mismatch` with a halfword byte-swap + partial write in the compute output (only elements `2,3,4,5` of expected `6,7,8,9`). Next blocker: `compute_output_readback_byte_swap`.

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

OMP task executors record this command for the supervisor; they do not run it in task mode.

### C0A23 compute output readback byte-swap diagnostic

This is the C0A23 diagnostic-only command (`docs/superpowers/plans/2026-08-18-compute-output-readback-byte-swap.md`). It builds the tinygrad-free native probe, runs `--kernel-proof`, and writes `logs/c0m-native-amdev-readback-byte-swap.log`. The C0A23 change is instrument-only: the run must be **behavior-identical** to `logs/c0l-native-amdev-mec-rs64-pipe-activation.log` — same `failure_stage: readback_mismatch`, same `observed_hex=0000020000000300000004000000050000000000000000000000000000000000`, same `expected_hex` — and must additionally emit the classifier field `compute_readback_anomaly: anomaly_class=swap_and_partial written_mask=0x0f swapped_mask=0x0f unswapped_match_mask=0x0f`. Confirm `kernel_launch_status: pass`, `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `sdma_h2d_status: pass`, `sdma_d2h_status: pass`, `cpu_comparison_status: fail`, `exit_status: 1`, `wrapper_exit_status: 1`. A `readback_mismatch` with identical `observed_hex` to C0A22 is the expected **UNCHANGED-SIGNATURE** result that confirms the byte-swap/partial-write are stable GPU-side signatures (the SDMA copy engine is byte-faithful; see plan Task 1 grounding). The C0A24 fix lane is selected from the plan Task 2 kernel-store decode, not implemented here.

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0m-native-amdev-readback-byte-swap.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; }'

OMP task executors record this command for the supervisor; they do not run it in task mode.

### C0A24 kernel store byte-swap + partial-write fix

This is the C0A24 fix command (`local://c0a24-kernel-store-fix-plan.md`). It builds the tinygrad-free native probe (now embedding the per-u32 `GLOBAL_STORE_B32` lane kernel, 64 bytes, source id `c0a-minimal-u32-add-one-v2`, dispatched 1 workgroup × 8 lanes via `kDispatchGlobalSizeX=1,kDispatchLocalSizeX=8`), runs `--kernel-proof`, and writes `logs/c0o-native-amdev-kernel-store-fix.log`. Hardware `2026-08-18T17:51:35Z` records `kernel_launch_status: pass`, `sdma_h2d_status: pass`, `sdma_d2h_status: pass`, `mec_rs64_cntl_readback: 0x04000000`, doorbell hit, and removes the byte-swap (`swapped_mask=0x00`) and partial-write (`written_mask=0xff`, all 8 written), but `observed_hex=01000000`×8 (`unswapped_match_mask=0x00`, classifier `other_mismatch`) → `failure_stage: readback_mismatch`, `cpu_comparison_status: fail`, `exit_status: 1`. Next blocker = C0A25 load-path value lane. A future full pass requires `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `failure_stage: none`, `exit_status: 0`, `wrapper_exit_status: 0`, and readback `2,3,4,5,6,7,8,9`.

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0o-native-amdev-kernel-store-fix.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

OMP task executors record this command for the supervisor; they do not run it in task mode.

### C0A25 load-path value-lane fix (PASS)

This is the C0A25 fix command (`docs/tasks/native-r9700-producer/phase-c0a25-load-path-fix.md`, commit `45d7b95`). It builds the tinygrad-free native probe (now embedding the per-u32 `GLOBAL_STORE_B32` lane kernel with the load saddr corrected to the input-VA pair `s[6:7]`, 64 bytes, source id `c0a-minimal-u32-add-one-v3`, sha256 `08fd705ca25c7a1d5531e504eb9905ce84dab9c0a31b7ef6ecfc62475b98f965`, dispatched 1 workgroup × 8 lanes), runs `--kernel-proof`, and writes `logs/c0p-native-amdev-kernel-load-fix.log`. Hardware run `2026-08-18` records **KERNEL_PROOF_PASS**: `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `failure_text: none`, `exit_status: 0`, `kernel_elapsed_usec: 1506`, `sdma_h2d_status: pass`, `sdma_d2h_status: pass`, `mec_rs64_cntl_readback: 0x04000000`, `compute_doorbell_probe_post doorbell_hit=1`, `compute_readback_anomaly: not_run`, and readback matching `expected_output_bytes_hex: 0200000003000000040000000500000006000000070000000800000009000000` (`out[i]=in[i]+1` = `2,3,4,5,6,7,8,9`). This is the passing minimal macOS kernel proof that unblocks the C0 substrate decision (macOS selected for C1).

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0p-native-amdev-kernel-load-fix.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

### C0D MEC doorbell delivery diagnostic proof

This is the diagnostic-only MEC doorbell delivery/ring-fetch command. It builds the tinygrad-free native probe, runs `--kernel-proof`, and writes `logs/c0d-native-amdev-doorbell-delivery.log`. The observed classification from that run was `compute_doorbell_not_consumed`; the log includes `compute_doorbell_probe_status: submitted`, pre/post/timeout snapshots, `doorbell_hit=0`, `hqd_pq_rptr=0x00000000`, `cp_stat=0x00000000`, emitted `failure_stage: kernel_timeline_timeout`, `exit_status: 1`, and `wrapper_exit_status: 1`. That blocker was subsequently resolved: after the MEC RS64 pipe-activation replay (C0A22), the diagnostic doorbell is consumed — see `logs/c0l-native-amdev-mec-rs64-pipe-activation.log` with `kernel_launch_status: pass`, `doorbell_hit=1`, and `failure_stage: readback_mismatch` (byte-swap/partial-write in output).

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0d-native-amdev-doorbell-delivery.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

OMP task executors record this command for the supervisor; they do not run it in task mode.



### C0B gfx12 VM/PTE/TLB prerequisite

This split-out implementation plan and task-doc set completed the previous `failure_stage: vm_mapping` blocker. The SDMA follow-up completed the transfer command above; the latest hardware log records `failure_stage: none` and host-device transfer pass evidence.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

After SDMA ring setup/submission is implemented, supervisor reruns the C0B native AMDev/SDMA transfer proof command above and accepts only a real transfer pass or a later precise nonzero blocker.

### C0 macOS stale libusb-only probe

This negative-control command targets tinygrad's separate `USBIface` path (`USB3.list_devices(0xADD1, 0x0001)`) and does not represent the working local R9700 path. The working path above is TinyGPU.app/IOKit PCI through `APLRemotePCIDevice` and `PCIIface`.

Run from the repo root only when intentionally checking the stale libusb-only assumption against `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp`:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra -I/opt/homebrew/include/libusb-1.0 experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp -L/opt/homebrew/lib -Wl,-rpath,/opt/homebrew/lib -lusb-1.0 -o build/native-r9700-runtime/macos_tinygpu_minimal && ./build/native-r9700-runtime/macos_tinygpu_minimal"; date -u +"timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra -I/opt/homebrew/include/libusb-1.0 experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp -L/opt/homebrew/lib -Wl,-rpath,/opt/homebrew/lib -lusb-1.0 -o build/native-r9700-runtime/macos_tinygpu_minimal && ./build/native-r9700-runtime/macos_tinygpu_minimal; status=$?; printf "exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

This command shape is concrete on the local macOS toolchain: `xcrun --find clang++` resolves to `/Library/Developer/CommandLineTools/usr/bin/clang++`, `/opt/homebrew/include/libusb-1.0/libusb.h` exists, and `/opt/homebrew/lib/libusb-1.0.dylib` exists. It is a negative control only. It must not be used for C0A host↔device transfer acceptance because it targets `USB3.list_devices(0xADD1, 0x0001)`/`USBIface`, not TinyGPU.app/APLRemotePCIDevice/PCIIface.

### C0 Linux ROCm/HIP reference probe

Immediate execution is blocked in this macOS worktree because no Linux ROCm/HIP host is attached to this task and `hipcc` is not installed here (`which hipcc` exits 1). Task set 3 owns provisioning a ROCm-capable AMD Linux host or recording the remote-access blocker. Once that host has the repo worktree and HIP SDK, run this exact reference command from the repo root after task set 3 adds `experiments/native-r9700-runtime/linux_hip_minimal.cpp`:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-linux-hip-minimal-runtime.log; { printf "%s\n" "command: hipcc -std=c++17 -O2 experiments/native-r9700-runtime/linux_hip_minimal.cpp -o build/native-r9700-runtime/linux_hip_minimal && ./build/native-r9700-runtime/linux_hip_minimal"; date -u +"timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; hipcc --version; if command -v rocminfo >/dev/null 2>&1; then rocminfo; fi; hipcc -std=c++17 -O2 experiments/native-r9700-runtime/linux_hip_minimal.cpp -o build/native-r9700-runtime/linux_hip_minimal && ./build/native-r9700-runtime/linux_hip_minimal; status=$?; printf "exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

The probe executable must print HIP device identity/architecture, CPU comparison, host↔device transfer result, kernel timing, and any HIP error text into the captured log.

### C0 handoff documentation check

Supervisor verification command for final C0 handoff documentation:

```sh
git diff --check docs/tasks/native-r9700-producer/README.md docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md
```

OMP task executors record this command for the supervisor; they do not run it in task mode.

### Documentation whitespace check

Use this after task-doc or design-doc edits:

```sh
git diff --check
```

## Commands that must be discovered before execution

| Phase | Command | Owning task set | Status |
|---|---|---|---|
| C0 | macOS eGPU minimal runtime build/run/log command | `phase-c0-runtime-discovery.md` task set 1; continued by `phase-c0a-macos-egpu-runtime-focus.md` task sets 1-4 and `gx1202-compute-dispatch` | Visibility discovery path is TinyGPU.app/IOKit PCI via `APLRemotePCIDevice`/`PCIIface`, with observed `arch gfx1201`; task set 3 transfer proof passes through the C0B native AMDev/SDMA command; task set 4 kernel proof PASSES through the C0A25 command (`logs/c0p-native-amdev-kernel-load-fix.log`): `cpu_comparison_status: pass`, `failure_stage: none`, `exit_status: 0`, exact `out[i]=in[i]+1` readback. C0 substrate decision Done: macOS TinyGPU/AMDev native selected for C1. |replay); the launch `kernel_timeline_timeout` blocker is eliminated and the run advances to `failure_stage: readback_mismatch` (halfword byte-swap + partial write in compute output). The superseded `logs/c0d-native-amdev-doorbell-delivery.log` recorded `kernel_timeline_timeout`/`compute_doorbell_not_consumed`; stale libusb-only probe is retained as a negative control only. |
| C0B | native AMDev/SDMA transfer proof build/run/log command | `phase-c0b-native-amdev-sdma-transfer.md` task set 5 | Exact build/run/log command recorded above for `logs/c0b-native-amdev-sdma-transfer.log`; current native probe has reviewed VM/PTE/TLB pass evidence and SDMA0 7.0.1 queue0 pass evidence with `host_device_transfer_status: pass`, `cpu_comparison_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0` |
| C0 | Linux ROCm/HIP reference build/run/log command | `phase-c0-runtime-discovery.md` task set 1 | Reference fallback; immediate local execution remains blocked pending ROCm Linux host/HIP SDK; provisioned-host command recorded above |
| C1 | native loader/config validation command | `phase-c1-native-producer-parity.md` task set 1 or 2 | Executed by Lane B (task set 2): see "C1 loader (Lane B — weight/config, task set 2)" above; container decided = MLX safetensors dir; loader reads config.json + safetensors headers only |
| C1 | reference-fixture generation command | `phase-c1-native-producer-parity.md` task set 1 or 3 | Executed by Lane B2 (task set 3): see "C1 reference fixtures (Lane B2 — task set 3)" below; deterministic on-disk MLX oracle fixtures landed under `tests/native_r9700/fixtures/` (prompts.json, baseline_r_tokens.json, kv_state.npz, primitives_fixtures.npz, fixtures_schema.json) |
| C1 | native runtime shell validation command | `phase-c1-native-producer-parity.md` task set 1 or 4 | C0 substrate SELECTED (macOS TinyGPU/AMDev native, C0A25); C1 command discovery now in scope under a C1 contract-freeze plan; not yet executed |
| C1 | primitive kernel test commands | `phase-c1-native-producer-parity.md` task set 5 | C0 substrate SELECTED (macOS TinyGPU/AMDev native, C0A25); C1 command discovery now in scope under a C1 contract-freeze plan; not yet executed |
| C1 | attention/RoPE/KV writer test command | `phase-c1-native-producer-parity.md` task set 6 | Exact focused RED/GREEN command recorded below; supervisor expects RED until `native_r9700.attention` implements the frozen Llama-only API and KV parity contract |
| C1 | full-stack native prefill smoke command | `phase-c1-native-producer-parity.md` task set 7 | Exact focused RED/GREEN command recorded below; supervisor expects RED until `native_r9700.prefill` implements the full-layer Llama prefix prefill API and CLI |
| C1 | native KV emitter/load round-trip command | `phase-c1-native-producer-parity.md` task set 8 | Exact focused RED/GREEN command recorded below; supervisor expects RED until `native_r9700.kv_cache` implements the prompt-cache safetensors emitter API/CLI |
| C1 | native producer parity command | `phase-c1-native-producer-parity.md` task set 9 | C0 substrate SELECTED (macOS TinyGPU/AMDev native, C0A25); C1 command discovery now in scope under a C1 contract-freeze plan; not yet executed |
| C2 | mlx-lm wrapper focused test command | `phase-c2-serving-integration.md` task set 1 or 2 | Dependency-blocked by C1 parity; not discovered |
| C2 | fallback/error-state test command | `phase-c2-serving-integration.md` task set 1 or 3 | Dependency-blocked by C1 parity; not discovered |
| C2 | serving integration command | `phase-c2-serving-integration.md` task set 4 | Dependency-blocked by C1 parity; not discovered |
| C2 | oMLX integration command, if in scope | `phase-c2-serving-integration.md` task set 5 or 6 | Dependency-blocked by C1 parity and later scope decision; not discovered |
| C3 | backend prototype command, if approved | `phase-c3-native-backend-decision.md` task set 2 or 4 | Dependency-blocked by C2 evidence and backend decision; not discovered |
| C3 | backend comparison command, if approved | `phase-c3-native-backend-decision.md` task set 5 | Dependency-blocked by C2 evidence and backend decision; not discovered |

### C1 loader (Lane B — weight/config, task set 2)

Selected first native producer weight container: **MLX safetensors directory**
(`mlx_models/meta-Llama-3.2-1B-Instruct`). Rationale: it is the single
self-contained source carrying BOTH the fp16 weights (safetensors header
dtype `F16`) AND the complete `config.json` sidecar (geometry + Llama-3
`rope_scaling`). The F16 GGUF is fp16 but records `rope.freq_base` and not the
Llama-3 `rope_scaling` fields, so it cannot provide exact consumer config
parity on its own (Phase 0 harness patched tinygrad from the MLX sidecar for
exactly this reason).

Loader validation command (reads only `config.json` + safetensors header
records, never model weights):

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.loader \
  --model mlx_models/meta-Llama-3.2-1B-Instruct
```

Expected on the official model dir (reference model lives under
`.worktrees/tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct`
if not present here): exit 0, report lines `num_layers: 16`, `n_kv_heads: 8`,
`head_dim: 64`, `hidden_size: 2048`, `rope_theta: 500000.0`,
`rope_scaling: rope_type=llama3 factor=32.0 ...`, `weight_dtype: F16`,
`config_source: <abs path>/config.json`, and `exit_status: 0`. Unsupported
models/dtypes/missing config exit nonzero with a descriptive `error:` line.

Focused loader tests (no model weights required — use a small on-disk config
fixture):

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
```

### C1 native runner runtime shell (Lane A — task set 4)

The `native_r9700` runner shell (`native_r9700/runtime.h`, `runtime.cpp`,
`runner.cpp`) refactors the proven C0 probe lifecycle into a narrow,
harness-callable shell. Its no-hardware contract mode `--lifecycle-dry-run`
exercises lifecycle ordering, the frozen 24-byte kernarg layout
`{output_va@0, input_va@8, scalar_va@16, scalar:u32@24}`, the SDMA/PM4 packet
encodings, and standardized log writing under `logs/` — no TinyGPU socket.

Runtime shell build command (build only):

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/runtime.cpp native_r9700/runner.cpp \
  -o build/native-r9700-runtime/native_r9700_runner
```

Runtime shell hardware-free run + log command:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
mkdir -p logs
log=logs/c1-runner-lifecycle-dry-run.log
{ printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -o build/native-r9700-runtime/native_r9700_runner && build/native-r9700-runtime/native_r9700_runner --lifecycle-dry-run"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -o build/native-r9700-runtime/native_r9700_runner && build/native-r9700-runtime/native_r9700_runner --lifecycle-dry-run; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"
```

Expected: `status: pass`, `exit_status: 0`, `wrapper_exit_status: 0`,
`kernarg_layout_offsets: output_va=0,input_va=8,scalar_va=16,scalar=24`,
`kernarg_byte_size: 24`, `lifecycle_reinit_rejected: yes`,
`lifecycle_skip_rejected: yes`, and a timestamped log under `logs/`. The
focused pytest compiles the runner and runs `--lifecycle-dry-run`:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -q   # C0 regression (23 passed)
```

Focused runner contract tests (compile the runner + exercise the no-hardware
lifecycle contract):

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
```

### C1 native tensor primitives (Lane A — task set 5)

The `native_r9700/primitives.py` module (Lane A2, marker `c1w2-lane-a2`)
provides the narrow host-side primitives for the first Llama-3.2-1B-Instruct
fp16 prefill slices: fp16↔fp32 copy/cast (`cast_fp32_to_fp16`,
`cast_fp16_to_fp32`), fp16×fp16→fp16 matmul with an fp32 accumulator
(`matmul`), Llama RMSNorm (`rms_norm`, eps=1e-5 from the MLX config sidecar),
and SiLU (`silu`). These are the CPU/numpy host-reference kernels the native
GPU kernels are checked against — the C++ `RuntimeSession` shell is a
hardware-free lifecycle contract and performs no tensor math, so it is not the
matmul substrate. Unsupported shapes/dtypes fail loudly
(`UnsupportedShapeError`/`UnsupportedDtypeError`). Focused correctness tests
compare each primitive against a deterministic host oracle; the fixture-
consumer seam reads Lane B2's on-disk MLX reference fixture
`tests/native_r9700/fixtures/primitives_fixtures.npz` and `pytest.skip`s when
it is absent (so this focused suite is green independently of Lane B2).

Focused primitive tests (green with or without Lane B2's fixtures; skips only
when the fixture file is absent):

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_primitives.py -v
```

Expected **current state** (Lane B2 fixtures landed at
`tests/native_r9700/fixtures/primitives_fixtures.npz`): **23 passed, 0 skipped**
(19 focused oracle tests + 4 seam comparisons against the MLX reference
tensors, all bit-exact). Without the fixture file present the same command
yields **19 passed, 4 skipped** (the seam `pytest.skip`s with "Lane B2
reference fixture ... not found"), so the focused suite is green independently
of Lane B2.

Full-suite regression guard (per `phase-c1-native-producer-parity.md` task set 5
validation):

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v
```

### C1 attention/RoPE/KV writer contract (task set 6)

Focused RED/GREEN contract tests for the future `native_r9700.attention`
module. The tests lock the Llama-only C1-6 public API, S-1 prompt-prefix
splitting, Llama-3 split-half RoPE math and sidecar scaling, prompt-0 layer-0
fp16 K/V shape `(1,8,5,64)`, bounded deltas against `kv_state.npz`, and loud
failure for wrong `rope_scaling`. Qwen3.8-27B is intentionally deferred and not
part of this command.

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_attention_kv.py -v
```

Current observed RED before implementation: collection succeeded and the command
exited `1` with 9 failures, all caused by missing `native_r9700.attention`.
Current observed GREEN after task set 6: command exits `0` with **9 passed**.
Supervisor CLI smoke:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.attention \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --layer 0 \
  --prompt-name prompt-0 \
  --log logs/c1-attention-kv-layer0.log
```

Observed: exit `0`; log includes `layer=0`, `n_prefix=5`, K/V max/mean deltas
(`K max=0.00390625`, `K mean=0.00013293116`, `V max=0.00024414062`,
`V mean=1.6966555e-05`), and `exit_status: 0`.

### C1 full-layer prefix prefill contract (task set 7)

Focused RED/GREEN contract tests for the future `native_r9700.prefill` module.
The tests lock the narrow Llama-3.2-1B-Instruct prompt-prefix prefill API:
prompt-0 S-1 prefix tokens from `prompts.json`, all 16 ordered layers, fp16 K/V
arrays shaped `(1,8,5,64)`, layer-0/layer-15 bounded deltas against
`kv_state.npz`, CLI NPZ emission, review-log fields, and loud failure for
prefixes shorter than two tokens. Qwen support, partial-layer prefill, emitter
safetensors, parity harness wiring, and C++ runtime integration remain outside
this task set.

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py -v
```

Observed RED before implementation: collection succeeded and the focused command
exited `1` with 5 failures, all caused by missing `native_r9700.prefill`.
Observed GREEN after task set 7: focused command exits `0` with **5 passed**.
Supervisor CLI smoke:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.prefill \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --prompt-name prompt-0 \
  --out logs/c1-prefill-prompt0.npz \
  --log logs/c1-prefill-prompt0.log
```

Observed: exit `0`; log includes `n_prefix: 5`, `num_layers: 16`,
`output: logs/c1-prefill-prompt0.npz`, layer0/layer15 K/V max/mean deltas,
and `exit_status: 0`.

### C1 native KV prompt-cache emitter contract (task set 8)

Focused RED/GREEN contract tests for the future `native_r9700.kv_cache` module.
The tests lock the mlx-lm prompt-cache safetensors ABI for C1 prefill results:
16 ordered `KVCache` layers, tensor keys `{i}.0`/`{i}.1`, metadata keys
`0.{i}`, `2.{i}`, `1.offset`, `1.num_layers`, `1.n_kv_heads`, and
`1.head_dim`, fixture NPZ conversion, loud validation failures for malformed
K/V arrays, and CLI conversion/logging from a prefill NPZ. Qwen support,
decode/parity-harness wiring, C2 integration, and C++ runtime integration
remain outside this task set.

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_kv_cache.py -v
```

Expected RED before implementation: collection succeeds and the focused command
fails with a clear missing `native_r9700.kv_cache` module/API failure.

### C1 reference fixtures (Lane B2 — task set 3)

The `native_r9700/ref_fixtures.py` module (Lane B2, marker `c1w2-lane-b2`)
produces deterministic, small on-disk MLX oracle fixtures under
`tests/native_r9700/fixtures/` that Lane A2's primitive seam and later task
sets consume for CPU/MLX comparison. The helper code is pure stdlib + numpy
(NO tinygrad); mlx-lm is the reference oracle only during `--generate` (native
baseline R tokens + per-layer KV state), mirroring the Phase 0 native baseline
in `tinygrad_kv_worker/harness.py`. Fixture files:

- `prompts.json` — prompt texts, mlx token ids, S per Phase 0 prompt
  (prompt-0 S=6, prompt-1 S=222, prompt-2 S=661).
- `baseline_r_tokens.json` — mlx-lm native-baseline R token ids per prompt.
- `kv_state.npz` — per-layer K/V for prompt-0 honoring the S-1 prefix +
  final-token injection: `(1,8,5,64)` fp16, 16 layers, `final_token_id=374`.
- `primitives_fixtures.npz` — deterministic small intermediate tensors for the
  primitive seam (cast, matmul, rms_norm, silu) per the Lane A2-agreed schema.
- `fixtures_schema.json` — self-describing schema + sha256 digests.

Regenerable by command (supervisor runs this; the default `--model` is
`mlx_models/meta-Llama-3.2-1B-Instruct` — pass the reference safetensors dir
explicitly when `mlx_models/` is absent in this worktree):

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.ref_fixtures \
  --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures
```

Expected: writes 5 fixture files to `tests/native_r9700/fixtures/` and prints
their paths; regeneration is byte-for-byte deterministic.

Focused fixture tests (schema, determinism, size; `pytest.skip`s gracefully
when the fixture dir is absent so the focused suite stays green
independently):

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py -v
```

Combined focused suite (Lane A2 + Lane B2 — exercises the primitive seam and
the reference fixtures together):

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
```

Expected current state (Lane B2 fixtures landed): **57 passed, 0 skipped**
(19 loader + 8 runtime + 19 focused primitive + 4 `TestPrimitiveFixtureSeam`
comparisons consuming `primitives_fixtures.npz` bit-exact + 7 reference-fixture
tests = 57).

## Log requirements for all GPU/native runs

Every GPU/native run must write a reviewable local log under `logs/` or record an explicit remote log artifact path. Logs must include:

- command line;
- timestamp;
- runtime substrate and device identity if discoverable;
- model/config path or note that no model is used;
- prompt length or input shape;
- output comparison result or digest;
- exit status;
- failure traceback/error text when failing.

Logs and model files must not be committed.

## Gate reminders

- Producer-swap acceptance is token-exact `P == R`, not semantic equivalence.
- mlx-lm injected decode uses an imported `S-1` prefix cache plus the final prompt token.
- Llama-3 RoPE scaling must match the MLX sidecar config.
- Path C native producer code must not depend on tinygrad.
- C1 native producer commands are now unblocked (C0 substrate SELECTED: macOS TinyGPU/AMDev native, recorded by the C0A25 minimal kernel proof `logs/c0p-native-amdev-kernel-load-fix.log` with `cpu_comparison_status: pass`, `failure_stage: none`, `exit_status: 0`). Begin the C1 contract freeze and native producer parity under a separate C1 plan; Linux ROCm/HIP remains the reference/deferred fallback.
```

## File: `.superpowers/swarm/progress.md`

```text
# Swarm Progress Ledger — Path A Phase 0

Phase doc: `docs/tasks/tinygrad-kv-worker/phase-0-parity.md`
Work boundary: `…/egpu/.worktrees/tinygrad-kv-worker-phase0` on branch `feature/tinygrad-kv-worker-phase0`

| Task | Status | Owner | Dependencies | Report | Evidence | Blocker |
|---|---|---|---|---|---|---|
| 1. Exporter implementation | Done | ExporterImpl | — | `.superpowers/swarm/reports/task-1-exporter.md` | Verified source: mlx-lm 0.31.3 `_BaseCache.meta_state` setter raises → per-layer `meta_state=str(N)` not loadable; deviation legit; exported offset is reconstructed from state shape; global metadata carries `offset=str(N)`; 8/8 tests pass on top | |
| 2. Exporter unit test (no GPU) | Done | UnitTestAgent | Task 1 | `.superpowers/swarm/reports/task-2-unit-test.md` | `python3 -m pytest tests/test_exporter.py -v` → 8 passed; exporter untouched, no bugs | |
| 3. Injection harness + numeric parity gate | Done (fp16 PASS) | HarnessFix | Tasks 1, 2 | `.superpowers/swarm/reports/task-3-harness.md`, `fix-phase0-harness.md`, `docs/path-a-validation-results.md` | Geometry fix applied + verified (`export()` derives n_kv_heads/head_dim/num_layers from tensors; real Llama 3.2 1B head_dim=64). Initial cached-GGUF run failed because producer was Q6_K (ftype=18, imatrix) vs mlx fp16. Superseding run used official meta fp16 on both sides: F16 GGUF producer `mlx_models/meta-Llama-3.2-1B-Instruct.F16.gguf` + mlx consumer `mlx_models/meta-Llama-3.2-1B-Instruct`. Harness applies MLX sidecar Llama-3 RoPE scaling (GGUF lacks `rope_scaling`) and exports `S-1` prefix cache for `generate_step` (final prompt token supplied as suffix). Final gate PASS: P==R for all 3 prompts (S=6/222/661), log `logs/runs/20260816-191810-659350000_meta-f16-final.log`; per-layer suite-level worst-case deltas recorded in validation report. Logging infrastructure writes every harness run under `logs/runs/`; CPU tests cover exporter, logging, RoPE config, report output, delta aggregation, and injected cache split. | none |
 
## Task 3 final note (supersedes earlier Q6_K negative run)
- **GGUF + AMD present**: Llama 3.2 1B loads on AMD here (USB4/TinyGPU, arch gfx1201 = AI PRO R9700).
- **Geometry fixed**: real Llama 3.2 1B is `n_kv_heads=8`, `head_dim=64`, `num_layers=16`; the harness derives geometry from actual block-cache tensor shapes.
- **Initial negative finding preserved**: tinygrad's cached model-zoo GGUF was Q6_K (`general.file_type=18`, imatrix metadata), so Q6_K-vs-fp16 produced P!=R and large deltas. That was a weight-precision confound, not an interchange defect.
- **Official fp16 parity now proven**: official `meta-llama/Llama-3.2-1B-Instruct` weights converted to mlx fp16 and F16 GGUF; final run `20260816-191810-659350000_meta-f16-final.log` reports P==R for all prompts.
- **Two contract fixes were required**: Llama-3 RoPE scaling comes from the MLX `config.json` sidecar because the generated GGUF records `rope.freq_base` but not `rope_scaling`; mlx-lm `generate_step` always processes its supplied prompt, so the injected cache must cover `S-1` and the last prompt token must be supplied separately.

---

# Swarm Progress Ledger — Native R9700 Producer

Phase docs: `docs/tasks/native-r9700-producer/`
Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer` (fallback linked worktree; source checkout was `main`).
Supervisor artifact: `.superpowers/swarm/native-r9700-producer-supervisor.md`
Baseline evidence: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v` -> `17 passed, 2 warnings`.

## Orchestration map

- Sequential blockers: C0-1 -> C0 proof lanes -> C0-5/6 -> C1; C1-1 -> C1-2/3/4 -> C1 kernel ladder -> C1-9/10 -> C2; C2-1 -> C2 wrapper/test/scope -> C2-4/7 -> C3; C3 evidence/decision before prototype.
- Parallel lanes: initial continuation is serial macOS eGPU runtime work (`phase-c0a-macos-egpu-runtime-focus.md`); Linux HIP remains a reference fallback. Earlier C0-2/C0-3/C0-4 proof lanes are complete/blocked as recorded; C1-1 unblocked by macOS substrate selection (ADR 0004), after which C1-2+/-4 ran in parallel (Wave 1) and C1-3+/-5 ran in parallel (Wave 2), all Done; C1-6 -> C1-7 -> C1-8 are sequential kernel-ladder waves (each consumes the prior + C1-3 fixtures); C2-2/C2-3/C2-5 can run in parallel only after C2-1.
- Shared contracts/artifacts: `docs/tasks/native-r9700-producer/validation-commands.md`, `logs/`, `.superpowers/swarm/reports/`, KV interchange format, Phase 0 prompt suite, fallback worktree/branch above.
- Coordination risks: only substrate-decision owner chooses C0 outcome; no DwarfStar dependency/fork; no C1/C2/C3 execution before gates; logs and model files remain uncommitted.
- Verification gates: exact commands recorded in `validation-commands.md`; supervisor runs focused commands and review after each wave.
- Commit boundary: supervisor checkpoint commits after reviewed/verified waves only; agents never run git; push remains user responsibility.

| Task | Status | Owner | Dependencies | Report | Evidence | Blocker |
|---|---|---|---|---|---|---|
| C0-1. Validation and source-layout discovery | Done | C0ValidationLayout | Phase 0 baseline | `.superpowers/swarm/reports/c0-task-1-validation-layout.md` | Source root `experiments/native-r9700-runtime/` recorded; C0 macOS/Linux/doc commands recorded in `validation-commands.md`; supervisor `git diff --check docs/tasks/native-r9700-producer/validation-commands.md docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md` passed. | |
| C0-2. macOS eGPU minimal runtime probe | Blocked | C0MacOSEGPU | C0-1 | `.superpowers/swarm/reports/c0-task-2-macos-egpu.md` | Tinygrad-free libusb probe source added at `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp`; supervisor macOS command compiled and ran, writing `logs/c0-macos-egpu-minimal-runtime.log`; exit 3 with `tinygpu_device_count: 0`. | No matching TinyGPU USB device was visible in this worktree run, and native TinyGPU DMA mapping, command queue, and kernel dispatch ABI remain unpinned for tinygrad-free use. |
| C0-3. Linux ROCm HIP reference probe | Blocked | C0LinuxHIP | C0-1 | `.superpowers/swarm/reports/c0-task-3-linux-hip.md` | Tinygrad-free HIP vector-add source added at `experiments/native-r9700-runtime/linux_hip_minimal.cpp`; provisioned-host command remains the C0 Linux command in `docs/tasks/native-r9700-producer/validation-commands.md`; supervisor confirmed local `hipcc` is absent (`which hipcc` exit 1). | No immediately usable ROCm-capable AMD Linux host/toolchain; blocked pending Linux host with HIP SDK/repo checkout. |
| C0-4. DwarfStar runtime reference extraction | Done | C0DwarfStar | C0-1 | `.superpowers/swarm/reports/c0-task-4-dwarfstar.md` | DwarfStar use/non-use note recorded in `docs/tasks/native-r9700-producer/dwarfstar-reference-notes.md`; no vendoring/dependency/architecture adoption; reviewer found no Critical/Important issue beyond ledger reconciliation. | |
| C0-5. Runtime substrate decision | Blocked | C0SubstrateDecision | C0-2, C0-3, C0-4 | `.superpowers/swarm/reports/c0-task-5-substrate-decision.md` | Supervisor `git diff --check docs/DESIGN.md docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md` passed; decision recorded no substrate selected. | C1 must not start until a passing macOS TinyGPU or Linux HIP probe selects a substrate or actionable split plan. |
| C0-6. C0 report and handoff update | Done | C0Handoff | C0-5 | `.superpowers/swarm/reports/c0-task-6-handoff.md` | Final C0 handoff recorded selected state `blocked`; supervisor `git diff --check docs/tasks/native-r9700-producer/README.md docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md` passed. | |
| C0A-1. Mac-first runtime focus plan | Done | Main | C0-5/C0-6 plus user steering | `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` | Created macOS-first task packets: device visibility rerun, TinyGPU ABI pinning note, host-device transfer proof, minimal kernel launch proof, and mac-focused C0 decision rerun. Linux HIP remains a reference fallback. | |
| C0A-2. Mac device visibility rerun | Done | C0AMacDeviceVisibility | C0A-1 | `.superpowers/swarm/reports/c0a-task-1-mac-device-visibility.md` | Corrected: stale libusb-only probe saw `tinygpu_device_count: 0`, but supervisor verified the working Phase 0 path: `System.list_devices(...) -> APLRemotePCIDevice '1002:7551'`, `Device['AMD'] -> PCIIface`, `arch gfx1201`, `pcibus usb4`; user-provided command `JITBEAM=2 DEV=AMD python3 -m tinygrad.llm` uses this substrate. | |
| C0A-3. TinyGPU ABI pinning note | Done | C0ATinyGPUABI | C0A-2 | `.superpowers/swarm/reports/c0a-task-2-tinygpu-abi.md` | ABI/source contract pinned to TinyGPU.app/APLRemotePCIDevice/PCIIface, RemoteCmd/sysmem fd/BAR/MMIO/config operations, AMDev memory-manager ownership, SDMA transfer path, and kernel-launch boundary. Reviewer found stale libusb validation wording; supervisor corrected docs and `git diff --check` passed. | |
| C0A-4. Host device transfer proof | Done | C0ATransferProof / C0BSDMATransfer / C0BSDMAHardware / Main | C0A-3 | `.superpowers/swarm/reports/c0a-task-3-transfer-proof.md`; `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md`; `.superpowers/swarm/reports/c0b-vm-task-4-transfer-resume.md`; `.superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md` | Native TinyGPU.app/APLRemotePCIDevice transfer proof passes: supervisor focused pytest passed `11 passed in 9.94s`; `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z` records `sdma_ip_version: 7.0.1`, `sdma_queue_setup_status: pass`, `sdma_submit_status: pass`, `sdma_timeline_status: pass`, `transfer_byte_count: 32`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`. | |
| C0A-5. Minimal kernel launch proof | Blocked | Main / C0AKernelProof / C0AKernelImpl / C0A compute dispatch swarm | C0A-3, C0A-4 | `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md`; `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`; `.superpowers/swarm/reports/c0a-compute-task-5-pm4-review.md`; `.superpowers/swarm/reports/c0a-compute-task-5-pm4-rereview.md`; `.superpowers/swarm/reports/c0a-compute-split-decision.md` | Focused no-hardware contract now passes `17 passed in 19.87s`; latest hardware command wrote `logs/c0c-native-amdev-kernel-dispatch.log` at `2026-08-17T17:53:08Z` and exited nonzero after reaching `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, `arch: gfx1201`, `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `sdma_h2d_status: pass`, `vmid0_context_status: pass`, `vm_gc_context_status: pass`, `mm_tlb_flush_status: pass`, `gc_tlb_flush_status: pass`, `compute_ring_setup_status: pass`, and `compute_hqd_active_status: pass`; it then emitted `failure_stage: kernel_timeline_timeout` with diagnostic `rptr=0`, `wptr=59`, `hqd_pq_rptr=0`, `hqd_pq_doorbell_control=0x40000018`, and `cp_stat=0`. PM4 re-review found 0 Critical/Important/Minor findings and accepted inferred blocker `compute_doorbell_not_consumed`. | Blocked on MEC doorbell delivery/ring fetch; C1 must not start until pass tokens exist or a user-approved fallback/split changes the path. |
| C0A-6. Mac focused C0 decision rerun | Blocked | Main | C0A-5 | `.superpowers/swarm/reports/c0a-task-5-mac-decision-rerun.md`; `.superpowers/swarm/reports/c0a-compute-split-decision.md` | Compute split decision recommends continuing the native macOS GFX port with a named MEC doorbell delivery primitive; C0 scope remains unchanged. | Blocked until macOS kernel proof produces CPU-verified pass tokens, or user approval changes the C0 substrate path. |
| C0B-1. RED native transfer contract tests | Done | C0BRedContract | Native AMDev/SDMA spec approved | `.superpowers/swarm/reports/c0b-task-1-red-contract.md` | Supervisor `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` exited `1` with both tests failing on `AssertionError: native transfer probe source missing`; `C0BRedReviewer` accepted with no findings. | |
| C0B-2. RemoteCmd transport self-tests | Done | C0BRemoteCmd | C0B-1 | `.superpowers/swarm/reports/c0b-task-2-remote-pci.md` | Supervisor pytest `tests/test_native_amdev_transfer_contract.py -v` passed `2 passed in 0.68s`; `C0BRemoteCmdReviewer` accepted exact RemoteCmd order, frame hex, failure paths, provenance, and no runtime tinygrad/libusb/hardware path. | |
| C0B-3. TinyGPU discovery smoke | Done | C0BDiscovery | C0B-2 | `.superpowers/swarm/reports/c0b-task-3-discovery.md` | Supervisor pytest passed `3 passed in 1.59s`; discovery command wrote `logs/c0b-discovery-smoke.log` and exited `0` with `pci_id: 1002:7551`, BAR0 `268435456`, BAR2 `2097152`, BAR5 `524288`, `vram_size_bytes: 34208743424`, `host_device_transfer_status: not_run`, and `failure_stage: none`; reviewer accepted after fixes. | |
| C0B-4. VM/sysmem mapping port | Done | C0BVMSysmem | C0B-3 | `.superpowers/swarm/reports/c0b-task-4-vm-sysmem.md` | Supervisor pytest passed `4 passed in 2.26s`; hardware `--transfer-proof` smoke reached MAP_SYSMEM_FD staging/readback page lists and failed closed with `failure_stage: vm_mapping` before SDMA/PTE/TLB work after MAP_SYSMEM_FD frame/lifetime review fixes; re-review accepted. | |
| C0B-4.5. gfx12 VM/PTE/TLB prerequisite | Done | Main / C0BVmRedContract / C0BVmSelfTests / C0BVmHardwareMapping / C0BVmPhase2Reviewer / C0BVmFinalReviewer | C0B-4, C0B-5 blocker | `docs/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md`; `docs/tasks/native-r9700-gfx12-vm-pte-tlb/README.md`; `.superpowers/swarm/reports/c0b-vm-task-1-contracts.md`; `.superpowers/swarm/reports/c0b-vm-task-2-selftests.md`; `.superpowers/swarm/reports/c0b-vm-phase1-review.md`; `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md`; `.superpowers/swarm/reports/c0b-vm-phase2-review.md`; `.superpowers/swarm/reports/c0b-vm-task-4-transfer-resume.md`; `.superpowers/swarm/reports/c0b-vm-final-review.md` | Phase 1 reviewed: RED focused pytest exited `1` on absent VM self-tests; GREEN focused pytest reported `8 passed in 4.87s`; reviewer found no findings. Phase 2 implemented fixed IP discovery/register/MMIO/page-table/MMHUB VMID0/TLB setup; final supervisor pytest passed `8 passed in 6.54s`; final hardware transfer command at that time exited `1` at precise post-VM `failure_stage: sdma_ring_setup`; phase 2 and final reviewers accepted with no Critical/Important/Minor findings. | VM/PTE/TLB prerequisite is complete; C0B-5 was later completed by the SDMA queue0 transfer proof. |
| C0B-5. SDMA transfer proof | Done | Main / C0BSDMAHardware | C0B-4.5 | `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md`; `.superpowers/swarm/reports/c0b-vm-task-4-transfer-resume.md`; `.superpowers/sdd/2026-08-17-native-sdma-ring-transfer/task-3-report.md`; `.superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md`; `.superpowers/swarm/reports/c0b-sdma-reset-rereview.md` | Task 3 implemented SDMA0 7.0.1 queue0 IP logging, `sdma_control` sysmem at VA `0x0000200000003000`, fourth VM PTB leaf/readback validation, source-grounded `gc_12_0_0` `regSDMA0_QUEUE0_*` registers, pre-setup queue0 disable plus `regGRBM_SOFT_RESET.soft_reset_sdma0` for repeated-run safety, BAR2 doorbell submission, bounded fence polling, and CPU comparison. Supervisor fixed the initial stale SDMA 4.4.2 assumption and repeated-run stale `RB_WPTR=0x48` regression after RED/root-cause evidence, focused pytest passed `11 passed in 9.94s`, and hardware transfer proof exited `0` with all pass tokens in `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z`. | |
| C0B-6. Review and C0 handoff | Done | Main | C0B-5 pass | `.superpowers/swarm/reports/c0b-task-6-review-handoff.md`; `.superpowers/swarm/reports/c0b-vm-task-4-transfer-resume.md`; `.superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md`; `.superpowers/swarm/reports/c0b-sdma-final-review.md`; `.superpowers/swarm/reports/c0b-sdma-reset-rereview.md` | C0B handoff now records host-device transfer pass evidence and reset re-review acceptance; C0A minimal kernel proof is unblocked. C1/C2/C3 remain blocked until kernel proof and C0 decision rerun select a substrate or actionable split. | |
| C1-1. C1 contract freeze and validation discovery | Done | Main | C0-6 | `docs/tasks/native-r9700-producer/phase-c1-native-producer-parity.md`; `docs/superpowers/plans/2026-08-18-c1-kickoff.md`; `docs/tasks/native-r9700-producer/README.md` | C1 contract FROZEN and committed `34a09ce` (docs-only): macOS TinyGPU.app/APLRemotePCIDevice/PCIIface native AMDev substrate; source root `native_r9700/`, test root `tests/native_r9700/`; 24-byte kernarg layout; KV interchange = mlx-lm prompt-cache `.safetensors`; P==R token-exact gate; no tinygrad in producer path; RoPE from MLX config sidecar; `S-1` + final-token injection. C0 selected macOS substrate (ADR 0004, commit `dd90595`). Baseline `pytest tests -v` = 67 passed. | |
| C1-2. Weight config container decision and loader | Done | C1WeightLoader (Lane B) | C1-1 | `.superpowers/swarm/reports/c1k-task-2-weight-loader.md`; `.superpowers/swarm/reports/c1k-task-2-review.md` | Container decision = MLX safetensors dir `mlx_models/meta-Llama-3.2-1B-Instruct` (fp16 weights + `config.json` sidecar with Llama-3 `rope_scaling`; GGUF lacks `rope_scaling` so cannot give exact config parity). Files: `native_r9700/__init__.py`, `config.py`, `loader.py`, `tests/native_r9700/test_loader.py` (19 tests). Loader reports geometry 16/8/64/2048, rope_theta 500000, rope_scaling llama3, weight_dtype F16, provenance from on-disk config the Phase 0 MLX consumer reads. Reviewer C1LoaderReviewer -> APPROVE. `pytest tests/native_r9700 -v` (combined) = 27 passed. Lane B reviewer 3 Minor notes (recorded, owner + evidence): (1) loader.py:54-72 speculative shard/index machinery for single-file first model, confidence 0.7 — defer/trim when a sharded second model exists; (2) config.py:218-221 `max_position_embeddings` not validated against SUPPORTED_* constant, confidence 0.6 — out-of-contract consistency improvement; (3) loader.py:176-179 `format_report` hardcodes validated geometry literals while reading others from cfg, confidence 0.8 — could read from cfg/SUPPORTED_* for drift-resistance. | |
| C1-3. CPU MLX reference fixtures | Done | C1RefFixtures (Wave 2, Lane B2) | C1-1, C1-2, C1-4 | `.superpowers/swarm/reports/c1k-task-3-reference-fixtures.md`; `.superpowers/swarm/reports/c1k-wave2-review.md` | Wave 2 done. `native_r9700/ref_fixtures.py` (pure stdlib+numpy, no tinygrad; mlx-lm only as generation oracle), `tests/native_r9700/test_ref_fixtures.py` (7 tests), committed deterministic fixtures under `tests/native_r9700/fixtures/`: prompts.json (prompt-0 S=6, prompt-1 S=222, prompt-2 S=661 token ids), baseline_r_tokens.json (mlx-lm greedy R tokens), kv_state.npz (per-layer K/V (1,8,5,64) fp16, 16 layers, S-1 + final_token_id=374 injection contract), primitives_fixtures.npz (11-key seam schema consumed by Lane A2 bit-exact), fixtures_schema.json. Regenerable byte-for-byte (sha256-identical; supervisor verified). Fixtures small (KV ~160 KB, no weights). Supervisor verified: combined `tests/native_r9700 -q` 57 passed; full `tests -v` 97 passed. Wave 2 reviewer C1Wave2Review -> APPROVE. Wave 2 Minor (recorded, owner C1RefFixtures + evidence `c1k-wave2-review.md`): `rms_eps` stored as fp32 while ground truth uses fp64 1e-5 (probe ref_fixtures.py:146-156) — semantic inconsistency, zero observable impact (bit-exact verified), Info-level schema-exactness note, not actionable; leave stored fp32 (matches what the seam consumer passes) or document the narrowing. | |
| C1-4. Runtime wrapper and logged execution shell | Done | C1RunnerScaffold (Lane A) / C1RunnerFix / C1RunnerFix2 / C1RunnerReviewer / C1RunnerRereview | C1-1 | `.superpowers/swarm/reports/c1k-task-4-runner-scaffold.md`; `.superpowers/swarm/reports/c1k-task-4-runner-fix.md`; `.superpowers/swarm/reports/c1k-task-4-runner-review-fix.md`; `.superpowers/swarm/reports/c1k-task-4-runner-rereview.md`; report `c1k-task-4-review.md` does NOT exist — reviewer findings in `agent://C1RunnerReviewer` | Files: `native_r9700/runtime.h`, `runtime.cpp`, `runner.cpp`, `tests/native_r9700/test_runtime_contract.py`. API `native_r9700::RuntimeSession` with `initialize/allocate_buffers/copy_input/load_kernel/write_kernargs/dispatch_and_poll/readback_and_compare/cleanup/dry_run`. Reviewer C1RunnerReviewer -> CHANGES_REQUIRED (3 Important + 1 Minor); fix agents (C1RunnerFix, C1RunnerFix2) ported C0 probe encodings byte-faithfully: SDMA linear-copy `[0x000001, byte_count-1U, 0U, src_lo, src_hi, dst_lo, dst_hi]` + fence `[kFenceHeader=0x00030005, fence_va_lo, fence_va_hi, value]` (11 dwords), PM4 compute dispatch 12 packets/59 dwords (`pm4_packet3` first dword `0xc0065800`), removed dead `kSdmaFenceValue` and never-populated RAII members, hardware stubs made honest (deferred to task sets 5-8). Re-review C1RunnerRereview -> APPROVE, 0 findings, 96% confidence. Probe untouched (`git diff --stat experiments/...probe.cpp` empty); `git diff --check` clean. Supervisor verified: `tests/native_r9700 -v` 27 passed; `test_native_amdev_transfer_contract.py -q` 23 passed; `tests -v` 67 passed; build warning-free (exit 0); `--lifecycle-dry-run` exit 0 with sdma_copy_dword_count 11, pm4_dispatch_dword_count 59, sdma_copy_header_hex 00000001, pm4_dispatch_first_dword_hex c0065800, lifecycle_reinit_rejected yes, lifecycle_skip_rejected yes. | |
| C1-5. Native tensor primitives | Done | C1Primitives (Wave 2, Lane A2) | C1-1, C1-2, C1-3, C1-4 | `.superpowers/swarm/reports/c1k-task-5-primitives.md`; `.superpowers/swarm/reports/c1k-wave2-review.md` | Wave 2 done. `native_r9700/primitives.py`: narrow fp16 host kernels (`cast_fp32_to_fp16`/`cast_fp16_to_fp32` exact widening / round-to-nearest, `matmul` fp16x fp16→fp16 fp32-accumulate single-round, `rms_norm` Llama eps=1e-5 fp32-internal per-row, `silu` fp32-internal), each with loud `UnsupportedDtypeError`/`UnsupportedShapeError` rejection; no tinygrad; no GPU execution claimed (CPU/numpy host reference is substrate-correct — the C++ RuntimeSession performs no tensor math). `tests/native_r9700/test_primitives.py`: 19 focused oracle tests + 4 `TestPrimitiveFixtureSeam` tests reading Lane B2 `primitives_fixtures.npz` (cast/matmul bit-exact, rms/silu within 1-fp16-ulp; pytest.skip when fixtures absent). Observed error bounds all under 1e-3 fp16 probe tolerance (matmul ~1.7e-6, rms ~1.3e-4, silu ~1e-4). Supervisor verified: combined `tests/native_r9700 -q` 57 passed; full `tests -v` 97 passed. Wave 2 reviewer C1Wave2Review -> APPROVE. | |
| C1-6. Attention RoPE KV writer path | Done | Main / C1AttentionKV / C1AttentionRed / C1AttentionImpl / C1AttentionReview | C1-1, C1-2, C1-3, C1-4, C1-5 | `.superpowers/swarm/reports/c1-task-6-attention-kv.md`; `.superpowers/swarm/reports/c1-task-6-attention-kv-red.md`; `.superpowers/swarm/reports/c1-task-6-attention-kv-review.md`; `agent://C1LlamaAttentionScout`; `agent://C1QwenTargetScout` | RED observed: focused pytest exited 1 with 9 expected failures from missing `native_r9700.attention`. GREEN implementation added `native_r9700/attention.py`; focused pytest exits 0: 9 passed; CLI smoke wrote `logs/c1-attention-kv-layer0.log` with layer0 prompt-0 deltas K max 0.00390625, K mean 0.00013293116, V max 0.00024414062, V mean 1.6966555e-05, exit_status 0; combined `tests/native_r9700 -v` exits 0: 66 passed; full `tests -v` exits 0: 106 passed, 2 warnings; `git diff --check` clean. Reviewer approved 0 Critical/Important/Minor. Qwen3.8-27B local MLX target is recognized as deferred/unsupported for the C1 Llama ladder: `qwen3_5`, mlx-vlm VLM, 4-bit affine, hybrid linear/full attention, non-C1 KV schema. | |
| C1-7. Full layer stack prefill path | Done | Main / C1PrefillRed / C1PrefillImpl / C1PrefillReview | C1-5, C1-6 | `.superpowers/swarm/reports/c1-task-7-full-prefill.md`; `.superpowers/swarm/reports/c1-task-7-full-prefill-red.md`; `.superpowers/swarm/reports/c1-task-7-full-prefill-review.md`; `.superpowers/swarm/reports/c1-task-7-review-package.md` | Added `native_r9700/prefill.py`: narrow Llama 3.2 1B CPU/NumPy+safetensors all-16-layer prefix prefill, no tinygrad/MLX production import, ordered layer dicts with fp16 K/V `(1,8,N,64)`, NPZ writer, and CLI log. RED observed: focused pytest exited 1 with 5 failures from missing module/API. GREEN observed: focused pytest exits 0: 5 passed; CLI smoke wrote `logs/c1-prefill-prompt0.log` and `logs/c1-prefill-prompt0.npz` with n_prefix 5, num_layers 16, layer0 deltas K max 0.00390625/V max 0.00024414062, layer15 deltas K max 0.0078125/V max 0.00390625, exit_status 0; combined `tests/native_r9700 -v` exits 0: 71 passed; full `tests -v` exits 0: 111 passed, 2 warnings; `git diff --check` clean; review approved 0 findings. Qwen remains deferred/unsupported. | |
| C1-8. KV interchange emitter | In progress | Main / C1EmitterScout | C1-7 | `.superpowers/swarm/reports/c1-task-8-kv-emitter.md` | Unblocked by C1-7 reviewed/verified prefill array contract. C1EmitterScout confirmed the mlx-lm prompt-cache ABI: safetensors tensor keys `{i}.0`/`{i}.1`, empty per-layer meta_state keys `0.{i}`, class keys `2.{i}=KVCache`, global `1.offset=str(N)`, `1.num_layers=16`, `1.n_kv_heads=8`, `1.head_dim=64`, and N must be S-1. | |
| C1-9. Parity harness and report writer | Blocked | TBD | C1-8 | `.superpowers/swarm/reports/c1-task-9-parity.md` | | Blocked by C1 KV emitter/native producer. |
| C1-10. C1 review and handoff | Blocked | TBD | C1-9 | `.superpowers/swarm/reports/c1-task-10-review-handoff.md` | | Blocked by C1 parity gate. |
| C2-1. C2 integration contract and validation discovery | Blocked | TBD | C1-10 | `.superpowers/swarm/reports/c2-task-1-contract.md` | README and C0 handoff record C2 dependency state. | Blocked by C1 token-exact native producer parity. |
| C2-2. mlx-lm imported-cache wrapper | Blocked | TBD | C2-1 | `.superpowers/swarm/reports/c2-task-2-mlx-wrapper.md` | | Blocked by C2-1/C1 parity. |
| C2-3. Fallback and error-state tests | Blocked | TBD | C2-1 | `.superpowers/swarm/reports/c2-task-3-fallback-tests.md` | | Blocked by C2-1/C1 parity. |
| C2-4. mlx-lm integration run and report append | Blocked | TBD | C2-2, C2-3 | `.superpowers/swarm/reports/c2-task-4-integration.md` | | Blocked by C2 wrapper/test work. |
| C2-5. oMLX imported-cache scope decision | Blocked | TBD | C2-1 | `.superpowers/swarm/reports/c2-task-5-omlx-scope.md` | | Blocked by C2-1/C1 parity. |
| C2-6. oMLX imported-cache seam optional | Blocked | TBD | C2-5 | `.superpowers/swarm/reports/c2-task-6-omlx-seam.md` | | Blocked by oMLX scope decision and C1 parity. |
| C2-7. C2 security review handoff | Blocked | TBD | C2-4, C2-5/C2-6 | `.superpowers/swarm/reports/c2-task-7-security-review.md` | | Blocked by C2 integration work. |
| C3-1. C2 evidence intake and backend justification | Blocked | TBD | C2-7 | `.superpowers/swarm/reports/c3-task-1-evidence.md` | README and C0 handoff record C3 dependency state. | Blocked by missing C2 serving/performance evidence. |
| C3-2. Backend seam decision | Blocked | TBD | C3-1 | `.superpowers/swarm/reports/c3-task-2-seam-decision.md` | | Blocked by C3-1 evidence intake. |
| C3-3. Boundary ADR design update | Blocked | TBD | C3-2 if boundary changes | `.superpowers/swarm/reports/c3-task-3-adr-design.md` | | Blocked by C3 backend seam decision. |
| C3-4. Narrow native backend prototype | Blocked | TBD | C3-2, C3-3 if required | `.superpowers/swarm/reports/c3-task-4-prototype.md` | | Blocked by C3 backend seam decision/design update. |
| C3-5. Prototype comparison and report | Blocked | TBD | C3-4 | `.superpowers/swarm/reports/c3-task-5-comparison.md` | | Blocked by C3 prototype. |
| C3-6. C3 final decision and handoff | Blocked | TBD | C3-5 or C3-2 if dropped/deferred | `.superpowers/swarm/reports/c3-task-6-final-handoff.md` | | Blocked by C3 comparison or backend decision. |

---

# Swarm Progress Ledger — gfx1201 Compute Dispatch Resolution

Phase docs: `docs/tasks/gx1202-compute-dispatch/`
Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer` (current feature branch; no new fallback worktree).
Supervisor artifact: `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`

## Orchestration map

- Sequential blockers: Phase 1 task set 1 RED tests -> supervisor RED pytest -> Phase 1 task set 2 GREEN implementation; Phase 2 task sets 1 -> 2 -> 3; Phase 3 task sets 1 -> 2 -> 3; Phase 4 task set 3 depends on task sets 1 and 2; Phase 5 closes pass/blocker state after review.
- Parallel lanes: none in Phases 1-3. Phase 4 task sets 1 and 2 may run in parallel after reviewed Phase 3 pass if agents coordinate on `am_compute` mappings. Phase 5 low-risk ledger drafting may prepare in parallel with final review, but final status changes serialize after review.
- Shared contracts/artifacts: focused pytest command in `tests/test_native_amdev_transfer_contract.py`; native proof source `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`; reports under `.superpowers/swarm/reports/`; TinyGPU.app/APLRemotePCIDevice/PCIIface acceptance path; no tinygrad runtime dependency.
- Coordination risks: high-risk GC and MEC/HQD register writes require review gates; direct PM4 only when native discovery proves one GC/XCC; Phase 4 kernel text provenance gap must be resolved before embedding/loading bytes; agents never run tests, package managers, git commands, or hardware commands.
- Verification gates: focused pytest after every wave; exact `--kernel-proof` hardware command after hardware waves; exact `--transfer-proof` command when transfer behavior changes or compute state is suspect; `git diff --check` at final.
- Commit boundary: supervisor checkpoint commits after reviewed/verified waves only; agents never commit; push remains user responsibility.

| Task | Status | Owner | Dependencies | Report | Evidence | Blocker |
|---|---|---|---|---|---|---|
| C0A Compute 1. RED compute contract tests | Done | C0AComputeContracts / Main | C0A-5 blocker state | `.superpowers/swarm/reports/c0a-compute-task-1-contracts.md` | Supervisor focused pytest exited `1` with expected RED: 5 failed, 11 passed; four new self-tests return unknown self-test and help lacks new names. Test function names were aligned to the source plan. | |
| C0A Compute 2. Compute constants and self-tests | Done | C0AComputeSelfTests / Main / C0AComputePhase1Review | C0A Compute 1 | `.superpowers/swarm/reports/c0a-compute-task-2-vm-layout.md` | Supervisor focused pytest exited `0`: `16 passed in 14.89s`. Phase 1 reviewer found one Important resume-artifact issue and one Minor stale-status issue; supervisor added all artifacts to the patch and fixed the supervisor row; re-review found no Critical/Important/Minor findings and `ready_for_phase2: true`. | |
| C0A Compute 3. GC contract topology validation | Done | C0AComputeGCContracts / Main | C0A Compute 2 | `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md` | Supervisor focused pytest exited `1` with expected RED: 2 failed, 15 passed; `gc-hub-sequence` returns unknown self-test and help lacks the new entry. Source-only `validate_direct_pm4_topology` and `gc_instance_count` landed with no GC register writes/calls. | |
| C0A Compute 4. GC register programming | Done | C0AComputeGCProgramming / Main | C0A Compute 3 | `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md` | Supervisor focused pytest exited `0`: `17 passed in 15.59s`. Exact GC regs, source-only GC VMID0/TLB helpers, and `gc-hub-sequence` self-test/help/dispatch landed; hardware integration deferred. | |
| C0A Compute 5. GC hardware integration review gate | Done | C0AComputeGCHardware / Main / C0AComputeGCReview / C0AComputeGCFix / C0AComputeGCReReview | C0A Compute 4 | `.superpowers/swarm/reports/c0a-compute-task-3-gc-preflight.md` | Initial reviewer found one Important stage-mapping issue and two Minor resume/contract issues; fix applied. Post-fix supervisor focused pytest exited `0`: `17 passed in 16.62s`; `git diff --check HEAD` produced no output. Hardware `--kernel-proof` wrote `logs/c0-macos-egpu-minimal-runtime.log` at `2026-08-17T16:11:18Z`, exited `1`, and reached `vm_gc_context_status: pass`, `gc_tlb_flush_status: pass`, `failure_stage: compute_ring_setup`. Transfer preservation command wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T16:11:27Z` and passed with `exit_status: 0`, `wrapper_exit_status: 0`. Re-review found no Critical/Important findings and `ready_for_phase3: true`; Minor report tracking note will be closed by checkpoint commit. | |
| C0A Compute 6. Compute log contract | Done | C0AComputeLogContract / Main | C0A Compute 5 reviewed pass or accepted blocker | `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md` | Supervisor focused pytest exited `0`: `17 passed in 16.69s`; `git diff --check HEAD` produced no output. Added compute ring/control log lines and no-hardware contract only; no MQD/HQD/register/doorbell/hardware behavior added. | |
| C0A Compute 7. MQD builder queue reset | Done | C0AComputeMqdReset / Main | C0A Compute 6 | `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md` | Supervisor tightened MQD builder to fill HQD-copy dwords required by plan lines 537-543. Focused pytest exited `0`: `17 passed in 16.48s`; `git diff --check HEAD` produced no output. New helpers are not called from runtime paths; no hardware behavior changed. | |
| C0A Compute 8. HQD ring hardware gate | Done | C0AComputeHqdHardware / Main / C0AComputeHqdReview / C0AComputeHqdFix / C0AComputeHqdReReview | C0A Compute 7 | `.superpowers/swarm/reports/c0a-compute-task-4-ring-setup.md`; `.superpowers/swarm/reports/c0a-compute-task-4-hqd-review.md`; `.superpowers/swarm/reports/c0a-compute-task-4-hqd-fix.md`; `.superpowers/swarm/reports/c0a-compute-task-4-hqd-rereview.md` | Initial review found 2 Important findings; fix corrected MQD/HQD copy dword indices and added direct-PM4 dword-unit handoff. Post-fix focused pytest exited `0`: `17 passed in 17.84s`; hardware `--kernel-proof` wrote `logs/c0-macos-egpu-minimal-runtime.log` at `2026-08-17T17:02:27Z`, exited `1` as expected, and reached `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, `kernel_launch_status: blocked`, `failure_stage: kernel_blob_load`; transfer preservation wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T17:02:36Z`, exited `0`, and reached `failure_stage: none`; `git diff --check HEAD` produced no output. Re-review found 0 Critical/Important/Minor findings and marked Phase 4 ready after checkpoint commit. | |
| C0A Compute 9. Single-copy SDMA primitive | Done | C0AComputeSdmaPrimitive / Main / C0AComputePrereqReview | C0A Compute 8 reviewed pass or accepted blocker | `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`; `.superpowers/swarm/reports/c0a-compute-task-5-prereq-review.md` | Added reusable `submit_sdma_copy`, preserved `--transfer-proof`, and supervisor wired `--kernel-proof` pre-dispatch H2D through the primitive. Focused pytest exited `0`: `17 passed in 18.58s`; SDMA hardware transfer wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T17:26:23Z`, exited `0`, and reached `host_device_transfer_status: pass`. Prereq review found 0 Critical/Important/Minor issues and allowed Task 3. | |
| C0A Compute 10. Kernel provenance mappings | Done | C0AComputeKernelProvenance / Main / C0AComputePrereqReview | C0A Compute 8 reviewed pass or accepted blocker | `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`; `.superpowers/swarm/reports/c0a-compute-task-5-prereq-review.md` | Embedded source-grounded 512-byte kernel text, added BAR0 code readback, kernarg/control mapping, SDMA H2D status fields, and current blocker handoff. Focused pytest exited `0`: `17 passed in 18.58s`; hardware `--kernel-proof` wrote `logs/c0b-native-amdev-kernel-ref.log` at `2026-08-17T17:26:32Z`, exited `1` as expected, and reached `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `sdma_h2d_status: pass`, `compute_ring_setup_status: pass`, `compute_hqd_active_status: pass`, `failure_stage: kernel_dispatch_submit`. Prereq review found 0 Critical/Important/Minor issues and allowed Task 3. | |
| C0A Compute 11. PM4 dispatch readback | Done | Main / C0AComputePm4Review / C0AComputePm4ReReview | C0A Compute 9, C0A Compute 10 | `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`; `.superpowers/swarm/reports/c0a-compute-task-5-pm4-review.md`; `.superpowers/swarm/reports/c0a-compute-task-5-pm4-rereview.md` | Implemented source-grounded direct-PM4 packet build/submission and timeout diagnostics. Supervisor focused PM4 pytest passed, then full no-hardware pytest exited `0`: `17 passed in 19.87s`. Hardware `--kernel-proof` wrote `logs/c0c-native-amdev-kernel-dispatch.log` at `2026-08-17T17:53:08Z`, exited `1`, and produced accepted blocker evidence: emitted `failure_stage: kernel_timeline_timeout`; diagnostic `rptr=0`, `wptr=59`, `hqd_active=1`, `hqd_pq_doorbell_control=0x40000018`, `hqd_pq_rptr=0`, `cp_stat=0`; inferred `compute_doorbell_not_consumed`. PM4 re-review found 0 Critical/Important/Minor findings and marked ready for split decision. | Accepted blocker: MEC doorbell/ring fetch not observed. |
| C0A Compute 12. Repeated-run review packet | Done | Main / C0AComputePm4Review / C0AComputePm4ReReview | C0A Compute 11 | `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`; `.superpowers/swarm/reports/c0a-compute-task-5-pm4-review.md`; `.superpowers/swarm/reports/c0a-compute-task-5-pm4-rereview.md` | Repeated-run pass is intentionally not run because Task 11 produced an accepted native blocker before D2H/CPU compare. Review packet and re-review close Phase 4 with precise blocker evidence instead of pass tokens. | |
| C0A Compute 13. Blocker split decision | Done | Main / C0AComputeFinalReview / C0AComputeFinalReReview | C0A Compute 12 if native path remains blocked | `.superpowers/swarm/reports/c0a-compute-split-decision.md`; `.superpowers/swarm/reports/c0a-compute-final-review.md` | Split decision report written and final re-review accepted after fixes. Recommendation: continue native macOS GFX port with one named MEC doorbell delivery/ring-fetch investigation; C0 scope does not change, so no user approval is required. | |
| C0A Compute 14. Final reviewer fix loop | Done | C0AComputeFinalReview / Main / C0AComputeFinalReReview | C0A Compute 12 and C0A Compute 13 if needed | `.superpowers/swarm/reports/c0a-compute-final-review.md` | Initial final review found 3 Important and 1 Minor docs/ledger/report consistency findings. Supervisor fixed all four; final re-review found 0 Critical/Important/Minor findings and `ready_for_ledger_checkpoint: true`. Supervisor final focused pytest exited `0`: `17 passed in 20.21s`. | |
| C0A Compute 15. Ledgers checkpoint prep | Done | Main | C0A Compute 14 | `.superpowers/swarm/progress.md`, `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`, `.superpowers/swarm/native-r9700-producer-supervisor.md` | Ledgers, validation docs, supervisor artifacts, split decision, and final review report updated for accepted blocker state. Supervisor final focused pytest exited `0`: `17 passed in 20.21s`; `git diff --check` produced no output. | |
| C0A Compute 16. MEC doorbell delivery / ring-fetch primitive | Done | Main / DoorbellReview | C0A Compute 15 | `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`; `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md` | Reviewed diagnostic complete: final no-hardware pytest passed `18 passed in 21.31s`; hardware command wrote `logs/c0d-native-amdev-doorbell-delivery.log` at `2026-08-17T19:06:34Z`, emitted all five `compute_doorbell_probe_*` fields, `failure_stage: kernel_timeline_timeout`, `exit_status: 1`, `wrapper_exit_status: 1`, and `compute_doorbell_probe_classification: compute_doorbell_not_consumed`; review found 0 Critical/Important and `ready_for_checkpoint: true`; `git diff --check` printed no output. | Reviewed blocker: `compute_doorbell_not_consumed`; next boundary is source-grounding BAR2 doorbell index/value, MEC doorbell ranges, and GDC S2A routing before changing a register. C0A/C1/C2/C3 remain blocked. |
| C0A Compute 17. MEC doorbell source grounding | Done | Main / DoorbellBar2Audit / DoorbellRangeAudit / DoorbellRoutingAudit / DoorbellSourceDecision / DoorbellSourceReview | C0A Compute 16 | `.superpowers/swarm/reports/c0a-compute-task-7-bar2-doorbell-index.md`; `.superpowers/swarm/reports/c0a-compute-task-7-mec-doorbell-range.md`; `.superpowers/swarm/reports/c0a-compute-task-7-gdc-s2a-routing.md`; `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding.md`; `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding-review.md` | Phase 4 source grounding complete: BAR2 index/value `gap`, CP MEC range `matches`, GDC/S2A routing `gap`; consolidated report selected `blocked_source_gap`; review found 0 Critical/Important/Minor and `ready_for_next_plan: true` for a source-gap resolution/blocker plan only. | Runtime-path implementation remains blocked until the gfx1201/TinyGPU doorbell assignment-family selector and GDC/S2A route readback/field-semantics gaps resolve to either cited contradiction or all audited contracts matching; C0A/C1/C2/C3 remain blocked unless CPU pass tokens exist or user-approved fallback/split changes the path. |
| C0A Compute 18. Doorbell source-gap resolution | Done | Main / DoorbellBar2AssignmentAudit / DoorbellGdcS2aCoverageAudit / DoorbellRouteContract / DoorbellRouteReadback / DoorbellRouteReview / DoorbellSourceGapReview | C0A Compute 17 | `docs/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md`; `docs/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md`; `.superpowers/swarm/reports/c0a-compute-task-8-bar2-assignment-selector.md`; `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-coverage.md`; `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-contract.md`; `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-instrumentation.md`; `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-instrumentation-review.md`; `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-readback.md`; `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-decision.md`; `.superpowers/swarm/reports/c0a-compute-task-8-doorbell-source-gap-review.md` | Source-gap execution complete as a reviewed blocker: BAR2 assignment-family selector `matches` and closes the queue0/index `3` gap; CP MEC range remains `matches`; GDC/S2A raw programming and hardware readback match (`gdc_s2a_route_readback_matches`) but coverage semantics remain `gap`; decision selected `blocked_source_gap`; final review found 0 Critical/Important/Minor with `ready_for_checkpoint: true`, `ready_for_next_plan: false`, and implementation dispatch disallowed. Verification: RED focused pytest failed for missing route-readback contract lines before implementation; focused pytest after instrumentation passed `18 passed in 21.75s`; hardware log `logs/c0d-native-amdev-doorbell-source-gap.log` exited with `wrapper_exit_status: 1` as expected and preserved `compute_doorbell_not_consumed`; final focused pytest passed `18 passed in 21.78s`; final `git diff --check` printed no output. | Runtime-path implementation remains blocked on exact GDC/S2A `range_offset=0`/`range_size=0` coverage semantics for BAR2 byte offset `0x18`; no BAR2 index/value, CP MEC range, GDC/S2A route, PM4 packet, scheduler, retry loop, AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work is authorized from this `gap`. |
| C0A Compute 19. Doorbell blocker resolution | Done | Main / DoorbellSourceGapExitReview / DoorbellConsumptionContract / DoorbellConsumptionInstrumentation / DoorbellConsumptionDecisionReview / DoorbellMqdHqdCopyFix / DoorbellTask10Review / DoorbellCpMecVisibility / DoorbellCpMecReview | C0A Compute 18 | `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md`; `docs/tasks/amdev-doorbell-delivery/phase-6-doorbell-blocker-resolution.md`; `docs/tasks/amdev-doorbell-delivery/phase-7-mqd-hqd-copy-fix.md`; `docs/tasks/amdev-doorbell-delivery/phase-8-cp-mec-visibility-diagnostic.md`; `.superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit.md`; `.superpowers/swarm/reports/c0a-compute-task-9-source-gap-exit-review.md`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-contract.md`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation.md`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation-review.md`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-instrumentation-rereview.md`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-hardware.md`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision.md`; `.superpowers/swarm/reports/c0a-compute-task-9-consumption-decision-review.md`; `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-contract.md`; `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-fix.md`; `.superpowers/swarm/reports/c0a-compute-task-10-mqd-hqd-copy-proof.md`; `.superpowers/swarm/reports/c0a-compute-task-10-review.md`; `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility-instrumentation.md`; `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility.md`; `.superpowers/swarm/reports/c0a-compute-task-10-cp-mec-visibility-review.md` | Reviewed blocker checkpoint complete. Phase 6 source-gap exit, RED contract, GREEN instrumentation, hardware diagnostic, decision review, Phase 7 `mqd_hqd_copy_fix`, and Phase 8 CP/MEC visibility diagnostic are complete. MQD/HQD mismatch was resolved (`mqd_hqd_mismatch_count=0`), but CPU pass tokens remain absent. Latest hardware log `logs/c0g-native-amdev-cp-mec-visibility.log` exited `1` with `cp_mec_rs64_interrupt=0x0000000a`, `cp_mec_rs64_pending_interrupt=0x00000400`, `cp_mec_rs64_exception_status=0x0000c67a`, and `compute_doorbell_consumption_classification: doorbell_not_reaching_hqd_unclassified`; CP/MEC review found 0 Critical/Important/Minor and accepted `cp_mec_rs64_exception_status_needs_source_grounding`. Final focused pytest passed `19 passed in 25.28s`; final `git diff --check` printed no output. | C0A/C1/C2/C3 remain blocked until CP/MEC RS64 exception status is source-grounded and a one-field fix is reviewed, or the user explicitly approves a fallback/split path. |
| C0A Compute 20. CP/MEC RS64 source grounding | Done | Main / DoorbellRs64SourceReview / DoorbellRs64Context / DoorbellRs64ContextReviewPreHardware / DoorbellRs64ContextDecisionRereview / DoorbellRs64FinalRereview | C0A Compute 19 | `docs/superpowers/plans/2026-08-17-cp-mec-rs64-exception-grounding.md`; `docs/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-source-grounding.md`; `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding.md`; `.superpowers/swarm/reports/c0a-compute-task-11-rs64-source-grounding-review.md`; `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-contract.md`; `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-instrumentation.md`; `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-instrumentation-review.md`; `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context.md`; `.superpowers/swarm/reports/c0a-compute-task-11-rs64-context-review.md`; `.superpowers/swarm/reports/c0a-compute-task-11-rs64-classifier-fix.md`; `.superpowers/swarm/reports/c0a-compute-task-11-rs64-final-review.md` | Source-grounded RS64 exception bits; added diagnostic-only RS64 context readbacks and classifier; focused classifier regression passed `1 passed in 1.51s`; full focused pytest passed `20 passed in 27.51s`; hardware rerun `logs/c0h-native-amdev-rs64-context.log` exited `1` with `compute_doorbell_consumption_classification: rs64_exception_context_needed`; context re-review and final review found 0 Critical/Important/Minor. | Reviewed blocker `cp_mec_rs64_context_still_multicausal_needs_source_mapping`; C0A/C1/C2/C3 remain blocked until CPU pass tokens or source mapping identifies one host-controlled RS64 field. |

| C0A Compute 21. Sysmem ring backing isolation | Done | Main | C0A Compute 20 | `docs/superpowers/plans/2026-08-17-sysmem-ring-backing-isolation.md`; `logs/c0j-native-amdev-unord-dispatch-0.log` | Plan written and reviewed; unord_dispatch=0 kept; ring-backing change surface mapped (sysmem ring pages 2..9, ring PTE remap, ring-word sysmem write); reviewed blocker cp_mec_rs64_instr_state_needs_firmware_config; ring backing eliminated; unord carried in 30d573b | C0A/C1/C2/C3 remain blocked until `--kernel-proof` passes or records a reviewed next blocker |
| C0A Compute 22. MEC RS64 pipe activation replay | Done | Main | C0A Compute 21 | `docs/superpowers/plans/2026-08-17-mec-rs64-pipe-activation.md`; `logs/c0l-native-amdev-mec-rs64-pipe-activation.log`; `.superpowers/swarm/reports/c0a-compute-task-13-mec-rs64-pipe-activation.md` (+ -task1, -review, -hardware-review) | Launch blocker ELIMINATED: kernel_launch_status=pass (was kernel_timeline_timeout), mec_rs64_cntl readback 0x04000000 active, doorbell_hit=1; new failure_stage=readback_mismatch with halfword byte-swap + partial write (only outputs 2,3,4,5 of 8); classification changed_signature_launch_eliminated_readback_byte_swap; kernel_proof_pass=false; behavior_fix_authorized=true (pipe-activation retained); next_blocker=compute_output_readback_byte_swap; C0A21 reviewed blocker 9862430 discarded | C0A/C1/C2/C3 remain blocked until `--kernel-proof` CPU pass-token; next blocker readback byte-swap |
| C0A Compute 23. Compute output readback byte-swap diagnostic | Done | Main | C0A Compute 22 | `docs/superpowers/plans/2026-08-18-compute-output-readback-byte-swap.md` | Reviewed accepted (Wave 1 + Wave 2, 0 findings). Instrument-only classifier + RDNA4 kernel decode localized the anomaly to the GPU store side: single `VGLOBALOp.GLOBAL_STORE_B128` (op 29), base+offset, `vsrc=v[0:3]`, grid=2 work-items; hardware `c0m` UNCHANGED-SIGNATURE vs `c0l` (byte-identical `observed_hex`), classifier `swap_and_partial 0x0f/0x0f/0x0f`. Root cause = kernel-store format (D16-swizzled packed-128 lane + dropped ADDTID addressing) → C0A24 single fix lane: kernel-text rewrite to per-u32 B32/ADDTID store, reviewed separately. pytest 23 passed, build OK, `git diff --check` clean. Reports: `.superpowers/swarm/reports/c0a-compute-task-14{,a,b,-wave1,-wave2}-review.md` | C0A/C1/C2/C3 remain blocked until `--kernel-proof` CPU `pass` / `failure_stage: none` / exit 0; next blocker is the C0A24 kernel-store rewrite |
| C0A Compute 24. Kernel store byte-swap + partial-write fix | Done | Main / T1Implementer / T2Implementer / T1Reviewer / T2Reviewer / WholeReviewer | C0A Compute 23 | `local://c0a24-kernel-store-fix-plan.md`; `logs/c0o-native-amdev-kernel-store-fix.log`; `.superpowers/swarm/reports/c0a-compute-task-15-kernel-store-fix.md`; `.superpowers/sdd/c0a24-kernel-store-fix/task-1-report.md` (+task-2, +review packages) | Reviewed+approved (T1, T2, whole-branch; 0 Critical/Important). Replaced 512B B128 kernel with source-grounded 64B per-u32 `GLOBAL_STORE_B32` lane kernel (sha256 081ad254…, source id c0a-minimal-u32-add-one-v2) and dispatched 1 workgroup × 8 lanes (`global_size_x=1, local_size_x=8`, commit `11099e5`+`d86acb5`). Hardware `c0o`: **kernel_launch_status=pass**, **byte-swap ELIMINATED** (`swapped_mask=0x00`), **partial-write ELIMINATED** (`written_mask=0xff`, all 8 written), but values read back uniform `0x00000001` (`unswapped_match_mask=0x00`, `observed_hex=01000000`×8) → classifier `other_mismatch`; classification **changed_signature_store_format_and_coverage_fixed_value_load_path_remaining**; kernel_proof_pass=false; CPU contract not relaxed; pytest 23 passed, `git diff --check` clean; next_blocker=compute_output_value_load_path (C0A25) | C0A/C1/C2/C3 remain blocked until `--kernel-proof` passes or records a reviewed next blocker (C0A25 load-path lane) |
| C0A Compute 25. Load-path value-lane fix | Done | Main / C0A25Implementer / C0A25Reviewer | C0A Compute 24 | `docs/tasks/native-r9700-producer/phase-c0a25-load-path-fix.md`; `logs/c0p-native-amdev-kernel-load-fix.log`; `.superpowers/swarm/reports/c0a-compute-task-16-load-path-fix.md`; `.superpowers/sdd/c0a25-load-path-fix/task-fused-wave1-report.md` (+review-wave1) | Reviewed+approved (C0A25Reviewer, 0 findings; fused plan Tasks 1+2, commit `45d7b95`). Root cause: `global_load_b32` used misaligned SGPR base s[5:6] instead of input-VA pair s[6:7] (tinygrad `custom_add_var`), so the per-lane load read 0 → uniform `0+1=1`. Fixed to s[6:7] (single byte 0x1c 0x05→0x06; store bytes byte-identical; sha256 08fd705c…; source id -v3). Hardware `c0p`: **kernel_launch_status=pass, cpu_comparison_status=pass, host_device_transfer_status=pass, failure_stage=none, failure_text=none, exit_status=0**, kernel_elapsed_usec=1506, mec_rs64 readback 0x04000000, doorbell_hit=1, `compute_readback_anomaly: not_run`, readback matches 02000000…09000000 (out[i]=in[i]+1=2..9). **KERNEL_PROOF_PASS=true — minimal macOS kernel proof PASS.** Verification: 3 self-tests pass, focused pytest 23 passed, `git diff --check` clean. C0 readback blocker cleared; macOS TinyGPU/AMDev native substrate SELECTED for C1 (C0 task set 5 rerun).

## MEC doorbell delivery execution map

- Source docs read: `docs/tasks/amdev-doorbell-delivery/phase-1-no-hardware-contract.md`, `phase-2-diagnostic-proof.md`, `phase-3-review-ledger-checkpoint.md`, and `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md`.
- Ledger path: `.superpowers/swarm/progress.md`.
- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer` (current feature branch; no new fallback worktree).
- Sequential blockers: Phase 1 Task set 1 RED contract -> supervisor RED pytest -> Phase 1 Task set 2 self-test implementation -> supervisor GREEN/help pytest -> Phase 2 instrumentation -> supervisor full no-hardware pytest -> supervisor hardware diagnostic -> report/docs update -> Phase 3 review -> fix/re-review if needed -> ledger updates -> final verification -> local checkpoint commit.
- Parallel lanes: none in this task-doc set. Every phase touches shared contract/source/docs or consumes the prior diagnostic output.
- Shared contracts/artifacts: `--self-test compute-doorbell-delivery`, `EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES`, `run_compute_doorbell_delivery_self_test()`, `compute_doorbell_probe_status`, `compute_doorbell_probe_pre`, `compute_doorbell_probe_post`, `compute_doorbell_probe_timeout`, `compute_doorbell_probe_classification`, `logs/c0d-native-amdev-doorbell-delivery.log`, `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`, `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-review.md`.
- Coordination risks: diagnostic-only work; no register/PM4 fix, retry loop, scheduler, AQL fallback, Linux HIP fallback, allocator/framework, or C1/C2/C3 execution until reviewed hardware evidence selects a lane or CPU pass tokens exist.
- Verification gates: focused RED pytest for the new self-test, focused GREEN pytest and help pytest, full `tests/test_native_amdev_transfer_contract.py -v`, exact hardware `--kernel-proof` command writing `logs/c0d-native-amdev-doorbell-delivery.log`, final full pytest, and `git diff --check`.
- Commit boundary: supervisor commits only after review and verification; agents never run git; push remains user responsibility.

## MEC doorbell source-grounding execution map

- Source docs read: `docs/tasks/amdev-doorbell-delivery/phase-4-doorbell-source-grounding.md`, `docs/tasks/amdev-doorbell-delivery/phase-3-review-ledger-checkpoint.md`, `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md`, and `docs/superpowers/plans/2026-08-17-mec-doorbell-delivery.md`.
- Ledger path: `.superpowers/swarm/progress.md`.
- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer` (current feature branch; no new fallback worktree).
- Sequential blockers: Phase 3 completion -> Phase 4 Task sets 1-3 source audits -> Task set 4 consolidated boundary decision -> Task set 5 review -> final ledger/supervisor updates -> `git diff --check` -> local checkpoint commit.
- Parallel lanes: Phase 4 Task sets 1, 2, and 3 run concurrently as read-only source audits and write separate reports.
- Shared contracts/artifacts: `compute_doorbell_not_consumed`, BAR2 doorbell index `3`, BAR2 byte offset `0x18`, doorbell value unit `dwords`, PM4 dispatch dword count `59`, CP MEC range snapshot `0x00000000..0x000000f8`, GDC/S2A ports `0` and `3`, reports under `.superpowers/swarm/reports/`.
- Coordination risks: no audit may claim a register mismatch without cited source/log evidence; no source edits, hardware command, PM4/register fix, scheduler, retry loop, fallback substrate, or C1/C2/C3 work in this phase; agents never run tests, package managers, hardware commands, or git.
- Verification gates: supervisor validates report fields by reading reports; final phase verification is `git diff --check`.
- Commit boundary: supervisor commits after reviewed report state and final verification only; agents never commit; push remains user responsibility.

## Doorbell source-gap resolution execution map

- Source docs read: `docs/superpowers/plans/2026-08-17-doorbell-source-gap-resolution.md`, `docs/tasks/amdev-doorbell-delivery/phase-4-doorbell-source-grounding.md`, `.superpowers/swarm/progress.md`, and `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`.
- Ledger path: `.superpowers/swarm/progress.md`.
- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer` (current feature branch; no new fallback worktree).
- Sequential blockers: Phase 5 task doc creation -> Wave 1 source-only BAR2 assignment and GDC/S2A coverage audits -> supervisor report review -> conditional TDD/readback instrumentation -> supervisor focused pytest -> supervisor hardware readback proof -> readback report -> consolidated decision -> reviewer/fix loop -> final verification -> local checkpoint commit.
- Parallel lanes: Wave 1 runs BAR2 assignment-family selector audit and GDC/S2A source-semantics audit concurrently; later instrumentation/hardware/decision work is sequential because each step consumes the prior report/log.
- Shared contracts/artifacts: BAR2 doorbell index `3`, BAR2 byte offset `0x18`, CP/HQD doorbell-control dword offset `6`, CP MEC range status `matches`, GDC/S2A route raw values `0x30000007` and `0x3000000d`, route readback fields `compute_doorbell_route_readback` and `compute_doorbell_route_classification`, hardware log `logs/c0d-native-amdev-doorbell-source-gap.log`, reports under `.superpowers/swarm/reports/`.
- Coordination risks: no source-only audit may select a fix from a gap; no executor may run tests, linters, formatters, package managers, git commands, project-wide suites, or hardware commands; no runtime fix, fallback, scheduler, retry loop, allocator/runtime framework, or C1/C2/C3 work is allowed in this source-gap phase.
- Verification gates: supervisor validates Wave 1 by reading reports; if instrumentation lands, supervisor runs `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v`; hardware proof uses `logs/c0d-native-amdev-doorbell-source-gap.log`; final verification is fresh `git diff --check` plus any instrumentation pytest/hardware evidence required by the selected path.
- Commit boundary: supervisor makes local checkpoint commits only after reviewed/verified waves; agents never commit or push; push remains user responsibility.

## Doorbell blocker resolution execution map

- Source docs read: `docs/superpowers/plans/2026-08-17-c0-doorbell-blocker-resolution.md`, `docs/tasks/amdev-doorbell-delivery/phase-5-doorbell-source-gap-resolution.md`, `.superpowers/swarm/progress.md`, `.superpowers/swarm/gx1202-compute-dispatch-supervisor.md`, and Task 8 reports/logs.
- Ledger path: `.superpowers/swarm/progress.md`.
- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer` (current feature branch; no new fallback worktree).
- Sequential blockers: Phase 6 task doc and source-gap exit report -> source-gap exit review -> RED consumption contract -> supervisor RED pytest -> diagnostic-only HQD/PQ instrumentation -> supervisor GREEN pytest -> instrumentation review -> hardware consumption diagnostic -> hardware report -> consumption decision -> decision review -> selected narrow lane or reviewed blocker -> final verification -> local checkpoint commit.
- Parallel lanes: none before selected-lane execution; every phase consumes the previous contract/report/log. Conditional fix lanes are mutually exclusive.
- Shared contracts/artifacts: `source_gap_exit_status: diagnostic_override_allowed`, `--self-test compute-doorbell-consumption`, `EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_LINES`, `ComputeDoorbellConsumptionSnapshot`, `format_compute_doorbell_consumption_snapshot(...)`, `classify_compute_doorbell_consumption_timeout(...)`, `compute_doorbell_consumption_timeout`, `compute_doorbell_consumption_classification`, `logs/c0e-native-amdev-doorbell-consumption.log`, and `.superpowers/swarm/reports/c0a-compute-task-9-*`.
- Coordination risks: no BAR2 index/value, CP MEC range, GDC/S2A route, PM4 packet, scheduler, retry loop, AQL, Linux HIP fallback, allocator/runtime framework, or C1/C2/C3 work is authorized from the remaining GDC/S2A coverage semantic `gap`. Agents do not run tests, linters, formatters, package managers, hardware commands, or git commands.
- Verification gates: source-gap exit review; RED focused pytest for `compute-doorbell-consumption`; full focused pytest after instrumentation; instrumentation review; exact hardware command writing `logs/c0e-native-amdev-doorbell-consumption.log`; decision review; final focused pytest and `git diff --check` before checkpoint.
- Commit boundary: supervisor creates local checkpoint commits only after reviewed/verified waves; agents never commit or push; push remains user responsibility.
```

## File: `.superpowers/swarm/native-r9700-producer-supervisor.md`

```text
# Swarm Supervisor Plan: Native R9700 Producer

## Source and resume state
- Source docs read: `docs/tasks/native-r9700-producer/README.md`, phase C0-C3 docs, `validation-commands.md`, existing `.superpowers/swarm/progress.md`.
- Ledger path: `.superpowers/swarm/progress.md`.
- Rows preserved: prior Path A Phase 0 rows remain unchanged and are treated as completed baseline evidence.
- Baseline command on fallback worktree: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v` -> `17 passed, 2 warnings`.

## Orchestration map
- Sequential blockers:
  - C0 task set 1 blocks C0 proof lanes.
  - C0 task set 5 blocks C0 handoff and all C1 work.
  - C1 task set 1 blocks C1 code work.
  - C1 task set 2 blocks model-loader/kernel slices.
  - C1 task set 9 blocks C1 acceptance and all C2 work.
  - C2 task set 1 blocks C2 wrapper/test work.
  - C2 task set 2 blocks C2 integration run.
  - C2 task set 5 blocks optional C2 task set 6.
  - C3 task sets 1-2 block any C3 prototype; task set 3 is required only if the backend decision changes the KV interchange boundary.
- Parallel waves:
  - C0 Wave 1: task set 1 only.
  - C0 Wave 2: task sets 2, 3, and 4 in parallel after task set 1.
  - C0 Wave 3: task set 5, then task set 6.
  - C1 Wave 1: task set 1 only.
  - C1 Wave 2: task sets 2, 3, and 4 in parallel after task set 1.
  - C1 kernel waves: task sets 5 and 6 only after a shared tensor/layout contract is frozen; task set 7 after primitives/KV writer; task set 8 after full cache candidate; task set 9 final; task set 10 review.
  - C2 Wave 1: task set 1 only.
  - C2 Wave 2: task sets 2, 3, and 5 in parallel after task set 1; task set 4 after wrapper; task set 6 only if task set 5 selects ship; task set 7 review.
  - C3 Wave 1: task set 1 after C2 evidence; task set 2 after evidence; later tasks only if justified.
- Shared contracts/artifacts:
  - Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer` (fallback linked worktree because source checkout was on `main`).
  - Task docs and uncommitted design/source-grounding docs from the main checkout were copied into the fallback worktree before execution.
  - C0 experimental source root default: `experiments/native-r9700-runtime/` unless task set 1 finds a stronger repo convention.
  - Validation command ledger: `docs/tasks/native-r9700-producer/validation-commands.md`.
  - Run logs: local `logs/` or documented remote artifact path; logs/model files are not committed.
  - Review reports: `.superpowers/swarm/reports/`.
- Coordination risks:
  - Proof lanes may touch the same C0 phase doc and validation ledger; agents must only update their assigned row/report and coordinate by `hub` if overlapping.
  - Only C0 task set 5 decides the runtime substrate; proof agents record evidence only.
  - DwarfStar remains reference-only; no vendoring, architecture adoption, or dependency.
  - C1/C2/C3 are blocked until preceding gates pass; do not let agents skip phase gates.
- Verification gates:
  - Baseline: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v`.
  - C0 docs: `git diff --check docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md` plus exact C0 probe commands discovered by task set 1.
  - C1/C2/C3: exact commands must be recorded in `validation-commands.md` by their contract-discovery task sets before execution.
- Publish boundary:
  - Supervisor makes local checkpoint commits after reviewed/verified waves. Agents never run git. Push remains the user's responsibility.

## User steering: macOS initial runtime focus

- Steering received after C0 handoff: work initially on the mac eGPU runtime.
- Decision: keep C0 final state `blocked` because no substrate passed, but make the next execution plan macOS-first.
- New task doc: `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`.
- Linux ROCm/HIP remains a reference fallback, not the initial work lane.
- C1 remains blocked until the macOS proof, or an explicitly reactivated fallback proof, produces CPU-verified minimal kernel launch/transfer/readback evidence and C0 task set 5 reruns to a selected substrate.

## Wave 1: C0 validation and source-layout discovery
### Shared context
# Goal
Freeze C0 proof-lane command shape and source root so macOS, Linux, and DwarfStar lanes produce comparable evidence.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every agent stays in this cwd/branch.
- Do not touch C1/C2/C3 implementation or docs except `validation-commands.md` if command dependencies must be named.
- OMP task executors do not run tests, linters, formatters, package managers, git commands, or project-wide suites; supervisor runs verification after the wave.
- Report path: `.superpowers/swarm/reports/c0-task-1-validation-layout.md`.

# Contract
- Default experimental source root is `experiments/native-r9700-runtime/` unless repo convention proves a better path.
- C0 command entries must be exact commands or explicit blockers tied to task set owner and missing prerequisite.
- Proof code must be tinygrad-free on execution paths.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0ValidationLayout | C0 task set 1 | `validation-commands.md`, C0 phase ledger | none | `.superpowers/swarm/reports/c0-task-1-validation-layout.md` | Done |

### Supervisor gates
- Report checks: source root chosen once; C0 macOS/Linux/doc commands recorded or blocked with exact prerequisite; no probe implementation done.
- Quality bar result: passed; supervisor docs check completed.
- Review agents: after wave if docs/commands are ambiguous or over-broad.
- Verification command(s) supervisor will run: `git diff --check docs/tasks/native-r9700-producer/validation-commands.md docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md`.
- Ledger update: mark C0 task set 1 `Needs review`, then `Done` after review/verification.

## Wave 2: C0 proof lanes
### Shared context
# Goal
Produce C0 macOS, Linux HIP, and DwarfStar reference evidence after C0 task set 1 froze the source root and command ledger.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Frozen C0 experimental root: `experiments/native-r9700-runtime/`.
- Proof execution paths must be tinygrad-free. DwarfStar remains reference-only.
- OMP task executors do not run tests, linters, formatters, package managers, git commands, or project-wide suites; supervisor runs verification/proof commands after the wave.
- Reports: `.superpowers/swarm/reports/c0-task-2-macos-egpu.md`, `.superpowers/swarm/reports/c0-task-3-linux-hip.md`, `.superpowers/swarm/reports/c0-task-4-dwarfstar.md`.

# Contract
- macOS lane owns `macos_tinygpu_minimal.cpp` and C0-2 evidence only.
- Linux lane owns `linux_hip_minimal.cpp` and C0-3 evidence only.
- DwarfStar lane owns reference notes and C0-4 evidence only.
- Only C0-5 chooses the runtime substrate after supervisor verifies this wave.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0MacOSEGPU | C0-2 | macOS tinygrad-free minimal probe | C0-1 | `.superpowers/swarm/reports/c0-task-2-macos-egpu.md` | Blocked |
| C0LinuxHIP | C0-3 | Linux HIP reference probe/blocker | C0-1 | `.superpowers/swarm/reports/c0-task-3-linux-hip.md` | Blocked |
| C0DwarfStar | C0-4 | DwarfStar reference extraction | C0-1 | `.superpowers/swarm/reports/c0-task-4-dwarfstar.md` | Done |

### Supervisor gates
- Report checks: proof sources exist where claimed; rows record evidence; DwarfStar note separates use/non-use; no proof agent made the final substrate decision.
- Quality bar result: passed for C0 proof wave after reviewer found only ledger-resume cleanup; final C0 state is blocked, not successful substrate selection.
- Review agents: after wave if native probe code or docs carry ambiguity/over-engineering risk.
- Verification command(s) supervisor will run: C0 macOS command, Linux command or blocker confirmation, `git diff --check docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/validation-commands.md` plus any new note path.
- Ledger update: mark C0-2/3/4 `Needs review`, then `Done` or `Blocked` after verification.

## Wave 3: Mac-first C0A planning
### Shared context
# Goal
Convert user steering into a macOS-first continuation plan without falsely unblocking C1.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- C0 selected state remains `blocked`; this wave changes execution focus only.
- Do not run C1/C2/C3 work until C0 task set 5 reruns to a passing substrate decision.

# Contract
- Initial follow-up focuses on local macOS eGPU runtime.
- The plan must start with device visibility and ABI pinning before transfer/kernel implementation.
- Linux HIP stays as reference fallback if macOS remains blocked.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | C0A-1 | `phase-c0a-macos-egpu-runtime-focus.md`, README/C0/validation/progress updates | C0 handoff and user steering | n/a | Done |

### Supervisor gates
- Report checks: new C0A doc has executable task sets, exact validation where known, and no C1 unblocking without a passing macOS proof.
- Quality bar result: pending verification/review.
- Verification command(s) supervisor will run: `git diff --check docs/tasks/native-r9700-producer/README.md docs/tasks/native-r9700-producer/phase-c0-runtime-discovery.md docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md`.
- Ledger update: mark C0A plan row Done; C0A execution rows are now tracked separately in `.superpowers/swarm/progress.md`.

## Wave 4: C0A mac device visibility rerun
### Shared context
# Goal
Correct the macOS device-visibility gate against the actual working Phase 0 substrate.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer` (fallback linked worktree because the source checkout was on `main`).
- Serial C0A gate: do not touch transfer implementation, kernel-launch implementation, C1, C2, or C3.
- Use the C0A TinyGPU.app/IOKit PCI discovery command in `validation-commands.md` and the user-provided working command `JITBEAM=2 DEV=AMD python3 -m tinygrad.llm` as substrate evidence.
- The discovery command may import tinygrad as reference evidence. Path C native producer code must still be tinygrad-free.
- Report path: `.superpowers/swarm/reports/c0a-task-1-mac-device-visibility.md`.

# Contract
- Success requires `System.list_devices(...)` reporting `APLRemotePCIDevice '1002:7551'` and `Device['AMD']` instantiating as `PCIIface`, `pcibus usb4`, `arch gfx1201`.
- The stale libusb-only `USBIface` probe is a negative control, not the C0A visibility gate.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AMacDeviceVisibility | C0A task set 1 | Corrected TinyGPU.app/IOKit PCI discovery evidence and C0A progress row | C0A plan | `.superpowers/swarm/reports/c0a-task-1-mac-device-visibility.md` | Done |

### Supervisor gates
- Report checks: passed after correction; report identifies the bad libusb-only assumption and the working TinyGPU.app/APLRemotePCIDevice/PCIIface path.
- Quality bar result: correctness corrected; maintainability improved by preserving stale negative-control evidence separately; architectural fit passed because tinygrad is only reference evidence and Path C remains tinygrad-free; simplicity passed because the next wave pins the existing runtime path instead of inventing another substrate.
- Review agents: not dispatched for this correction; evidence comes from tinygrad source and focused runtime discovery output.
- Verification command(s) supervisor ran: `PYTHONPATH=${HOME}/Development/ml/tools/tinygrad DEV=AMD DEBUG=1 ${HOME}/.pyenv/versions/3.12.8/bin/python3 -c ...` -> `APLRemotePCIDevice '1002:7551'`, `PCIIface`, `arch gfx1201`, `pcibus usb4`.
- Ledger update: C0A visibility is Done; C0A ABI pinning starts against TinyGPU.app/APLRemotePCIDevice/PCIIface.

## Wave 5: C0A TinyGPU ABI pinning
### Shared context
# Goal
Pin the minimal native contract behind the working macOS TinyGPU.app/APLRemotePCIDevice/PCIIface path so transfer and kernel proof can be implemented without guessing.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Source of truth for discovery: `${HOME}/Development/ml/tools/tinygrad/`.
- Do not implement transfer/kernel code in this wave.
- Do not vendor or copy tinygrad code; record facts, source paths, line references, and license/safety boundaries.
- OMP task executors do not run tests, linters, formatters, package managers, git commands, or project-wide suites.
- Report path: `.superpowers/swarm/reports/c0a-task-2-tinygpu-abi.md`.

# Contract
- ABI note path: `docs/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md`.
- Must cover `APLRemotePCIDevice`, `RemotePCIDevice`, `PCIIface`, `PCIIfaceBase`, `AMDev`, allocation/mapping/readback, queue creation/submission/synchronization, and where TinyGPU.app/kernel extension owns privileged operations.
- Must distinguish tinygrad reference commands from Path C tinygrad-free native producer requirements.

### Supervisor gates
- Report checks: passed; ABI note and report identify TinyGPU.app/APLRemotePCIDevice/PCIIface as the active path and `USBIface`/libusb as a stale negative control.
- Reviewer: `C0AABIReviewer` found no Critical findings and two Important wording/validation findings: stale libusb visibility dependency and task sets 3/4 pointing at the negative-control command.
- Fix result: supervisor corrected `phase-c0a-macos-egpu-runtime-focus.md` and `validation-commands.md` so task sets 3/4 must discover TinyGPU.app/APLRemotePCIDevice/PCIIface commands and must not use the stale libusb command for acceptance.
- Quality bar result: correctness passed after correction; maintainability passed because the active substrate and negative control are now separated; architectural fit passed because Path C remains tinygrad-free and tinygrad remains reference/discovery only; simplicity passed because no runtime abstraction was added.
- Verification command(s) supervisor ran: `git diff --check docs/tasks/native-r9700-producer/validation-commands.md docs/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md .superpowers/swarm/reports/c0a-task-2-tinygpu-abi.md` -> no output.
- Ledger update: C0A ABI pinning is Done; C0A transfer proof is In progress.

## Wave 6: C0A host-device transfer proof
### Shared context
# Goal
Implement or precisely block the smallest tinygrad-free host↔device transfer proof on the working macOS TinyGPU.app/APLRemotePCIDevice/PCIIface path.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Source of truth for discovery and ABI: `docs/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md` and `${HOME}/Development/ml/tools/tinygrad/`.
- Do not call or import tinygrad from the native proof; tinygrad is allowed only for separately labeled reference/discovery commands already recorded in `validation-commands.md`.
- Do not use `USBIface`, libusb, or `USB3.list_devices(0xADD1, 0x0001)` as the working path; `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp` is a stale negative control unless replaced/bypassed.
- No model code, no generic runtime framework, no C1 runtime wrapper, no kernel dispatch in this wave.
- OMP task executors do not run tests, linters, formatters, package managers, project-wide suites, or git commands.
- Report path: `.superpowers/swarm/reports/c0a-task-3-transfer-proof.md`.

# Contract
- Transfer proof target: update or replace `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp` only if it becomes the TinyGPU.app/APLRemotePCIDevice/PCIIface proof; otherwise add the narrowest source under `experiments/native-r9700-runtime/` and label the stale libusb source as negative control in docs.
- Validation command: add an exact build/run/log command to `docs/tasks/native-r9700-producer/validation-commands.md` for the TinyGPU.app/APLRemotePCIDevice/PCIIface transfer proof.
- Required log fields: substrate, PCI id `1002:7551` if selected, arch if discovered, transfer byte count, CPU comparison result, `host_device_transfer_status`, `failure_text` on error, and wrapper `exit_status`.

### Supervisor gates
- Report checks: passed; report states source was not changed, stale libusb was not used, and command discovery is blocked rather than faked.
- Reviewer: `C0ATransferReviewer` found no Critical or Important findings. Minor documentation consistency finding on `macos-tinygpu-abi-notes.md` line 124 was fixed.
- Quality bar result: correctness passed for a blocked gate because the blocker is source-grounded; maintainability passed because the next unblocker is explicit; architectural fit passed because Path C does not import tinygrad and does not use stale libusb for acceptance; simplicity passed because no partial AMDev/SDMA scaffold or broad runtime framework was added.
- Verification command(s) supervisor will run: `git diff --check docs/tasks/native-r9700-producer/README.md docs/tasks/native-r9700-producer/validation-commands.md docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md .superpowers/swarm/reports/c0a-task-3-transfer-proof.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md`.
- Ledger update: C0A transfer proof is Blocked; C0A kernel proof and mac-focused decision rerun are Blocked downstream.

## Wave 7: C0B task documentation and RED contract setup
### Shared context
# Goal
Convert the approved native AMDev/SDMA boundary spec into executable task documents and start the TDD gate for the native transfer proof.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Source spec: `docs/superpowers/specs/2026-08-16-native-amdev-sdma-boundary-design.md`.
- Implementation plan: `docs/superpowers/plans/2026-08-16-native-amdev-sdma-transfer.md`.
- Task doc: `docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`.
- TDD gate: no production C++ source before supervisor observes the focused contract pytest fail for expected missing-source reason.
- OMP task executors do not run tests, linters, formatters, package managers, project-wide suites, or git commands.

# Contract
- C0B task set 1 owns only `tests/test_native_amdev_transfer_contract.py`, validation-command RED entry, its task-doc row, and `.superpowers/swarm/reports/c0b-task-1-red-contract.md`.
- Later C0B tasks are serial: RemoteCmd self-tests -> discovery smoke -> VM/sysmem -> SDMA transfer -> review/handoff.
- The native proof may port minimal MIT tinygrad slices with provenance; it must not import/call tinygrad at runtime and must not use libusb/`USBIface` as acceptance.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BRedContract | C0B-1 | RED pytest contract and validation ledger entry | approved spec | `.superpowers/swarm/reports/c0b-task-1-red-contract.md` | Done |

### Supervisor gates
- Report checks: passed; report names changed files, exact supervisor command, expected RED failure text, and confirms no production source was added.
- Quality bar result: correctness passed because both tests fail only on the required missing-source assertion before task set 2; maintainability passed because the test names and required log-field list match the public contract; architectural fit passed because the gate does not import tinygrad, use libusb, or touch hardware; simplicity passed because the contract has two focused self-test checks and no helper framework.
- Review agents: `C0BRedReviewer` accepted with no Critical, Important, or Minor findings.
- Verification command(s) supervisor ran: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> exit `1`, `2 failed`, both at `AssertionError: native transfer probe source missing`; `git diff --check docs/superpowers/plans/2026-08-16-native-amdev-sdma-transfer.md docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md docs/tasks/native-r9700-producer/README.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c0b-task-1-red-contract.md tests/test_native_amdev_transfer_contract.py` -> no output.
- Ledger update: C0B-1 Done; C0B-2 unblocked.

## Wave 8: C0B RemoteCmd transport self-tests
### Shared context
# Goal
Turn the RED no-hardware contract green by adding the minimal native probe source with RemoteCmd request framing and log-contract self-tests.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Task doc: `docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md` task set 2.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- This wave must not connect to TinyGPU.app, map BAR/sysmem, create VM mappings, run SDMA, use libusb, import/call tinygrad, or touch C1/C2/C3.
- OMP task executors do not run tests, linters, formatters, package managers, project-wide suites, or git commands.

# Contract
- Implement `RemoteCmd` enum in exact tinygrad order: `PROBE`, `MAP_BAR`, `MAP_SYSMEM_FD`, `CFG_READ`, `CFG_WRITE`, `RESET`, `MMIO_READ`, `MMIO_WRITE`, `MAP_SYSMEM`, `SYSMEM_READ`, `SYSMEM_WRITE`, `RESIZE_BAR`, `PING`.
- Implement a little-endian request frame equivalent to tinygrad `struct.pack('<BIIQQQ', cmd, dev_id, bar, arg0, arg1, arg2)`.
- Implement `--self-test remote-cmd-frame` to validate and print `frame_size: 33` plus `frame_hex: 0251750000050000000807060504030201887766554433221100ffeeddccbbaa99`; implement `--self-test log-contract` and `--help` only; hardware transfer remains unimplemented after this wave.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BRemoteCmd | C0B-2 | Minimal native probe source and no-hardware self-tests | C0B-1 | `.superpowers/swarm/reports/c0b-task-2-remote-pci.md` | Done |

### Supervisor gates
- Report checks: passed; report names changed files, provenance, guardrails, exact supervisor command, and expected/observed `2 passed`.
- Quality bar result: correctness passed because pytest validates the exact 33-byte little-endian frame hex and required log fields; maintainability passed because the C++ source is a small local CLI with direct helpers; architectural fit passed because it only ports the pinned RemoteCmd framing and does not connect to TinyGPU.app or introduce runtime tinygrad/libusb; simplicity passed because no transport framework or allocator was added.
- Review agents: `C0BRemoteCmdReviewer` accepted with no Critical, Important, or Minor findings.
- Verification command(s) supervisor ran: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `2 passed in 0.68s`; `git diff --check experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp tests/test_native_amdev_transfer_contract.py docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c0b-task-2-remote-pci.md` -> no output.
- Ledger update: C0B-2 Done; C0B-3 unblocked.

## Wave 9: C0B TinyGPU discovery smoke
### Shared context
# Goal
Extend the native probe from no-hardware RemoteCmd self-tests to a TinyGPU.app discovery-smoke mode that produces precise discovery evidence or a precise failure stage without claiming transfer success.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Task doc: `docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md` task set 3.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Tests target: `tests/test_native_amdev_transfer_contract.py`; supervisor already observed RED help coverage fail on missing `--discovery-smoke` while the two existing self-tests passed.
- This wave may connect to TinyGPU.app over UNIX socket and map BAR0/BAR2/BAR5. It must not create VM mappings, map sysmem, run SDMA, run kernel dispatch, use libusb, import/call tinygrad, or touch C1/C2/C3.
- OMP task executors do not run tests, linters, formatters, package managers, project-wide suites, git commands, or hardware commands.

# Contract
- Implement help entries for `--discovery-smoke` and `--transfer-proof`; `--transfer-proof` may return a precise `not_implemented_transfer_proof` failure until later task sets.
- Implement TinyGPU.app UNIX socket path selection using `APL_REMOTE_SOCK` if set, otherwise a temp socket path. Launch TinyGPU.app server only if a socket connect fails.
- Implement RemoteCmd response decoding for status/error text and enough BAR mapping to report PCI id `1002:7551`, BAR sizes, and `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`.
- Discovery smoke output must include `failure_stage` and `failure_text` on error, and must not log `host_device_transfer_status: pass`.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BDiscovery | C0B-3 | TinyGPU.app discovery-smoke CLI and validation command | C0B-2 | `.superpowers/swarm/reports/c0b-task-3-discovery.md` | Done |

### Supervisor gates
- Report checks: passed; report records the PROBE->CFG_READ root-cause correction, exact commands, hardware log evidence, review fixes, and remaining VM/sysmem scope.
- Quality bar result: correctness passed because hardware evidence selects `1002:7551`, records BAR and VRAM-size data, and never claims transfer success; maintainability passed because discovery remains a narrow CLI mode with explicit failure stages; architectural fit passed because it follows the TinyGPU.app/APLRemotePCIDevice/PCIIface path with no stale libusb/tinygrad runtime dependency; simplicity passed because VM/sysmem/SDMA remain in later task sets.
- Review agents: `C0BDiscoveryReviewer` found two Important issues and one Minor issue; supervisor fixed required VRAM-size failure behavior, `SO_NOSIGPIPE`, and the report changed-file list. `C0BDiscoveryReReviewer` accepted with no remaining findings.
- Verification command(s) supervisor ran after review fixes: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `3 passed in 1.59s`; discovery command from `validation-commands.md` -> exit `0`, `pci_id: 1002:7551`, BAR0 `268435456`, BAR2 `2097152`, BAR5 `524288`, `vram_size_bytes: 34208743424`, `host_device_transfer_status: not_run`, `failure_stage: none`; `git diff --check ...` -> no output.
- Ledger update: C0B-3 Done; C0B-4 unblocked.

## Wave 10: C0B VM/sysmem mapping port
### Shared context
# Goal
Extend the native probe from discovery-only evidence to the minimal sysmem page-list parsing and VM/sysmem mapping scaffolding needed before SDMA transfer proof.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Task doc: `docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md` task set 4.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Tests target: `tests/test_native_amdev_transfer_contract.py`.
- Report target: `.superpowers/swarm/reports/c0b-task-4-vm-sysmem.md`.
- Validation policy: OMP task executors do not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; supervisor runs verification after the wave.
- Non-goals: no SDMA submission, no compute queue, no kernel dispatch, no general allocator framework, no model code, no tinygrad runtime import/call/shell-out.

# Contract
- Add RED no-hardware coverage for `--self-test sysmem-page-list`.
- Port synthetic `(paddr, size)` page-list parsing from MAP_SYSMEM_FD mappings: little-endian `(uint64 paddr, uint64 size)` pairs ending in `(0, 0)`, expanded at 4 KiB granularity and truncated to the requested page count.
- Add only fixed transfer-proof VM/sysmem scaffolding for one VRAM buffer plus CPU-visible staging/readback buffers. If real hardware sysmem mapping cannot proceed, fail closed with `failure_stage: vm_mapping` and precise RemoteCmd/error text.
- Keep TinyGPU.app/APLRemotePCIDevice/PCIIface and the `CFG_READ` discovery path from C0B-3; do not reintroduce `PROBE` or stale libusb paths.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BVMSysmem | C0B-4 | sysmem page-list self-test and VM/sysmem mapping scaffolding | C0B-3 | `.superpowers/swarm/reports/c0b-task-4-vm-sysmem.md` | Done |

### Supervisor gates
- Report checks: passed; report names changed files, MAP_SYSMEM_FD/sysmem page-list behavior, fixed VM roles, guardrails, blocker, review fixes, and supervisor commands/results.
- Quality bar result: correctness passed because no-hardware parser contract passes and hardware smoke reaches sysmem page-list evidence before failing closed at the intended `vm_mapping` stage; maintainability passed because fd/mmap ownership is local RAII and the code remains fixed-role rather than allocator-framework; architectural fit passed because MAP_SYSMEM_FD framing now matches TinyGPU/tinygrad reference and no stale libusb/tinygrad runtime path was added; simplicity passed because SDMA/PTE/TLB remain in task set 5.
- Review agents: `C0BVMSysmemReviewer` rejected first pass for MAP_SYSMEM_FD frame fields and fd/mmap lifetime; supervisor fixed both. `C0BVMSysmemReReviewer` accepted with no remaining findings.
- Verification command(s) supervisor ran after review fixes: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `4 passed in 2.26s`; VM/sysmem smoke command -> exit `1`, wrapper exit `1`, staging page `0x0000000080000000`, readback page `0x0000000080008000`, `failure_stage: vm_mapping`, no transfer success claim; `git diff --check ...` -> no output.
- Ledger update: C0B-4 Done; C0B-5 unblocked.

## Wave 11: C0B SDMA transfer proof
### Shared context
# Goal
Move from VM/sysmem evidence to the actual 32-byte SDMA transfer proof or a precise hard blocker with no fake success.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Task doc: `docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md` task set 5.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Tests target: `tests/test_native_amdev_transfer_contract.py`.
- Validation command target: `docs/tasks/native-r9700-producer/validation-commands.md`.
- Report target: `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md`.
- Validation policy: OMP task executors do not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; supervisor runs verification after the wave.
- Non-goals: no compute kernel dispatch, no multi-device support, no production runtime wrapper, no C0 substrate decision, no model code, no tinygrad runtime import/call/shell-out, no stale libusb path.

# Contract
- Add RED no-hardware coverage for `--self-test sdma-packet-encoding`: a 32-byte linear-copy packet must encode source/destination addresses little-endian and count `31`.
- Implement SDMA queue 0 setup, ring write, BAR2 doorbell, linear-copy packets, completion/fence/timeline polling, and bounded timeout only as needed for one 32-byte staging -> VRAM -> readback staging proof.
- `--transfer-proof` may exit `0` only when `host_device_transfer_status: pass`, `transfer_byte_count: 32`, and CPU byte comparison success are logged. Any failure must name the exact stage and exit nonzero.
- Add exact build/run/log command for `logs/c0b-native-amdev-sdma-transfer.log` to `validation-commands.md`.
- If AMD register/PTE/TLB/SDMA setup cannot be completed from available source and observed hardware, record the exact blocker in the report/log rather than guessing defaults.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BSDMATransfer | C0B-5 | SDMA packet self-test and 32-byte transfer proof | C0B-4 | `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md` | Blocked |

### Supervisor gates
- Report checks: passed; report names changed files, SDMA packet self-test, validation command, exact `vm_mapping` blocker, guardrails, expected success evidence, supervisor evidence, and reviewer acceptance.
- Quality bar result: correctness passed for the blocked task-5 outcome because packet encoding is covered and hardware command exits nonzero with exact VM/PTE/TLB blocker rather than claiming transfer success; maintainability passed because the source stops at explicit missing prerequisites; architectural fit passed because it keeps TinyGPU.app/APLRemotePCIDevice and does not add stale libusb/tinygrad runtime paths; simplicity passed because it avoids guessing register/PTE defaults.
- Review agents: `C0BSDMAReviewer` accepted with no Critical, Important, or Minor findings.
- Verification command(s) supervisor ran: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `5 passed in 2.92s`; hardware transfer command from `validation-commands.md` -> exit `1`, wrapper exit `1`, `failure_stage: vm_mapping`, `sdma_linear_copy_packet_dwords: 7`, `host_device_transfer_status: fail`, `cpu_comparison_status: not_run`, no transfer success claim; `git diff --check ...` -> no output.
- Ledger update at Wave 11 time: C0B-5 was Blocked with a reviewed precise VM blocker; C0B-6 unblocked for blocker handoff. Later waves superseded this state.

## Wave 12: C0B review and C0 handoff
### Shared context
# Goal
Record the reviewed C0B outcome and keep downstream C0A/C1/C2/C3 gates blocked because the native transfer proof did not pass.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Task doc: `docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md` task set 6.
- Handoff must not select a C0 substrate without transfer pass plus later kernel evidence.
- Non-goals: no source implementation, no kernel proof, no production runtime wrapper.

# Contract
- C0B ledger reflects reviewed transfer state: C0B-5 Blocked at `vm_mapping`; C0B-6 records handoff.
- C0A task sets 3-5 remain blocked because transfer did not pass.
- README and reports state exact log path and next gate.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | C0B-6 | C0B handoff documentation and downstream gates | C0B-5 blocker | `.superpowers/swarm/reports/c0b-task-6-review-handoff.md` | Done |

### Supervisor gates
- Report checks: first final reviewer found stale global `C0A-4` blocker wording and pending-diff-check wording; supervisor corrected both. Re-review accepted.
- Quality bar result: correctness passed because C0B blocker/downstream gates are consistent across ledger, README, C0A/C0B docs, and report; maintainability passed because the next gate is exact and log-backed; architectural fit passed because no substrate is selected without transfer/kernel evidence; simplicity passed because no new workaround path or fallback abstraction was added.
- Review agents: `C0BHandoffReviewer` rejected first pass with one Important and one Minor finding; `C0BHandoffReReviewer` accepted with no remaining findings.
- Verification command(s) supervisor ran after review fixes: `git diff --check docs/tasks/native-r9700-producer/README.md docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c0b-task-6-review-handoff.md` -> no output.
- Ledger update: C0B-6 Done; C0B remains blocked at `vm_mapping`.

## Wave 13: gfx12 VM/PTE/TLB prerequisite planning
### Shared context
# Goal
Split the reviewed C0B `vm_mapping` blocker into an implementation plan and agent-executable task docs before returning to source work.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`.
- Documentation-only wave: no source implementation, no hardware rerun, no C0 substrate selection.
- Preserve C0B-5 history; add C0B-4.5 as the explicit missing prerequisite.

# Contract
- Implementation plan path: `docs/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md`.
- Task docs folder: `docs/tasks/native-r9700-gfx12-vm-pte-tlb/`.
- C0A/C0B docs and progress ledger point to the new prerequisite.
- C1/C2/C3 remain blocked.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| Main | C0B-4.5 planning | VM/PTE/TLB implementation plan and task docs | C0B-5 reviewed blocker | `docs/tasks/native-r9700-gfx12-vm-pte-tlb/README.md` | Done |

### Supervisor gates
- Report checks: implementation plan and task docs created; C0B-4.5 inserted without deleting C0B-5 evidence.
- Quality bar result: correctness passed because the new docs preserve the reviewed `vm_mapping` blocker and do not unblock C0A/C1/C2/C3; maintainability passed because the missing prerequisite is isolated as C0B-4.5 with source refs, TDD gates, exact validation commands, and handoff rules; architectural fit passed because no C0 substrate is selected and no new runtime boundary is invented; simplicity passed because the implementation scope is one fixed VM/PTE/TLB prerequisite, not a broad allocator/backend framework.
- Review agents: not dispatched; this is a task-doc creation wave, not implementation.
- Verification command(s) supervisor ran: `git diff --check docs/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md docs/tasks/native-r9700-gfx12-vm-pte-tlb/README.md docs/tasks/native-r9700-gfx12-vm-pte-tlb/phase-1-contracts-and-source-grounding.md docs/tasks/native-r9700-gfx12-vm-pte-tlb/phase-2-fixed-vm-mapping.md docs/tasks/native-r9700-gfx12-vm-pte-tlb/phase-3-transfer-resume-and-handoff.md docs/tasks/native-r9700-producer/README.md docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md` -> no output.
- Ledger update: C0B-4.5 Not started; C0B-5 remains Blocked on C0B-4.5.

## Wave 14: C0B-4.5 gfx12 VM/PTE/TLB prerequisite execution
### Shared context
# Goal
Implement the split-out gfx12 VM/PTE/TLB prerequisite for the native TinyGPU.app/APLRemotePCIDevice/PCIIface transfer proof, starting with RED no-hardware contracts and only then adding deterministic C++ VM self-tests.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every C0B-4.5 executor stays in this cwd/branch.
- Task docs: `docs/tasks/native-r9700-gfx12-vm-pte-tlb/`.
- Implementation plan: `docs/superpowers/plans/2026-08-16-gfx12-vm-pte-tlb-mapping.md`.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Tests target: `tests/test_native_amdev_transfer_contract.py`.
- Report targets: `.superpowers/swarm/reports/c0b-vm-task-1-contracts.md`, `.superpowers/swarm/reports/c0b-vm-task-2-selftests.md`, `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md`, `.superpowers/swarm/reports/c0b-vm-task-4-transfer-resume.md`.
- TDD gate: VM contract pytest expectations are written and observed RED before new VM C++ helpers or hardware VM writes are added.
- OMP task executors do not run tests, linters, formatters, package managers, git commands, hardware commands, or project-wide suites; supervisor runs verification after each wave.
- Do not start C0A kernel proof, C1, C2, or C3 unless the transfer command later logs a real 32-byte pass with CPU comparison and exit 0.
- Do not guess gfx12 PTE flags, register offsets, VM context values, or TLB flush sequences; every constant must cite tinygrad or generated AMD header lines.
- No libusb/`USBIface` acceptance path, no runtime tinygrad import/call/shell-out, no TinyGPU.app server/kernel-extension rewrite, no allocator/backend/scheduler framework.

# Contract
- Phase 1 is serial: `C0BVmRedContract` edits only `tests/test_native_amdev_transfer_contract.py` and writes the RED report; `C0BVmSelfTests` edits only `native_amdev_transfer_probe.cpp` after supervisor observes RED.
- Self-test names: `am-vm-pte-encoding`, `am-vm-page-table-plan`, `am-vm-tlb-sequence`.
- Expected PTE/page-table/TLB outputs are exactly those in `phase-1-contracts-and-source-grounding.md`.
- Phase 2 remains one source owner because page-table writes, VMID0 context programming, and TLB invalidation all touch the same C++ transfer path.
- Review gates reject guessed constants, stale generic `vm_mapping` blocker text after implementation, fake success, hidden tinygrad runtime paths, libusb acceptance, broad abstractions, or missing provenance.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BVmRedContract | C0B-4.5 Phase 1 task set 1 | VM RED pytest contract tests | C0B-5 reviewed `vm_mapping` blocker | `.superpowers/swarm/reports/c0b-vm-task-1-contracts.md` | Done |
| C0BVmSelfTests | C0B-4.5 Phase 1 task set 2 | Deterministic C++ VM self-tests | Supervisor-observed RED | `.superpowers/swarm/reports/c0b-vm-task-2-selftests.md` | Done |
| C0BVmHardwareMapping | C0B-4.5 Phase 2 task sets 1-2 | Fixed page tables, VMID0 context, TLB sequence | GREEN/reviewed self-tests | `.superpowers/swarm/reports/c0b-vm-task-3-hardware-mapping.md` | Done |
| C0BVmTransferResume | C0B-4.5 Phase 3 task sets 1-2 | Transfer rerun classification and durable handoff update | Phase 2 review/evidence | `.superpowers/swarm/reports/c0b-vm-task-4-transfer-resume.md` | Done |

### Supervisor gates
- Report checks: passed for Phase 1 reports, Phase 2 hardware mapping report, Phase 2 review report, transfer resume report, final review report, changed source/tests, and updated handoff docs.
- Quality bar result: Phase 1, Phase 2, and final review accepted; correctness, maintainability, architectural fit, and simplicity/no over-engineering all passed.
- Review agents: `C0BVmPhase1Reviewer` accepted with no Critical, Important, or Minor findings; durable report `.superpowers/swarm/reports/c0b-vm-phase1-review.md`. `C0BVmPhase2Reviewer` accepted with no Critical, Important, or Minor findings; durable report `.superpowers/swarm/reports/c0b-vm-phase2-review.md`. `C0BVmFinalReviewer` accepted with no Critical, Important, or Minor findings; durable report `.superpowers/swarm/reports/c0b-vm-final-review.md`.
- Verification command(s) supervisor ran: RED focused pytest exited `1` with absent VM self-tests; GREEN focused pytest reported `8 passed in 4.87s`; final focused pytest after Phase 2 reported `8 passed in 6.54s`; final hardware transfer command wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T12:40:50Z` and exited `1` with `failure_stage: sdma_ring_setup`, `vm_page_tables_written: pass`, `vmid0_context_status: pass`, and `mm_tlb_flush_status: pass`.
- Ledger update at Wave 14 time: C0B-4.5 was Done; C0B-5 and C0A-4 remained Blocked on SDMA ring setup/submission. Wave 15 later completed C0B-5 and C0A-4.

## Wave 15: C0B SDMA queue0 hardware submit implementation
### Shared context
# Goal
Implement the source-grounded SDMA queue 0 setup/submission path for the fixed 32-byte C0B transfer proof without running validation in task-agent mode.

# Constraints
- Required work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Reports: `.superpowers/sdd/2026-08-17-native-sdma-ring-transfer/task-3-report.md` and `.superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md`.
- Task agent did not run focused pytest, build, hardware transfer, lint, formatter, package-manager, git, or project-wide commands.
- Do not unblock C0A kernel proof/C1/C2/C3 without supervisor-observed pass tokens.

# Contract
- Transfer proof must pass only after CPU readback matches the fixed 32-byte payload.
- Nonzero hardware outcomes must classify precisely at `sdma_ring_setup`, `sdma_submit`, `timeline_timeout`, or `readback_mismatch` after VM setup.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0BSDMAHardware | C0B SDMA Task 3 | SDMA0 IP logging, sdma_control mapping, fourth PTB leaf, queue0 setup, submit, fence poll, CPU compare | C0B-4.5 and SDMA Task 2 helpers | `.superpowers/swarm/reports/c0b-sdma-task-3-hardware-submit.md` | Done |

### Supervisor gates
- Verification command(s) supervisor ran: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` passed `11 passed in 9.94s`.
- Hardware transfer proof command from `docs/tasks/native-r9700-producer/validation-commands.md` wrote `logs/c0b-native-amdev-sdma-transfer.log` at `2026-08-17T13:31:58Z` and exited `0`.
- Pass tokens observed: `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, `arch: gfx1201`, `sdma_ip_version: 7.0.1`, `sdma_queue_setup_status: pass`, `sdma_submit_status: pass`, `sdma_timeline_status: pass`, `transfer_byte_count: 32`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`.
- Quality bar result: initial stale SDMA 4.4.2 assumption was rejected by the focused RED and source/root-cause review; repeated-run `RB_WPTR=0x48` regression was traced to missing queue teardown/reset before setup and fixed with the tinygrad `AM_SDMA.fini_hw` disable/soft-reset sequence. Corrected implementation is source-grounded to the local gfx1201 SDMA0 7.0.1 `regSDMA0_QUEUE0` path, fixed-shape, and avoids new scheduler/runtime abstractions.
- Ledger update at Wave 15 time: C0B-5 and C0A-4 were Done; C0A-5 minimal kernel proof was Not started/unblocked then and is resumed in Wave 16; C1, C2, and C3 remain blocked until kernel proof and C0 decision rerun select a substrate or actionable split.

## Wave 16: C0A minimal kernel launch proof
### Shared context
# Goal
Prove the smallest tinygrad-free macOS R9700 kernel dispatch/readback path on the existing TinyGPU.app/APLRemotePCIDevice/PCIIface substrate, reusing the reviewed native AMDev/VM/SDMA transfer substrate rather than creating a new runtime abstraction.

# Constraints
- Required shared work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every executor and reviewer stays in this cwd/branch.
- Source target: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Test target: `tests/test_native_amdev_transfer_contract.py`.
- Task doc row: `docs/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md` task set 4 / ledger row `C0A-5. Minimal kernel launch proof`.
- Report target: `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md`.
- Validation ledger target: `docs/tasks/native-r9700-producer/validation-commands.md`.
- Runtime path must remain tinygrad-free. tinygrad source may be read only as source/provenance reference.
- Stale libusb-only `USBIface` / `USB3.list_devices(0xADD1, 0x0001)` path remains a negative control and must not be used as acceptance evidence.
- No C0 decision rerun, C1, C2, C3, scheduler, allocator framework, model path, MLX integration, or production runtime API in this wave.
- OMP task executors do not run tests, linters, formatters, package managers, broad validation suites, hardware commands, or git commands; supervisor runs focused tests and hardware proof after the wave.

# Contract
- TDD gate: add no-hardware contract expectations for a minimal kernel proof mode before production C++ changes, and supervisor must observe the focused RED failure.
- Kernel proof mode should be fixed-shape and reviewable: tiny sample input, explicit kernel metadata/blob/path, dispatch/timing/status fields, readback, exact CPU comparison.
- The hardware command must be added to `validation-commands.md` before supervisor execution and write `logs/c0-macos-egpu-minimal-runtime.log`.
- Passing log must include device identity, `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `failure_stage: none`, and wrapper `exit_status: 0`.
- If a true kernel dispatch is blocked, the report and log must classify the exact missing compute-queue/code-object/dispatch capability and must not substitute SDMA copy success, tinygrad execution, or libusb probing as a pass.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0AKernelProof / C0AKernelImpl | C0A task set 4 / C0A-5 | Minimal native kernel proof contract/source/docs plus precise compute blocker | C0A-4 transfer pass | `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md` | Blocked |

### Supervisor gates
- Report checks: executor report, source/docs inspection, and hardware log all agree that the current implementation verifies TinyGPU.app/APLRemotePCIDevice discovery, VM/MMHUB/TLB setup, and SDMA substrate round trip, then fails closed at `compute_ring_setup` instead of substituting SDMA success for kernel proof.
- Quality bar result: correctness passed for the non-passing precise-blocker outcome; maintainability passed because the contract fields, validation command, and report identify the missing GC/RLC/MEC/SH_MEM/MQD/HQD/compute-doorbell port; architectural fit passed because runtime remains experiment-local and tinygrad-free; simplicity passed because the code stays fixed-shape/direct and does not add a runtime framework.
- Review agents: `C0AKernelReview` completed with no Critical, Important, or Minor findings.
- Verification command(s) supervisor ran: focused pytest `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` passed `12 passed in 11.00s`; kernel proof command from `validation-commands.md` wrote `logs/c0-macos-egpu-minimal-runtime.log` at `2026-08-17T14:17:49Z` and exited `1` with `kernel_launch_status: blocked`, `host_device_transfer_status: pass`, `failure_stage: compute_ring_setup`, and `wrapper_exit_status: 1`; `git diff --check` passed.
- Ledger update: C0A-5 is Blocked on native gfx1201 compute ring setup; C0A-6 remains Blocked until a CPU-verified kernel proof pass exists or the user approves/reactivates a fallback substrate or split decision path.

## Wave 17: C0A compute dispatch split decision
### Supervisor gates
- Report checks: `.superpowers/swarm/reports/c0a-compute-split-decision.md` records emitted `kernel_timeline_timeout`, accepted inferred `compute_doorbell_not_consumed`, latest log `logs/c0c-native-amdev-kernel-dispatch.log`, pass-through prerequisite tokens, and three decision options.
- Quality bar result: final re-review accepted after supervisor fixed docs/ledger/report consistency findings. Current recommendation preserves C0 scope and C1/C2/C3 blocking state: continue native macOS GFX port with one named MEC doorbell delivery/ring-fetch investigation.
- Verification command(s) supervisor ran for the PM4 blocker before this decision: focused PM4 pytest passed; full `tests/test_native_amdev_transfer_contract.py -v` passed `17 passed in 19.87s`; hardware `--kernel-proof` exited `1` with accepted blocker evidence. Final checkpoint-prep pytest exited `0` with `17 passed in 20.21s`; `git diff --check` produced no output.
- Ledger update: C0A task set 4 remains Blocked on MEC doorbell delivery/ring fetch; C0A task set 5 remains Blocked; downstream C1/C2/C3 remain blocked until CPU-verified pass tokens or user-approved fallback/split. C0A compute dispatch checkpoint-prep rows are Done for the accepted-blocker state.

## Wave 18: C0A MEC doorbell delivery diagnostic
### Supervisor gates
- Report checks: `.superpowers/swarm/reports/c0a-compute-task-6-doorbell-delivery.md` records the exact C0D command, log path `logs/c0d-native-amdev-doorbell-delivery.log`, prerequisite pass tokens through kernel blob/kernargs/SDMA H2D/compute ring/HQD active, all five `compute_doorbell_probe_*` fields, emitted `failure_stage: kernel_timeline_timeout`, and inferred `compute_doorbell_not_consumed`.
- Quality bar result: `DoorbellReview` passed correctness, maintainability, architectural fit, and simplicity/no over-engineering. The implementation stays fixed-shape, tinygrad-free, and diagnostic-only with no register/PM4 fix, retry/fallback/scheduler/allocator framework, or C1/C2/C3 work.
- Review agents: `DoorbellPhase1Review` accepted Phase 1 with 0 Critical/Important/Minor and `ready_for_phase2: true`; `DoorbellReview` accepted the final diagnostic boundary with 0 Critical, 0 Important, 1 Minor ledger-propagation note, and `ready_for_checkpoint: true`.
- Verification command(s): RED focused self-test failed for the intended unknown self-test; GREEN focused self-test and help-list tests passed; full no-hardware pytest after instrumentation passed `18 passed in 20.97s`; hardware C0D command wrote `logs/c0d-native-amdev-doorbell-delivery.log` at `2026-08-17T19:06:34Z`, `exit_status: 1`, `wrapper_exit_status: 1`, classification `compute_doorbell_not_consumed`; final no-hardware pytest passed `18 passed in 21.31s`; `git diff --check` printed no output.
- Ledger update: C0A task set 4 remains Blocked on reviewed MEC doorbell delivery evidence; C0A task set 5 remains Blocked; C0A Compute 16 is Done for the diagnostic primitive; downstream C1/C2/C3 remain blocked until CPU-verified pass tokens or user-approved fallback/split.

## Wave 19: C0A MEC doorbell source grounding
### Supervisor gates
- Report checks: Task 7 BAR2 index/value audit records a source gap for the gfx1201/TinyGPU assignment-family selector; CP MEC range audit records a match for range `0x00000000..0x000000f8` including BAR2 offset `0x18`; GDC/S2A route audit records a source gap for route readback and range coverage semantics; consolidated decision selects `blocked_source_gap`.
- Quality bar result: `DoorbellSourceReview` passed correctness, maintainability, architectural fit, and simplicity/no over-engineering. The result is a blocker/source-gap checkpoint only, not an implementation lane.
- Review agents: `DoorbellSourceReview` wrote `.superpowers/swarm/reports/c0a-compute-task-7-doorbell-source-grounding-review.md` with 0 Critical, 0 Important, 0 Minor, `ready_for_next_plan: true`, `ready_for_implementation_plan: false`, and implementation dispatch blocked.
- Verification command(s): supervisor validates by reading source-audit, consolidated, and review reports, then runs `git diff --check` after final ledger/supervisor updates.
- Ledger update: C0A Compute 17 is Done for source grounding; C0A task set 4 remains blocked on source-gap resolution before any runtime-path change; C0A task set 5 and downstream C1/C2/C3 remain blocked until CPU-verified pass tokens or user-approved fallback/split.

## Wave 20: C0A23 compute output readback byte-swap diagnostic (T1 + T2 parallel)
### Shared context
- Goal: localize the 16-bit halfword byte-swap + 4-of-8 partial write to the GPU store side (review-gated, diagnostic-only).
- Constraints: shared work boundary `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`. NO kernel-behavior change: no change to kKernelText, kDispatchGlobalSizeX/Y/Z, kDispatchLocalSizeX/Y/Z, kernarg layout, BAR2/GDC-S2A routes, PM4, scheduler, AQL, Linux HIP fallback, allocator, program-counter registers, C1/C2/C3. Only additive instrumentation + self-tests. Executors do not run tests, linters, formatters, package managers, git, compiles, or hardware. Supervisor verifies after the wave.
- Contract: Task 1 adds `classify_compute_readback_anomaly` + `compute_readback_anomaly` log field + `--self-test compute-readback-classifier`. Task 2 adds `--self-test kernel-text-decode` + decodes the embedded kernel per rdna3 tables and fills the `<OPNAME_B32_OR_D16>`/`<ADDTID_OR_BASE>`/`<0..N>` markers. Both touch `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` and `tests/test_native_amdev_transfer_contract.py`; concurrent same-file edits auto-resolve; if a conflict appears, message peer via hub, do not negotiate by serial handoff.
- Reporting: each agent writes a report under `.superpowers/swarm/reports/` (T1: `c0a-compute-task-14a-readback-classifier.md`; T2: `c0a-compute-task-14b-kernel-decode.md`).
- Validation policy: agents record exact supervisor commands; do not run them.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C0A23Classifier | C0A Compute 23 T1 | classify_compute_readback_anomaly + kernel-proof wiring + self-test | C0A22 c263e11 | `.superpowers/swarm/reports/c0a-compute-task-14a-readback-classifier.md` | Done — Wave 1 + Wave 2 reviewed accepted |
| C0A23KernelDecode | C0A Compute 23 T2 | kernel-text-decode self-test + RDNA4 decode + marker fill | C0A22 c263e11 | `.superpowers/swarm/reports/c0a-compute-task-14b-kernel-decode.md` | Done — Wave 1 + Wave 2 reviewed accepted |

### Supervisor gates
- Report checks: each report cites exact source lines and defines the implemented interface.
- Review agents: dispatched after Wave 1 (post-merge), before Wave 2 hardware.
- Verification command(s) supervisor will run: focused pytest for the two new self-tests, then full `tests/test_native_amdev_transfer_contract.py -q` (expect 23 passed), then build.
- Ledger update: C0A Compute 23 stays In Progress until Wave 2 report + review.

## Wave 21: C1 attention/RoPE/KV writer planning
### Shared context
- Goal: unblock and execute C1 task set 6 after Wave 2 by adding the single-layer K/V writer path that later C1 tasks consume.
- Constraints: shared work boundary `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every executor/reviewer stays in this cwd/branch. The frozen C1 Llama contract remains the parity gate: MLX safetensors dir, no tinygrad in producer path, RoPE from config sidecar, S-1 prefix, fp16 K/V shape `(1,8,N,64)`, and token-exact `P == R` over Phase 0 prompts. Do not edit the frozen C0 probe, `docs/adr/*`, frozen `docs/ROADMAP.md` contract text, or the frozen C1 phase contract text.
- Qwen target decision: discovered local candidate `${HOME}/Development/ml/models/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff` is tracked as an additional target. The Llama path must not be generalized in a way that weakens the C1 gate. If Qwen proves incompatible with the Llama C1 ladder, record an explicit unsupported/deferred decision with config evidence and a follow-up task boundary instead of faking parity.
- TDD policy: supervisor observes focused RED tests before production C1 task-set-6 code. OMP executor agents do not run tests, linters, formatters, package managers, hardware commands, or git commands; supervisor verifies after the wave.
- Reports: `.superpowers/swarm/reports/c1-task-6-attention-kv.md`, plus scout evidence under agent outputs if needed.

### Agents
| C1LlamaAttentionScout | C1-6 prep | Llama attention/RoPE/KV writer API and tests | C1-3, C1-5 | `agent://C1LlamaAttentionScout` | Done |
| C1QwenTargetScout | C1/Qwen prep | Qwen3.8-27B local target feasibility | user Qwen scope | `agent://C1QwenTargetScout` | Done |
| C1AttentionRed | C1-6 RED | Focused RED tests and validation-command row | C1-6 prep | `.superpowers/swarm/reports/c1-task-6-attention-kv-red.md` | Done |
| C1AttentionImpl | C1-6 GREEN | `native_r9700/attention.py` layer0 writer and CLI | RED observed | `.superpowers/swarm/reports/c1-task-6-attention-kv.md` | Done |
| C1AttentionReview | C1-6 review | Task-scoped correctness/quality review | GREEN verified | `.superpowers/swarm/reports/c1-task-6-attention-kv-review.md` | Done |

- Report checks: RED report, implementation report, review package, and review report agree on a Llama-only layer0 K/V writer surface; scouts cite local source/model evidence; Qwen is recorded as unsupported/deferred for this C1 Llama ladder rather than faked.
- Quality bar: passed. Correctness covered by RED/GREEN tests, fixture deltas, CLI log, and review; maintainability passed because implementation is one narrow `attention.py` module using existing config/primitives/fixtures; architectural fit passed because producer path has no tinygrad and does not use MLX for producer math; simplicity passed because no target registry, Qwen abstraction, or runtime framework was added.
- Review agents: `C1AttentionReview` approved with 0 Critical, 0 Important, 0 Minor.
- Verification command(s) supervisor ran: RED focused pytest exited 1 with 9 expected failures; focused GREEN `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_attention_kv.py -v` exited 0 with 9 passed; CLI `python -m native_r9700.attention ... --log logs/c1-attention-kv-layer0.log` exited 0 and logged K/V max/mean deltas; combined `tests/native_r9700 -v` exited 0 with 66 passed; full `tests -v` exited 0 with 106 passed, 2 warnings; `git diff --check` printed no output.
- Ledger update: C1-6 Done; C1-7 is unblocked.

## Wave 22: C1 full layer stack prefill
### Shared context
- Goal: implement C1 task set 7 by assembling a narrow Llama 3.2 1B prefix prefill through all 16 layers, producing ordered per-layer K/V arrays for the downstream KV emitter.
- Constraints: shared work boundary `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every executor/reviewer stays in this cwd/branch. Producer path remains tinygrad-free. Use stdlib + numpy + safetensors and existing `native_r9700.config`, `primitives`, and `attention` contracts. Do not implement C2, prompt-cache safetensors emission, decode/sampling, Qwen support, or C++ runtime changes in this wave.
- TDD policy: supervisor observes focused RED tests before production C1 task-set-7 code. OMP executor agents do not run tests, linters, formatters, package managers, hardware commands, or git commands; supervisor verifies after the wave.
- Qwen target decision: Qwen3.8-27B remains recorded as unsupported/deferred for C1 because local evidence shows mlx-vlm VLM, 4-bit affine weights, hybrid linear/full attention, and non-C1 cache schema.
- Reports: `.superpowers/swarm/reports/c1-task-7-full-prefill.md` plus RED/review reports.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C1PrefillRed | C1-7 RED | Focused prefill tests and validation-command row | C1-6 Done | `.superpowers/swarm/reports/c1-task-7-full-prefill-red.md` | Done |
| C1PrefillImpl | C1-7 GREEN | `native_r9700/prefill.py` full 16-layer prompt-0 prefix prefill | RED observed | `.superpowers/swarm/reports/c1-task-7-full-prefill.md` | Done |
| C1PrefillReview | C1-7 review | Task-scoped correctness/quality review | GREEN verified | `.superpowers/swarm/reports/c1-task-7-full-prefill-review.md` | Done |

- Report checks: RED report, implementation report, review package, and review report agree on prompt-0 full-layer K/V scope, explicit longer-prompt handoff to task set 9, no tinygrad/MLX production dependency, and no fake Qwen support.
- Quality bar: passed. Correctness covered by RED/GREEN tests, fixture deltas, CLI log, and review; maintainability passed because implementation is one narrow `prefill.py` module using existing config/primitives/attention contracts; architectural fit passed because it hands task set 8 ordered C1 K/V arrays without safetensors schema leakage; simplicity passed because no generic model runner, target registry, Qwen abstraction, or runtime framework was added.
- Review agents: `C1PrefillReview` approved with 0 Critical, 0 Important, 0 Minor.
- Verification command(s) supervisor ran: RED focused pytest exited 1 with 5 expected missing-module/API failures; focused GREEN `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py -v` exited 0 with 5 passed; CLI `python -m native_r9700.prefill ... --out logs/c1-prefill-prompt0.npz --log logs/c1-prefill-prompt0.log` exited 0 and logged layer0/layer15 deltas; combined `tests/native_r9700 -v` exited 0 with 71 passed; full `tests -v` exited 0 with 111 passed, 2 warnings; `git diff --check` printed no output.
- Ledger update: C1-7 Done; C1-8 is unblocked and In progress.

## Wave 23: C1 KV interchange emitter
### Shared context
- Goal: implement C1 task set 8 by serializing the C1-7 native prefill arrays into an mlx-lm-loadable prompt-cache `.safetensors` file.
- Constraints: shared work boundary `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer`; every executor/reviewer stays in this cwd/branch. Producer/emitter production path remains tinygrad-free and should not need an MLX runtime import; tests may use mlx-lm to round-trip the emitted file. Do not implement C1 task set 9 parity/decode, C2 integration, Qwen support, or C++ runtime changes in this wave.
- ABI from `C1EmitterScout`: safetensors tensor keys are `{i}.0` for K and `{i}.1` for V; metadata keys are `0.{i}=""`, `2.{i}="KVCache"`, global `1.offset=str(N)`, `1.num_layers="16"`, `1.n_kv_heads="8"`, `1.head_dim="64"`; `N` is the S-1 prefix length, not full prompt S.
- TDD policy: supervisor observes focused RED tests before production C1 task-set-8 code. OMP executor agents do not run tests, linters, formatters, package managers, hardware commands, or git commands; supervisor verifies after the wave.
- Reports: `.superpowers/swarm/reports/c1-task-8-kv-emitter.md` plus RED/review reports and `agent://C1EmitterScout`.

### Agents
| Agent | Task row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| C1EmitterScout | C1-8 prep | mlx-lm prompt-cache ABI research | C1-7 API contract | `agent://C1EmitterScout` | Done |
| C1EmitterRed | C1-8 RED | Focused emitter tests and validation-command row | C1-7 Done | `.superpowers/swarm/reports/c1-task-8-kv-emitter-red.md` | Planned |
| C1EmitterImpl | C1-8 GREEN | `native_r9700/kv_cache.py` safetensors emitter and CLI | RED observed | `.superpowers/swarm/reports/c1-task-8-kv-emitter.md` | Planned |
| C1EmitterReview | C1-8 review | Task-scoped correctness/quality review | GREEN verified | `.superpowers/swarm/reports/c1-task-8-kv-emitter-review.md` | Planned |

### Supervisor gates
- Report checks: scout/RED/implementation/review reports agree on exact mlx-lm safetensors tensor keys, metadata keys, S-1 offset semantics, no casting/repair of bad producer arrays, and no fake Qwen support.
- Quality bar: correctness via safetensors header checks and mlx-lm load round-trip; maintainability via one narrow emitter module; architectural fit via existing KV interchange contract; simplicity via no duplicate generic exporter framework.
- Verification command(s) supervisor will run: focused C1-8 RED/GREEN pytest command recorded in `validation-commands.md`, CLI smoke converting a local prefill NPZ into safetensors and loading it, combined `tests/native_r9700 -v`, full `tests -v`, and `git diff --check`.
- Ledger update: C1-8 In progress while RED/GREEN/review run; dependent C1-9 remains blocked until C1-8 review/verification passes.
```
