# Task 1b — Validation & discovery report

**Agent:** ValidationDiscovery · **Status:** Needs review · **Phase:** 0 (Wave 2)

## 1. tinygrad baseline — importable and LLM module present at pinned repo path

- `python3 -c "import tinygrad; print(tinygrad.__file__)"` →
  `${HOME}/Development/ml/tools/tinygrad/tinygrad/__init__.py`. tinygrad is installed
  from a **local checkout** at `${HOME}/Development/ml/tools/tinygrad/` (not site-packages).
- `import tinygrad.llm` → OK (`tinygrad/llm/` is a package). `tinygrad/llm/` exists at the pinned
  repo path with `__init__.py`, `__main__.py`, `cli.py`, `gguf.py`, `model.py`, `serve.py` —
  matches `docs/pinned-upstream-interfaces.md` §1 ("Package moved from `tinygrad/llm.py` →
  `tinygrad/llm/`").
- Module entrypoint `python3 -m tinygrad.llm --help` works (argparse help renders); the module
  dispatches to `cli.main()` via `__main__.py`.

## 2. `JITBEAM=2 DEV=AMD python3 -m tinygrad.llm` — can it start on this box? YES (card present via USB4)

**Key finding: the AMD eGPU IS present and resolves on this box — via USB4/TinyGPU, not Vulkan/Metal.**

- `Device["AMD"]` opens successfully. With `DEBUG=6`:
  ```
  loading libusb from /opt/homebrew/lib/libusb-1.0.dylib
  am usb4: AM_GFX initialized
  am usb4: AM_SDMA initialized
  am usb4: boot done
  AMDDevice: opening 0 with target (12, 0, 1) arch gfx1201
  opened device AMD from pid:...
  ```
  Selected interface: `PCIIface` (over USB4), `is_am=True`, 8 SDMA engines, device_id 0, target
  arch **gfx1201** (RDNA4 / gfx12-class). This matches the Radeon AI PRO R9700 and the
  `ops_amd.py` supported-arch assert `(9,4,2)|(9,5,0)|gfx11|gfx12`.
- The `~/Library/Caches/tinygrad/downloads/` cache holds the TinyGPU USB4 toolchain
  (`TinyGPU_…/TinyGPU_c0d024f9ff0e1dc8fdf217f255da7101d91e8323.zip`) and `fw/` firmware blobs —
  consistent with §4 ("macOS AMD transport = USB/DMA (TinyGPU)"; `AMDDevice` selects `USBIface`
  on macOS). Here it boots through the PCIIface-over-USB4 path.

### End-to-end baseline smoke (reaches a defined point)

Ran the real baseline (no server, no long benchmark) against the locally-cached Llama 3.2 1B GGUF:

```
JITBEAM=2 DEV=AMD python3 -m tinygrad.llm \
  --model ${HOME}/Library/Caches/tinygrad/downloads/3cdb17618469285f97f176c434543c9c \
  --max_context 1024
→ using model "Llama 3.2 1B Instruct" with 1,021,800,576 bytes and 1,498,482,688 params
→ >>> (interactive prompt reached)
```

So the box reaches the "using model" line and the interaction loop: the model loads ON the AMD
device (the `TransformerBlock._init_state` `cache_kv` allocation `Tensor.empty([2,B,8,max_context,128])`
on `device=x.device` succeeds). Cached GGUF metadata confirms the target: `general.name = Llama 3.2
1B Instruct`, `llama.attention.head_count=32`, `head_count_kv=8` (**n_kv_heads=8**), `embedding_length=2048`.

### What happens with `DEV=AMD` when there is NO card (documented code path)

On a cardless macOS box, `AMDDevice.__init__` → `_select_iface()` (`runtime/support/hcq.py:493`)
runs `select_first_inited()` (`helpers.py:143`) over
`[KFDIface, PCIIface, USBIface, …]`. With no AMD device present, every interface init raises and
`select_first_inited` raises the single exception (or an `ExceptionGroup("No interface for AMD:0 is
available", excs)` when >1 fail). The macOS-visible leaf error is from `USBIface.__init__`
(`ops_amd.py:915`):

```
RuntimeError: AMD:0 does not exist (N devices available)
```

where `N = len(USB3.list_devices(0xADD1, 0x0001))` (0 with no card). **On this box it does NOT
fail** — the USB4 card is attached, so the harness can use `DEV=AMD` here directly (no separate
eGPU host needed for device bring-up; the eGPU at `192.168.2.80` remains a decode/concurrency
concern for Task 3).

## 3. Exact unit-test command (confirmed runnable)

```
python3 -m pytest tests/test_exporter.py -v
```

- **cwd:** `.worktrees/tinygrad-kv-worker-phase0` (repo root) — pytest `rootdir` resolves here.
- **pytest:** 9.0.3 (python 3.12.8 via pyenv). Run from the worktree root, no `PYTHONPATH` needed
  (test imports `tinygrad_kv_worker.exporter`, which lives in the repo root package).
- **Result:** 8 passed in 1.46s (verified). No GPU / no mlx eval in the exporter core path.

## 4. mlx-lm / cache reality (feeds pinned-interface update)

- **mlx_lm 0.31.3** (installed), `mlx` core present (`mlx` exposes no `__version__`; n/a).
- `mlx_lm/models/cache.py` (signatures verified by introspection):
  - `save_prompt_cache(file_name: str, cache: List[Any], metadata: Dict[str, str] = {})`
  - `load_prompt_cache(file_name, return_metadata=False)`
  - `KVCache.from_state(state, meta_state)`
  - `_BaseCache.meta_state` setter **raises `ValueError`** for any truthy value
    (`"This cache has no meta_state but a meta_state was set."`); standard `KVCache` does **not**
    override `meta_state` (inherits the raising setter).
  - `KVCache.state` setter (overrides base): `self.keys, self.values = v; self.offset = self.keys.shape[2]`
    → **`offset` is reconstructed from `keys.shape[2]`**. This is the mechanism by which
    `offset == S` survives the round-trip even though `meta_state` must stay `""`.
- **Empirical round-trip** (16-block fake, `B=1, n_kv_heads=8, head_dim=128, S=42`):
  - `export_prompt_cache(...)` → `load_prompt_cache(path, return_metadata=True)`:
    ```
    global metadata → {'offset': '42', 'n_kv_heads': '8', 'head_dim': '128', 'num_layers': '16'}
    layer i: class=KVCache  offset=42  keys.shape=(1, 8, 42, 128)  keys.dtype=fp16  meta_state=''
    ```
  - **Confirmed:** `layer.offset == S` and `meta_state == ''` (NOT `str(S)`); global metadata
    carries `{'offset': str(S), …}`. This validates the contract's REVISED-after-Wave-1 assertion:
    assert `layer.offset == S` per layer + `{'offset': str(S)}` in `load_prompt_cache(path,
    return_metadata=True)[1]`.
  - **Confirmed:** `KVCache.from_state((k, v), str(S))` raises
    `ValueError: This cache has no meta_state but a meta_state was set.` for `S > 0` — per-layer
    `meta_state=str(S)` is **not loadable** on 0.31.3.

## 5. Corrections for `docs/pinned-upstream-interfaces.md` §2 (Phase-0 needs)

1. **`meta_state → str(offset)` is WRONG for the standard `KVCache`.** The installed 0.31.3 and
   current upstream `main` define standard `KVCache` WITHOUT a `meta_state` override; it inherits
   `_BaseCache.meta_state`, whose setter raises `ValueError` for any truthy value. Only
   `QuantizedKVCache` / `RotatingKVCache` / `ChunkedKVCache` define their own (accepting)
   `meta_state`. **Per-layer `meta_state` is `""` for the standard cache.**
2. **`offset` comes from `state.keys.shape[2]`, not `meta_state`.** `KVCache.state` setter (and
   `from_state`) reconstruct `offset = keys.shape[2]`. So `offset == S == prompt length` holds per
   layer with `meta_state=""`. Assert **`offset == S`**, not `meta_state == str(S)`.
3. **`S` is carried in safetensors global metadata**, not per-layer `meta_state`:
   `save_prompt_cache(file, cache, metadata={'offset': str(S), 'n_kv_heads':…, 'head_dim':…,
   'num_layers':…})` → read back as `load_prompt_cache(path, return_metadata=True)[1]`.
4. **`save_prompt_cache` accepts a `metadata` kwarg** (missing from the pinned §2 signature line
   `save_prompt_cache(file, cache)`): full signature is
   `save_prompt_cache(file_name, cache, metadata={})`. `load_prompt_cache` accepts
   `return_metadata=False`.
5. **Switch-back note:** if upstream restores a `meta_state` override on the standard `KVCache`
   (as already present on the quantized variants), per-layer metadata can switch to `str(S)` — the
   global-metadata copy makes this a one-line change. Keep §2 green on the `offset==S` contract.

## Acceptance checklist

- [x] Report present: baseline facts (§1–§2), DEV=AMD behavior with/without card (§2)
- [x] Exact unit-test command recorded + verified running (§3)
- [x] Pinned-interface correction list for §2 (§4–§5)
- [x] mlx_lm version + `KVCache`/`save_prompt_cache`/`load_prompt_cache` reality captured (§4)
