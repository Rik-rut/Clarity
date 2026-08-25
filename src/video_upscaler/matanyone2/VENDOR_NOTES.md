# Vendored MatAnyone2 — provenance, license, and dependency audit

## Upstream source

- Repo: https://github.com/pq-yang/MatAnyone2
- Cloned: 2026-08-24
- **Important:** upstream tag `v1.0.0` (2025-12-15) contains only `README.md`
  and `assets/` — it does **not** contain the `matanyone2` package. The package
  exists only on `main`. This vendored copy is therefore pinned to
  `main` HEAD `0079197` (`Update EVAL.md`, 2026-07-27), which is the only
  upstream commit shipping `matanyone2/inference/inference_core.py`,
  `matanyone2/utils/get_default_model.py`, the model subnetworks, and
  `LICENSE.txt`. `VENDOR_DIR = src/video_upscaler/matanyone2/vendor`.
  Required downstream surface was verified to exist at this commit.

## License

The package ships with upstream's `LICENSE.txt` (included verbatim at
`vendor/LICENSE.txt`) and any reuse must honor it.

## How the vendored tree imports

The vendored package is nested at `video_upscaler.matanyone2.vendor.matanyone2`,
so upstream's absolute imports (`from matanyone2.X import Y`) were rewritten to
relative imports in place — each module resolved relative to its own depth in
the vendored package. No other behavior-changing edits except the three
flattenings below.

## Full audited import list (third-party)

Collected from every `.py` file under `vendor/matanyone2/`:

### (a) Already-core deps (no extra entry needed)

| import | usage |
|---|---|
| `torch` | everywhere (core) |
| `torchvision` | `utils/inference_utils.py` video decode (`torchvision.io.read_video`) |
| `numpy` | arrays throughout (via torch/cv2) |
| `opencv` (`cv2`) | frame/mask IO (core: opencv-python-headless) |
| `imageio` | `inference_core.py` `mimwrite` (core) |
| `torch.hub` (`download_url_to_file`, `get_dir`) | `utils/download_util.py` — download helper, downloads through stdlib tooling |

### (b) Core dependencies (audited, inference-path)

These live in core `dependencies` (not extras): an earlier placement in the
optional `matanyone2` extra made them vulnerable to being pruned by exact
`uv sync`/`uv run` runs that omit `--all-extras`, which broke matting jobs
with `ModuleNotFoundError: omegaconf` after any environment refresh.

| import | why |
|---|---|
| `omegaconf` | `DictConfig`/`OmegaConf` across `inference_core.py`, `memory_manager.py`, `args_utils.py`, `aux_modules.py`, `big_modules.py`, `matanyone2.py`, `object_summarizer.py`, `object_transformer.py`, `get_default_model.py` |
| `tqdm` | `inference_core.py` (progress), `download_util.py` |

The brief's *starter* extra (`einops`, `easydict`) does **not** match this
tree: `einops`, `easydict`, `av`, and `kornia` are imported **nowhere** under
`vendor/matanyone2/` at v1.0.0-source (`main` `0079197`). They were dropped.

### (c) Excluded upstream deps (not added to the extra)

| import | exclusion reason |
|---|---|
| `hydra` | config composition in `get_default_model.py` — flattened, see below |
| `huggingface_hub` | `PyTorchModelHubMixin` base class in `model/matanyone2.py` — flattened, see below |
| `PIL` | mask open in `inference_core.py` — flattened to `cv2.imread`, see below |
| `requests` | `download_util.py` only — weights-download helper, not on the matting inference path |
| `typer` | `cli.py` only — demo CLI |
| `gradio`, `PySide6`, `tensorboard`, `thinplate`, `cchardet`, `netifaces`, `pyqtdarktheme`, `xlsxwriter`, `pycocotools`, `gdown`, `gitpython`, `hickle`, `imgviz`, `einops`, `easydict` | upstream eval/training/demo extras absent from the shipped `.py` tree |

### Flattened edits inside `vendor/` (minimal, documented)

1. **`utils/get_default_model.py`** — removed `hydra` (`initialize`/`compose`).
   Replaced with a small `OmegaConf` loader in `_load_default_cfg()` that merges
   `config/eval_matanyone_config.yaml` with `config/model/base.yaml` under the top
   `model` key (exactly what the yaml's `defaults: [model: base]` did), and pops
   the hydra-only `defaults`/`hydra` keys. **No hydra dependency.**
2. **`model/matanyone2.py`** — removed `PyTorchModelHubMixin` inheritance and the
   `from huggingface_hub import ...` line; the class is now plain `nn.Module`.
   `from_pretrained`/`save_pretrained` (HF-hub only) are unavailable; load local
   checkpoints via `torch.load` + `load_weights` as `get_matanyone2_model` does.
3. **`inference/inference_core.py`** — two edits:
   - `from PIL import Image` removed; `mask = np.array(Image.open(mask_path).convert("L"))`
     became `mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)` (identical
     semantics, no Pillow dependency).
   - `network = MatAnyone2.from_pretrained(network)` (string-model-id path)
     replaced with an explicit `TypeError` pointing at `get_matanyone2_model`,
     since `from_pretrained` no longer exists after edit 2.

## Runtime note (not an import-time dependency)

`inference_core.process_video` reads video via `utils/inference_utils.read_frame_from_videos`
→ `torchvision.io.read_video`; on systems where `torchvision` lacks its bundled
decoder this needs `av` at **runtime**. `av` is intentionally not in the
inference-path import audit (it is not imported by any vendored module), and
Clarity's own ffmpeg pipe processor never calls that code path; if a later
task hits `ModuleNotFoundError: av`, add it to core dependencies then.

<!--
ponytail ceiling (from the vendoring task): if upstream ever re-tags with a
tag containing the package code, re-pin MAIN_COMMIT above to that tag and
re-run the audit; the extra list should be re-checked, not assumed stable.
-->
