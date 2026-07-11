# ImageInfoExtractor — Agent Guide

## Project Overview

GPU-accelerated pipeline that runs 15+ vision models over image datasets and stores results in a Polars DataFrame (Parquet+JSON). The orchestrator (`ImageInfoManager`) scans directories, maintains an image registry, and runs configurable YAML tool pipelines with incremental/force update semantics.

## Architecture

### Core Loop (`ImageInfoManager.infoUpdate`)

1. Load `ImageInfo.parquet` (or `.json`) into `self.imageInfoDF` (Polars DataFrame)
2. For each tool in the YAML pipeline:
   - Determine which rows need processing (missing/null fields or `forceUpdate`)
   - If tool supports batch inference → create `BatchInferenceDataset` + `DataLoader`, call `update_batch`
   - If single → iterate with `getUpdateDict`
   - If multi-GPU → split indices, use `multiprocessing.Pool` with `processFunc`
   - Accumulate results in `UpdateBuffer`, flush+save periodically
3. Save final DataFrame

### Adding a New Tool

Create a class in `ImageInfoExtractor.py` with:

```python
class MyTool:
    def __init__(self, topDir, device='cuda', **kwargs):
        # Load model, store self.transform for batch dataloader
        ...

    # For single-image inference (required)
    def getUpdateDict(self, imageInfoDF, idx, topDir):
        img = pil_loader(os.path.join(topDir, imageInfoDF['IMG'][idx]))
        return {'MY_FIELD': result}  # must match fieldSet()

    # For batch inference (optional)
    def update_batch(self, imgs):
        return [{'MY_FIELD': ...} for ...]

    @staticmethod
    def supportBatchInference():
        return False  # or True

    @staticmethod
    def fieldSet():
        return {'MY_FIELD'}  # fields this tool populates

    @staticmethod
    def updateFilter(imageInfoDF):
        # Optional custom filter logic instead of null-check
        mask = (pl.col('W') * pl.col('H') > 384 * 384)
        return imageInfoDF.filter(mask).get_column('IDX')
```

- Tool classes live directly in `ImageInfoExtector.py` (not as separate modules), except the standalone `Tools/ImageSizeInfoCorrectTool.py`.
- Each model has its own inference subpackage under `DL_INFERENCE_SUBPACKAGES/` with an `inference.py` exposing a `Predictor` class.
- New subpackages should follow the existing pattern: `Predictor(weightsDir, device)`, `predict(img) -> dict`, `predict_batch(imgs) -> list[dict]`, `transform` attribute.

### Naming Conventions

- **Files:** PascalCase for modules (`MultiDatasetExtractor.py`, `ImageClustering.py`)
- **Classes:** PascalCase (`ImageInfoManager`, `ImageQuailityTool`)
- **Methods:** camelCase (`getUpdateDict`, `updateImages`, `saveImageInfoList`)
- **Static methods:** lowercase\_with\_underscores (`fieldSet`, `supportBatchInference`)
- **DataFrame columns:** UPPER\_SNAKE\_CASE (`Q512`, `A_EAT`, `DBRU_TAG`, `IMG_EMBD`, `HQ_CAP`)
- **Schema:** defined as `CURRENT_SCHEMA` dict at top of `ImageInfoExtractor.py`

### DataFrame Operations

- Always use Polars (imported as `pl`). Never use pandas.
- `IDX` column is the row index (managed by `with_row_index(name="IDX")`).
- Use `batch_update(df, idx_series, updates_list)` for vectorised writes.
- Use `find_missing_or_null_indices(df, fields)` to find unprocessed rows.
- `UpdateBuffer` wraps batch updates with periodic auto-save.

### YAML Pipeline Config

Loaded via `yaml.safe_load`. Each tool dict is merged with defaults:
```python
{'forceUpdate': False, 'multiGPUs': None, 'excludeDirs': None, 'includeDirs': None,
 'args': {}, 'batchsize': 1, 'num_workers': 4}
```

Tool class is resolved by name via `getattr(sys.modules[__name__], toolClass)`.

### Inference Subpackage Pattern

Each subpackage under one of the `DL_INFERENCE_SUBPACKAGES` dirs should:
- Expose a `Predictor` class
- Accept `weightsDir` and `device` in `__init__`
- Provide a `transform` attribute (Compose of torchvision transforms) usable by `BatchInferenceDataset`
- Implement `predict(img: PIL.Image) -> dict` and optionally `predict_batch(imgs: list[PIL.Image]) -> list[dict]`
- Auto-download weights on first use (usually via `huggingface_hub.snapshot_download`)

### Incremental Processing

Each tool decides which images need processing:
1. Default: find rows where any field from `fieldSet()` is null or the column is missing
2. Custom: override `updateFilter()` to return matching `IDX` values
3. Force: `forceUpdate: true` processes all rows regardless

### Testing

No formal test suite exists. Testing is done via:
- Runner scripts in `TaskPipelines/`
- Sample datasets (`TestResultSample/`, `ClusterResultSample/`)
- Visual verification of output artifacts
- When adding features, create or update a pipeline YAML and run it on sample data

### Image Formats

Supported: `.jpg`, `.webp`, `.png`, `.heic` (HEIF support via `pillow_heif`).

### Common Patterns

- `pil_loader(path)` → opens and converts to RGB
- `register_heif_opener()` called at module import
- Model weights in `./DLToolWeights/` (gitignored)
- `topDir` is the dataset root; `topTopDir` is an optional higher-level root for relative path filtering
- Tools that modify images on disk (SR, matting) save alongside originals with backup copies
