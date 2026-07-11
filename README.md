# ImageInfoExtractor

A comprehensive GPU-accelerated pipeline for extracting rich metadata, annotations, and embeddings from large-scale image datasets. Integrates 15+ deep learning models for quality assessment, aesthetic scoring, captioning, OCR, object detection, pose estimation, tagging, matting, super-resolution, and more.

## Features

- **Image Quality Assessment** — HyperIQA scoring at multiple resolutions (Q512)
- **Aesthetic Scoring** — EAT (DAT transformer) aesthetic score 1–10
- **Caption Generation** — BLIP, BLIP2, MiniCPM-Llama3-V-2.5, OpenAI-compatible API
- **Danbooru Tagging** — WDVITaggerV3
- **OCR** — EasyOCR text extraction
- **Object Detection** — YOLO (custom weights)
- **Pose Estimation** — RTMPose via rtmlib
- **Watermark Detection** — Custom CNN classifier
- **Image Embeddings** — DINOv3 (ViT-L/16)
- **Image Matting** — BiRefNet (HR)
- **Super-Resolution** — Real-ESRGAN (photos), Real-CUGAN (anime)
- **Smart Cropping** — Attention-based crop centre
- **Clustering & Analysis** — FAISS GPU KMeans for embeddings & poses with t-SNE/UMAP visualisation
- **Dataset Creation** — Filter/sample subsets by quality, aesthetics, size, etc.
- **Incremental Processing** — Skips already-processed images; auto-save every hour
- **Multi-GPU** — Distribute tasks across devices
- **Batch & Single Inference** — Each tool declares batch support; dataloader used automatically

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)
- See `requirements.txt` for core dependencies; additional packages per subpackage (faiss-gpu, scikit-learn, webdataset, openai, etc.)

## Installation

```bash
git clone <repo-url>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Model weights auto-download to `DLToolWeights/` on first inference via Hugging Face `snapshot_download`.

## Configuration

Extraction pipelines are defined in YAML. See `MyExtractionBatchPipeline.yaml.example`:

```yaml
Tools:
  - toolClass: ImageQuailityTool
    forceUpdate: false
  - toolClass: ImageEATAestheticTool
    forceUpdate: true
    batchsize: 32
    num_workers: 16
    args:
      device: cuda:0
```

**Per-tool options:**

| Option        | Description                                    |
|---------------|------------------------------------------------|
| `forceUpdate` | Re-process images even if field already exists |
| `batchsize`   | Batch size for tools supporting batch mode     |
| `num_workers` | DataLoader workers                             |
| `args`        | Dict of constructor kwargs (device, model, …)  |
| `multiGPUs`   | List of GPU device indices for parallelisation |
| `excludeDirs` | Skip directories matching these paths          |
| `includeDirs` | Only process directories matching these paths  |

## Usage

### Python API

```python
from ImageInfoExtractor import ImageInfoManager

manager = ImageInfoManager(
    topDir="/path/to/dataset",
    toolConfigYAML="MyExtractionBatchPipeline.yaml"
)
manager.updateImages(filteredDirList=['raw_before_sr'])
manager.infoUpdate()
manager.saveImageInfoList()
```

### CLI

```bash
python ImageInfoExtractor.py --dsDir "/path/to/dataset" --toolConfig "MyExtractionPipeline.yaml"
```

### Multi-Dataset Recursive

```python
from MultiDatasetExtractor import MultiDatasetExtractor
ext = MultiDatasetExtractor("/datasets/root")
ext.scanDir(printScanResult=True)
ext.runExtractor("TaskPipelines/MyExtractionBatchPipeline.yaml")
```

### Dataset Filtering & Creation

```python
from DatasetCreator import ImageDsCreator
creator = ImageDsCreator(topDir="/datasets/root", outputDir="./subset", filteredDirList=[])
creator.generateCandidateList(criteria=lambda x: x['Q512'] > 60, wantNum=5000)
creator.copyImageSet(convertToWebP=True)
```

### Clustering

```bash
python ImageClustering.py       # FAISS GPU KMeans on DINOv3 embeddings
python ClusteringTree.py        # Recursive elbow-method clustering
python PoseClustering.py        # Pose embedding clustering
python PoseAnalyser.py          # t-SNE/UMAP visualisation
```

### Statistics & Utilities

```bash
python ImageInfoStatistics.py --jsonPath ImageInfo.json   # Histogram/CDF plots
python CleanImageType.py                                   # Fix file extensions
python DeleteDuplicate.py                                  # Remove duplicates
python GenerateUpdateTar.py                                # Create update patches
```

## Tool Classes

| Tool Class                 | Fields Written              | Batch | Description                          |
|----------------------------|-----------------------------|-------|--------------------------------------|
| `ImageSizeInfoCorrectTool` | `W, H`                      | No    | Verify/correct image dimensions      |
| `ImageQuailityTool`        | `Q512, W, H`                | Yes   | HyperIQA quality score               |
| `WatermarkDetectTool`      | `HAS_WATERMARK`             | Yes   | Watermark probability                |
| `SmartCropTool`            | `A_CENTER`                  | No    | Attention-based crop centre          |
| `ImageOCRTool`             | `TXT, W, H`                 | No    | EasyOCR text extraction              |
| `JpegQuailityTool`         | `QF, W, H`                  | No    | FBCNN JPEG quality factor            |
| `ImageEATAestheticTool`    | `A_EAT, W, H`               | Yes   | EAT aesthetic score                  |
| `ImageEmbeddingTool`       | `IMG_EMBD, W, H`            | No    | DINOv3 image embeddings (1024-d)     |
| `ImageMattingTool`         | `HAS_MATTING_MASK, W, H`    | No    | BiRefNet matting mask (saved to disk)|
| `ImageSRTool`              | `W, H`                      | No    | Real-ESRGAN / Real-CUGAN upscaling   |
| `ImagePoseEstimateTool`    | `POSE_KPTS`                 | No    | RTMPose keypoint estimation          |
| `ImageObjectDetectTool`    | `OBJS`                      | Yes   | YOLO object detection                |
| `DeepDanbooruTagTool`      | `DBRU_TAG`                  | Yes   | WDVITaggerV3 Danbooru tagging        |
| `ImageHQCaptionTool`       | `HQ_CAP`                    | No    | MiniCPM-Llama3-V-2.5 captioning      |
| `ImageCaptionTool`         | `CAP`                       | Yes   | BLIP / BLIP2 captioning              |

## Data Schema (Polars DataFrame)

| Column         | Type                         | Description                       |
|----------------|------------------------------|-----------------------------------|
| `IMG`          | `String`                     | Relative image path               |
| `W`            | `Int32`                      | Width                             |
| `H`            | `Int32`                      | Height                            |
| `Q512`         | `Float32`                    | HyperIQA quality score            |
| `CAP`          | `List(String)`               | BLIP/BLIP2 captions               |
| `A`            | `Float32`                    | Legacy aesthetic score            |
| `A_EAT`        | `Float32`                    | EAT aesthetic score               |
| `HQ_CAP`       | `List(String)`               | High-quality captions (MiniCPM)   |
| `A_CENTER`     | `List(Float32)`              | Smart-crop centre (x, y)          |
| `DBRU_TAG`     | `String`                     | Danbooru tags                     |
| `POSE_KPTS`    | `List(Struct)`               | Pose keypoints with bounding box  |
| `HAS_WATERMARK`| `Float32`                    | Watermark probability             |
| `IMG_EMBD`     | `List(Float32)`              | DINOv3 image embedding vector     |

Data is persisted as `ImageInfo.json` + `ImageInfo.parquet` alongside image directories.

## Project Structure

```
ImageInfoExtractor.py              # Core pipeline orchestrator + all Tool classes
ImageInfoExtractor/                # Package (DLToolWeights symlink)
DL_INFERENCE_SUBPACKAGES/          # Inference wrappers (one per model)
  ├── BLIPInference/
  ├── BLIP2Inference/
  ├── MiniCPMLlama3V25Inference/
  ├── hpyerIQAInference/
  ├── EATInference/
  ├── DINOv3Inference/
  ├── BiRefNetInference/
  ├── RealESRGANInference/
  ├── RealCUGANInference/
  ├── RTMPoseInference/
  ├── YoloInference/
  ├── OCRInference/
  ├── WDVitTaggerV3/
  ├── WatermarkDetectionInference/
  ├── SmartCropInference/
  └── FBCNNInference/
Tools/                             # Abstract base classes
TaskPipelines/                     # YAML pipeline configs + runner scripts
DLToolWeights/                     # Model weights (auto-downloaded, gitignored)
MultiDatasetExtractor.py           # Recursive multi-dataset processor
DatasetCreator.py / ImageSelect.py # Dataset filtering & webdataset export
ImageClustering.py / ClusteringTree.py  # FAISS GPU clustering
PoseClustering.py / PoseAnalyser.py     # Pose analysis & visualisation
OpenAPICaption.py                  # Async captioning via OpenAI-compatible API
ImageInfoStatistics.py             # Quality score histogram/CDF
CleanImageType.py / DeleteDuplicate.py # Dataset cleaning utilities
GenerateUpdateTar.py               # Update patch creation
```

## License

GNU General Public License v3.0
