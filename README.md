# Mars — SegMunich Dataset Loader

A clean PyTorch `Dataset` implementation for the [SegMunich](https://huggingface.co/datasets/GFM-Bench/SegMunich) dataset, part of the [GFM-Bench](https://github.com/uiuctml/GFM-Bench) benchmark for geospatial foundation models.

---

## Dataset Overview

SegMunich is a semantic segmentation dataset based on **Sentinel-2 satellite imagery** of Munich's urban landscape over a span of three years.

| Property | Value |
|---|---|
| Sensor | Sentinel-2 |
| Bands | 10 (B01, B02, B03, B04, B05, B06, B07, B8A, B11, B12) |
| Image size | 128 × 128 px |
| Spatial resolution | 10 m |
| Number of classes | 13 |
| Train samples | 3,000 |
| Val samples | 403 |
| Test samples | 403 |

---

## Sample Visualisation

All 10 Sentinel-2 bands alongside the RGB composite and segmentation label for a single training sample:

![SegMunich Sample](notebooks/segmunich_sample0.png)

---

## Project Structure

```
Mars/src/
├── utils/
│   ├── segmunich_dataset.py      # PyTorch Dataset class
│   └── segmunich_config.json     # Band stats, label remap, dataset metadata
├── notebooks/
│   ├── test_segmunich.py         # Visualisation and testing script
│   └── segmunich_sample0.png     # Sample visualisation
└── SegMunich/                    # Raw dataset (not tracked by git)
    ├── metadata.csv
    ├── train/
    ├── val/
    └── test/
```

---

## Installation

```bash
pip install torch numpy pandas tifffile matplotlib
```

---

## Usage

```python
from pathlib import Path
from utils.segmunich_dataset import SegMunich_Dataset

dataset = SegMunich_Dataset(
    data_root=Path("/path/to/Mars/src"),
    split="train",                          # "train" | "val" | "test"
    config_path=Path("utils/segmunich_config.json"),
    apply_remap=True,
)

sample = dataset[0]
print(sample["optical"].shape)          # (10, 128, 128)  — all 10 bands
print(sample["rgb"].shape)              # (3,  128, 128)  — B04, B03, B02
print(sample["nir"].shape)              # (128, 128)      — B8A
print(sample["swir"].shape)             # (2,  128, 128)  — B11, B12
print(sample["veg_red_edge"].shape)     # (3,  128, 128)  — B05, B06, B07
print(sample["coastal_aerosol"].shape)  # (128, 128)      — B01
print(sample["label"].shape)            # (128, 128)      — segmentation mask
```

### With DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
batch  = next(iter(loader))
print(batch["optical"].shape)   # (8, 10, 128, 128)
print(batch["label"].shape)     # (8, 128, 128)
```

---

## Band Reference

| Index | Band | Wavelength (nm) | Description |
|-------|------|-----------------|-------------|
| 0 | B01 | 442.7 | Coastal aerosol |
| 1 | B02 | 492.4 | Blue |
| 2 | B03 | 559.8 | Green |
| 3 | B04 | 664.6 | Red |
| 4 | B05 | 704.1 | Vegetation red edge |
| 5 | B06 | 740.5 | Vegetation red edge |
| 6 | B07 | 782.8 | Vegetation red edge |
| 7 | B8A | 864.7 | Narrow NIR |
| 8 | B11 | 1613.7 | SWIR |
| 9 | B12 | 2202.4 | SWIR |

---

## Label Remap

Raw label values in the `.tif` masks follow the ATKIS classification scheme and are remapped to contiguous indices 0–12 for training. See `segmunich_config.json` for the full mapping.

---

## Citation

```bibtex
@article{hong2024spectralgpt,
  title={SpectralGPT: Spectral remote sensing foundation model},
  author={Hong, Danfeng and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```
