# Facial Attribute Analysis on WFLW

This package provides a **modular, reproducible implementation** of the complete pipeline used in the accompanying Jupyter notebook **`Facial_Attr_WFLW.ipynb`**.  
It covers landmark alignment, data preparation, multi‑task learning and evaluation for facial attribute recognition on the **[WFLW dataset](https://wywu.github.io/projects/LAB/WFLW.html)**.

## 📂 Directory layout

```
facial_attr_wflw_pkg/
├── facial_attr_wflw/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   └── utils.py
├── Facial_Attr_WFLW.ipynb
├── README.md
└── LICENSE
```

> **Note**: The dataset folders `WFLW_images/`, `WFLW_annotations/` and the CSV splits (`train_data.csv`, `val_data.csv`, `test_data.csv`) must sit **next to** this package root.

## 🚀 Quickstart

```bash
# Clone & install in editable mode
git clone <your‑fork‑url>
cd facial_attr_wflw_pkg
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

```python
from facial_attr_wflw import set_seed
from facial_attr_wflw.train import main as train_main

set_seed(1337)
train_main(
    images_dir='WFLW_images',
    ann_file='WFLW_annotations/list_98pt_rect_attr_train_test.txt',
    csv_train='train_data.csv',
    csv_val='val_data.csv'
)
```

The full experiment—including EDA visuals, intermediate metrics and qualitative results—is documented in **`Facial_Attr_WFLW.ipynb`**.

## 🔧 Features

* **Landmark Alignment:** 98‑point landmark parsing and Procrustes alignment to a learned mean shape  
* **Flexible DataLoader:** CSV‑driven split management, balanced sampling, Albumentations transformations  
* **Multi‑task CNN & CLIP tuning:** Simultaneous prediction of five WFLW attributes with focal/weighted losses  
* **Comprehensive Metrics:** Accuracy, F1, ROC‑AUC, per‑attribute confusion matrices  
* **Reproducible Training:** Deterministic seed control, automatic checkpointing, TensorBoard logs  
* **Extensible:** Modular code structure makes it easy to swap backbone, add new heads or schedule hyper‑parameter sweeps.

## 🖥️ Requirements

* Python ≥3.9
* PyTorch ≥2.0
* torchvision
* transformers
* scikit‑learn
* pandas, numpy
* opencv‑python
* albumentations
* pillow

Install them via:

```bash
pip install torch torchvision transformers scikit‑learn pandas numpy opencv‑python albumentations pillow
```

## 📝 License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.

## 🔖 Citation

If you build on this codebase, please cite it:

```bibtex
@misc{facial_attr_wflw,  
  author = {Nabiee, Shima},  
  title  = {Facial Attribute Analysis on WFLW (Python Package)},  
  year   = {2025},  
  url    = {https://github.com/ShimaN19/facial_attr_wflw}  
}
```
