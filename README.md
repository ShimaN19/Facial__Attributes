# Facial Attribute Recognition on WFLW

This project presents an in-house deep learning pipeline for multi-label facial attribute recognition using the WFLW dataset. It investigates the interplay of facial landmarks and soft biometric features through statistical analysis and model-driven attribute learning. The approach leverages customized augmentation strategies, correlation-aware preprocessing, and hybrid model evaluation to ensure robustness and real-world generalization.

---

## 🔬 Overview

Facial attributes—such as smiling, wearing glasses, or facial hair—encode high-level semantic cues essential for downstream tasks like re-identification, emotion analysis, and bias auditing. Our work combines fine-grained facial landmark annotations with curated attribute labels to train deep attribute classifiers and explore latent attribute co-dependencies.

---

## 📁 Project Structure

├── Facial_Attr_WFLW.ipynb # Core training + evaluation notebook ├── train_data.csv # Training metadata split ├── val_data.csv # Validation metadata split ├── test_data.csv # Test metadata split ├── best_att_no_aug_new.h5 # Model weights w/o data augmentation ├── best_att_with_aug_new.h5 # Model weights w/ augmentation ├── best_deep_face.h5 # Alternative baseline model ├── Data_correlation_matrix.png # Attribute correlation visualization ├── WFLW_images/ # Raw facial images (WFLW subset) └── WFLW_annotations/ # Corresponding attribute + landmark labels


---

## 📊 Key Features

- **Multi-label Attribute Classification**  
  Predicts 10+ facial attributes simultaneously via shared CNN backbone with sigmoid activation heads.

- **Augmentation-Aware Model Comparison**  
  Benchmarks the impact of augmentation pipelines (flip, brightness, occlusion) on learning invariant facial features.

- **Correlation Matrix Visualization**  
  Statistical exploration of attribute co-occurrence to guide architectural and loss function tuning.

- **Custom Data Split**  
  Manual curation of balanced train/val/test splits for reproducibility and low-variance performance tracking.

---

## 🧠 Model Architecture

- Input: 128x128 normalized facial crops  
- Backbone: Custom CNN (alternatively testable with pretrained models)  
- Heads: Dense layers with multi-label sigmoid outputs  
- Loss: Binary Crossentropy (per attribute), optionally weighted by inter-attribute correlation  

---

## 🧪 Results Summary

| Model                | Augmentation | Accuracy | Observations                          |
|---------------------|--------------|----------|---------------------------------------|
| `best_att_no_aug`   | ❌           | ~83%     | Overfits more quickly, lower generalization |
| `best_att_with_aug` | ✅           | ~88%     | More robust to occlusion & rotation noise  |
| `best_deep_face`    | ❓           | Baseline | Serves as baseline architecture        |

Attribute correlation map reveals strong co-dependencies (e.g., `mustache ↔ beard`, `mouth open ↔ smile`) which may justify multi-task learning paradigms.

---

## ⚙️ How to Run

1. Clone the repo and place WFLW images and annotations in respective folders.
2. Open `Facial_Attr_WFLW.ipynb` in Jupyter.
3. Run the notebook sequentially to train or evaluate.
4. (Optional) Modify hyperparameters or augmentation pipeline for experimentation.

---

## 📚 Citation / Acknowledgment

- WFLW Dataset: [Wu et al., LAB: Look-At-Border](https://wywu.github.io/projects/LAB/WFLW.html)
- Keras/TensorFlow as backend framework.
- This work draws conceptual motivation from DeepFashion, CelebA, and FairFace attribute benchmarks.

---

## 🧬 Future Directions

- Integrate Transformer-based feature extractors for enhanced context modeling  
- Incorporate explainability (e.g., GradCAM) for attribute-region localization  
- Optimize for mobile deployment using pruning and quantization

---

## 📩 Contact

This repository is maintained by Shima N.  
For serious inquiries or collaboration, reach out via GitHub or associated academic contact.

