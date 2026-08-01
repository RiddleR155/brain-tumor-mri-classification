# Brain Tumor MRI Classification

A comparative deep learning study on classifying brain tumors from MRI scans, benchmarking four different model architectures to evaluate trade-offs between classical CNNs, hybrid approaches, and modern transformer/convolutional backbones.

Developed as a Deep Learning course project.

## 🧠 Overview

This project classifies brain tumor MRI images by training and comparing four distinct architectures:

| Model | Approach |
|---|---|
| **Simple CNN** | Baseline convolutional neural network |
| **VGG-16 + Random Forest** | Pretrained VGG-16 as a feature extractor, with a Random Forest classifier head |
| **ViT-B/16** | Vision Transformer, patch-based attention approach |
| **ConvNeXt-Base** | Primary model — modernized convolutional architecture |

**ConvNeXt-Base** was the primary/best-performing model in this study.

## 📊 Results

Confusion matrices and per-class metrics for the final model are included in this repo:

- `confusion_matrix.png` — confusion matrix visualization
- `Class metrics.png` — per-class precision/recall/F1 breakdown

*(Add a short summary of your final accuracy/F1 numbers here once finalized.)*

## 🛠️ Tech Stack

- Python
- PyTorch
- ConvNeXt-Base, ViT-B/16, VGG-16, Simple CNN
- scikit-learn (Random Forest classifier)
- Trained and benchmarked on Google Colab and Kaggle Notebooks

## 📁 Repository Contents

```
├── colab_final.py                          # Final training/evaluation script
├── confusion_matrix.png                     # Confusion matrix visualization
├── Class metrics.png                          # Per-class performance metrics
├── brain_tumor_report.docx                     # Full project report
├── Brain_Tumor_DL_Literature_Review-1.docx      # Literature review
└── Link to Colab Notebook.txt                     # Link to the interactive notebook
```

## 📓 Notebook

The full interactive training notebook (with all four models, training curves, and evaluation) is linked in `Link to Colab Notebook.txt`.

## 📄 Documentation

- **`brain_tumor_report.docx`** — full write-up of methodology, results, and analysis
- **`Brain_Tumor_DL_Literature_Review-1.docx`** — literature review covering prior work in MRI-based tumor classification

## 🎓 Context

Built as a Deep Learning course project, focused on evaluating how classical CNNs, hybrid CNN+ML approaches, and modern transformer/convolutional architectures compare on medical image classification tasks.

## 📝 License

Open for personal/academic reference.
