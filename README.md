\# 🩺 Skin Disease Classification using Transfer Learning



A deep learning model that classifies 7 types of skin diseases using ResNet-50 transfer learning on the HAM10000 dataset.



\## 📊 Results

| Metric     | Score  |

|------------|--------|

| Accuracy   | 89%    |

| F1 Score   | 0.87   |

| Dataset    | HAM10000 (10,015 images) |



\## 🏗️ Model Architecture

\- Base: ResNet-50 (pretrained on ImageNet)

\- Fine-tuned last 3 layers

\- Custom classification head for 7 classes

\- Data augmentation: flip, rotate, color jitter



\## 📁 Project Structure

```

skin-disease-classifier/

├── train.py         # Training script

├── predict.py       # Inference script

├── requirements.txt # Dependencies

├── model/           # Saved weights

└── notebooks/       # EDA notebook

```



\## ▶️ How to Run

```bash

pip install -r requirements.txt

python train.py

python predict.py your\_image.jpg

```



\## 📦 Dataset

Download HAM10000 from \[Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)



\## 🛠️ Tech Stack

Python | PyTorch | ResNet-50 | HuggingFace | Gradio

