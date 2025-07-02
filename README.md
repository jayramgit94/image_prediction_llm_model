# 🧠 image-llm-mini

> An explainable image classification system using CNN (ResNet50) — with built-in transparency and layer-by-layer model reasoning.

---

## 📸 About the Project

`image-llm-mini` is a mini LLM-style vision system that:
- Takes input images
- Classifies them using **ResNet50**
- Outputs:
  - ✅ Top-5 predicted classes with confidence scores
  - 🧠 Model's **thought process**
  - 🔍 **Layer-wise activation summaries**
  - 📊 Clean JSON result file for downstream usage

This is a foundational system for building **interpretable AI** for vision tasks.

---

## 🏗️ Features

- 🔬 **Top-5 classification** with softmax probabilities
- 📜 **Model reasoning output** ("thought_process") for each prediction
- 📊 **Layer activation shapes** for debugging/explainability
- 📦 Structured results in `results.json`
- 🖼️ Easily extendable to Grad-CAM, video, or multi-image pipelines

---

## 🗂️ Project Structure

```text
image-llm-mini/
│
├── app/                  # Main script and execution logic
├── model/                # CNN model, hooks, and prediction logic
│   └── cnn_model.py
│   └── describe_image.py
├── utils/                # Image loading and preprocessing
├── dataset/              # Sample images and classes file
│   └── imagenet_classes.txt
├── research/             # (Optional) Research utilities / experiments
├── results.json          # Output of classification (auto-generated)
├── requirements.txt      # Python dependencies
├── run.bat               # Optional batch file to run app
└── README.md             # You're here
🚀 How to Run
1. 📦 Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Make sure your virtual environment is active if you're using .venv.

2. 🧪 Run the Classifier
bash
Copy
Edit
python app/main.py
It will:

Process all images in dataset/sample_images

Classify them

Save detailed results in results.json

📊 Sample Output (results.json)
json
Copy
Edit
{
  "filename": "dog.jpg",
  "label": "German shepherd",
  "top5": [
    {"label": "German shepherd", "prob": 0.2975},
    {"label": "malinois", "prob": 0.0071},
    ...
  ],
  "layer_activations_shapes": {
    "conv1": [1, 64],
    "layer4_block0": [1, 2048],
    ...
  },
  "thought_process": [
    "Input shape: (3, 224, 224)",
    "Top-5 predictions:",
    "  German shepherd: 0.2975",
    ...
  ]
}
🧠 Future Ideas (To-Do)
 🔥 Add Grad-CAM heatmap for visual attention

 📽️ Support video frame classification

 🧪 Add CLI interface for single image

 🌐 Deploy as a Streamlit or Flask web app

🧑‍💻 Author
Jayram Sangawat
B.Tech CSE, J D College of Engineering
📫 sangawatjayram@gmail.com
🌐 LinkedIn

📄 License
This project is licensed under the MIT License. See LICENSE for details.

yaml
Copy
Edit

---

## ✅ Next Step

- Save this as `README.md` in your project root.
- Commit and push:
```bash
git add README.md
git commit -m "Added professional README"
git push