
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.cnn_model import load_model, predict_image
from utils.image_utils import load_and_preprocess

image_folder = 'dataset/sample_images'
output_file = 'results.json'

model = load_model()
results = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        img_tensor = load_and_preprocess(image_path)
        label, debug_info = predict_image(model, img_tensor)

        print(f"{filename}: {label}")

        results.append({
            "filename": filename,
            "label": label,
            "top5": debug_info["top5"],
            # "probabilities": debug_info["probabilities"],
            "layer_activations_shapes": {k: [len(v), len(v[0])] if isinstance(v, list) and len(v) > 0 else [] for k, v in debug_info["layer_activations"].items()},
            "thought_process": debug_info["thought_process"]
        })

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")
