### File: model/cnn_model.py

import torch
import torchvision.models as models
import torch.nn.functional as F
import os
from .describe_image import imagenet_classes


def load_model():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    return model


def predict_image(model, image_tensor):
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook

    # Register hooks for deeper insight
    hooks = []
    layers_to_hook = {
        'conv1': model.conv1,
        'layer1_block0': model.layer1[0],
        'layer2_block0': model.layer2[0],
        'layer3_block0': model.layer3[0],
        'layer4_block0': model.layer4[0],
        'avgpool': model.avgpool,
        'fc': model.fc,
    }

    for name, layer in layers_to_hook.items():
        hooks.append(layer.register_forward_hook(get_activation(name)))

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0))
        probabilities = F.softmax(outputs, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        _, predicted = torch.max(outputs, 1)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Prepare top-5 predictions
    top5 = [
        {"label": imagenet_classes[top5_catid[0][i]], "prob": float(top5_prob[0][i])}
        for i in range(top5_prob.size(1))
    ]

    # Build explainable info
    explain = [
        f"Input shape: {tuple(image_tensor.shape)}",
        "Top-5 predictions:",
        *[f"  {t['label']}: {t['prob']:.4f}" for t in top5],
        f"Final prediction: {imagenet_classes[predicted.item()]}",
        "Layer-wise activations (shapes):",
        *[f"  {layer}: {activations[layer].shape}" for layer in activations],
    ]

    return (
        imagenet_classes[predicted.item()],
        {
            "top5": top5,
            # "probabilities": probabilities.squeeze().tolist(),
            "layer_activations": {k: v.tolist() for k, v in activations.items()},
            "thought_process": explain
        }
    )
