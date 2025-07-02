import os

def get_imagenet_classes():
    classes_path = os.path.join(os.path.dirname(__file__), "../dataset/imagenet_classes.txt")
    with open(classes_path, "r") as f:
        return [line.strip() for line in f.readlines()]

imagenet_classes = get_imagenet_classes()