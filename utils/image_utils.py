from torchvision import transforms
from PIL import Image

def load_and_preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    return preprocess_pil_image(image)

def preprocess_pil_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image)