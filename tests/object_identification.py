import torch
from PIL import Image
import clip

# Load the CLIP model
model, preprocess = clip.load("ViT-B/32")

def identify_objects(image_path):
    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits_per_image, _ = model(image_tensor, None)
        probs = logits_per_image.softmax(dim=-1)

    print(f"Predicted descriptions: {probs}")

# Example usage
identify_objects("path_to_image.jpg")
