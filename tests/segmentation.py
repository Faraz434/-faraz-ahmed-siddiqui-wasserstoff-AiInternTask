import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load a pre-trained Mask R-CNN model from torchvision
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to perform segmentation
def segment_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)

    # Pass the image to the model
    with torch.no_grad():
        predictions = model(img_tensor)

    # Get masks and draw them on the image
    masks = predictions[0]['masks'].numpy()
    image_np = np.array(image)

    for mask in masks:
        mask = mask[0]
        mask = np.where(mask > 0.5, 1, 0).astype(np.uint8) * 255
        colored_mask = np.zeros_like(image_np)
        colored_mask[:, :, 2] = mask  # Color the mask red
        image_np = cv2.addWeighted(image_np, 1, colored_mask, 0.5, 0)

    # Show the segmented image
    plt.imshow(image_np)
    plt.axis("off")
    plt.show()

# Example usage
# Use the absolute path if necessary
segment_image("C:/Users/yogesh/Downloads/image.jpg")

