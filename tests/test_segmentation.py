import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow
import numpy as np

# Load the Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Download the image
!wget 'https://bikeshackleyton.com/cdn/shop/articles/shutterstock_211943116_1200x1200.jpg?v=1472662829'

# Read the image
image_path = '/content/shutterstock_211943116_1200x1200.jpg?v=1472662829'
image = Image.open(image_path)

# Transform the image to tensor
transform = T.ToTensor()
img_tensor = transform(image)

# Run the model
with torch.no_grad():
    pred = model([img_tensor])

# Process the outputs
masks = pred[0]['masks']
scores = pred[0]['scores']
threshold = 0.5  # Adjust the threshold for segmentation visibility
masks = masks[scores > threshold]

# Create an empty mask for visualization
segmented_image = np.zeros((image.height, image.width, 3), dtype=np.uint8)

# Loop through each mask and combine them
for i in range(masks.shape[0]):
    mask = masks[i, 0].mul(255).byte().cpu().numpy()  # Convert to binary mask
    color = np.random.randint(0, 255, size=(3,), dtype=int)  # Random color for each object
    segmented_image[mask > 0] = color  # Apply color to the mask region

# Convert segmented image to BGR for OpenCV
segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

# Display the original image and the segmented image
cv2_imshow(segmented_image_bgr)

original_image = np.array(image)
 for i in range(masks.shape[0]):
    mask = masks[i, 0].mul(255).byte().cpu().numpy()  # Convert to binary mask
    color = np.random.randint(0, 255, size=(3,), dtype=int)  # Random color for each object
    # Create a color overlay
    colored_mask = np.zeros_like(original_image)
    colored_mask[mask > 0] = color
    # Blend the colored mask with the original image
    alpha = 0.5  # Adjust transparency
    original_image = cv2.addWeighted(original_image, 1, colored_mask, alpha, 0)

# Display masked image
cv2_imshow(original_image)