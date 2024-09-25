import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import os
import sqlite3

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
labels = pred[0]['labels'][scores > threshold]

# Create a directory to save extracted objects
output_dir = '/content/extracted_objects'
os.makedirs(output_dir, exist_ok=True)

# Connect to SQLite database
conn = sqlite3.connect('segmented_objects.db')
cursor = conn.cursor()

# Create a table to store metadata
cursor.execute('''
CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY,
    master_id INTEGER,
    label TEXT,
    filename TEXT
)
''')

# Extract and save each object
master_id = 1  # Unique ID for the original image
for i in range(masks.shape[0]):
    mask = masks[i, 0].mul(255).byte().cpu().numpy()  # Convert to binary mask
    color = np.random.randint(0, 255, size=(3,), dtype=int)  # Random color for each object

    # Create a colored mask
    colored_mask = np.zeros_like(np.array(image))
    colored_mask[mask > 0] = color

    # Extract the object from the original image
    extracted_object = cv2.bitwise_and(np.array(image), np.array(image), mask=mask)

    # Save the extracted object image
    object_filename = f'object_{master_id}_{i+1}.png'
    cv2.imwrite(os.path.join(output_dir, object_filename), extracted_object)

    # Insert metadata into the database
    cursor.execute('''
    INSERT INTO objects (master_id, label, filename)
    VALUES (?, ?, ?)
    ''', (master_id, str(labels[i].item()), object_filename))

# Commit changes and close the database connection
conn.commit()
conn.close()

# Display a message
print(f"Extracted {masks.shape[0]} objects and saved to '{output_dir}' with metadata in 'segmented_objects.db'.")