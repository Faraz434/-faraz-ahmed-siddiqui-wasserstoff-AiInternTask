import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import cv2
import pytesseract
import numpy as np
from google.colab.patches import cv2_imshow

# Load the pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Download the image
!wget 'https://bikeshackleyton.com/cdn/shop/articles/shutterstock_211943116_1200x1200.jpg?v=1472662829'

# Read the image
img_path = '/content/shutterstock_211943116_1200x1200.jpg?v=1472662829'
ig = Image.open(img_path)

# Transform the image to tensor
transform = T.ToTensor()
img = transform(ig)

# Perform inference
with torch.no_grad():
    pred = model([img])

# Extract predictions
bboxes = pred[0]['boxes']
labels = pred[0]['labels']
scores = pred[0]['scores']

# Filter out low confidence detections
num = torch.argwhere(scores > 0.8).shape[0]

coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase",
              "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
              "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
              "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
              "hair brush"]

# Read the image using OpenCV
igg = cv2.imread(img_path)

# Prepare a list to store extracted text data
extracted_data = []

# Iterate over detected objects
for i in range(num):
    x1, y1, x2, y2 = bboxes[i].numpy().astype('int')
    class_name = coco_names[labels.numpy()[i] - 1]

    # Draw bounding boxes
    igg = cv2.rectangle(igg, (x1, y1), (x2, y2), (0, 255, 0), 1)
    igg = cv2.putText(igg, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Crop the object from the image
    object_img = igg[y1:y2, x1:x2]

    # Convert the cropped image to grayscale for OCR
    gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    # Store the class name and extracted text
    extracted_data.append({'class': class_name, 'text': text.strip()})

# Show the image with bounding boxes
cv2_imshow(igg)

# Output the extracted data
for data in extracted_data:
    print(f"Detected: {data['class']}, Extracted Text: {data['text']}")

















