import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2

# Load the pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO class names
coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush",
    "music instrument", 
]
# Function to process the image and make predictions
def process_image(uploaded_file):
    # Open and transform the image
    image = Image.open(uploaded_file).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(image)

    with torch.no_grad():
        pred = model([img_tensor])

    return image, pred

# Function to draw bounding boxes
def draw_boxes(image, pred):
    bboxes, labels, scores = pred[0]['boxes'], pred[0]['labels'], pred[0]['scores']
    num = torch.argwhere(scores > 0.8).shape[0]
    
    # Convert the image to a format suitable for OpenCV
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    for i in range(num):
        x1, y1, x2, y2 = bboxes[i].numpy().astype('int')
        class_name = coco_names[labels.numpy()[i] - 1]
        #overlay = image_cv.copy()
        #cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Fill the rectangle with green
        #alpha = 0.4  # Transparency factor
        #image_cv = cv2.addWeighted(overlay, alpha, image_cv, 1 - alpha, 0)
        image_cv = cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 1)
        image_cv = cv2.putText(image_cv, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return image_cv

# Streamlit app
def main():
    st.title("Image's Object Detection & Segmentation")
    st.subheader('©️ Faraz Siddiqui')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        image, pred = process_image(uploaded_file)
        output_image = draw_boxes(image, pred)

        st.image(output_image, caption='Detected Objects', channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
