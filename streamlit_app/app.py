import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2
import os
import shutil
import pandas as pd

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
    "music instrument"
]

# Function to process the image and make predictions
def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(image)
    
    with torch.no_grad():
        pred = model([img_tensor])

    return image, pred

# Function to draw bounding boxes with object names and IDs
def draw_boxes(image, pred, object_summary):
    bboxes, labels, scores = pred[0]['boxes'], pred[0]['labels'], pred[0]['scores']
    num = torch.argwhere(scores > 0.8).shape[0]
    
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    for i in range(num):
        x1, y1, x2, y2 = bboxes[i].numpy().astype('int')
        class_name = coco_names[labels.numpy()[i] - 1]
        unique_id = f"ID-{i+1}"
        
        # Draw bounding box
        image_cv = cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 1)
        image_cv = cv2.putText(image_cv, f"{unique_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Append object details to summary
        object_summary.append({
            'ID': unique_id,
            'Class': class_name,
            'Confidence': round(scores[i].item(), 2),
            'Bounding Box': f"({x1}, {y1}), ({x2}, {y2})"
        })

    return image_cv

# Function to clear output directory
def clear_output_dir(output_dir="output_objects"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Delete the folder and its contents
    os.makedirs(output_dir)  # Recreate the empty folder

# Function to extract objects and map them with unique IDs and attributes
def extract_objects(image, bboxes, labels, output_dir="output_objects"):
    clear_output_dir(output_dir)  # Clear old objects

    image_np = np.array(image)
    extracted_objects = []
    
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = bbox.numpy().astype('int')
        object_img = image_np[y1:y2, x1:x2]  # Extract the object based on bounding box
        object_img_pil = Image.fromarray(object_img)
        
        # Generate unique ID and object class name
        class_name = coco_names[label - 1]  # Adjust for COCO labels starting from 1
        unique_id = f"ID-{i+1}"
        
        # Save object image with unique ID and class name in the filename
        object_img_pil.save(f"{output_dir}/object_{unique_id}_{class_name}.png")
        
        # Store extracted object attributes
        extracted_objects.append({
            'ID': unique_id,
            'Class': class_name,
            'Bounding Box': (x1, y1, x2, y2)
        })

    return extracted_objects

# Streamlit app
def main():
    st.title("Image Object Detection & Segmentation with Object Summary")
    st.subheader('©️ Faraz Siddiqui')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        image, pred = process_image(uploaded_file)
        
        # Prepare a summary for objects
        object_summary = []
        output_image = draw_boxes(image, pred, object_summary)
        
        # Filter bounding boxes and labels by score > 0.8
        bboxes = pred[0]['boxes'][pred[0]['scores'] > 0.8]
        labels = pred[0]['labels'][pred[0]['scores'] > 0.8]
        
        # Extract objects with unique IDs and object names, return object attributes
        extracted_objects = extract_objects(image, bboxes, labels)
        
        st.image(output_image, caption='Detected Objects', channels="BGR", use_column_width=True)

        # Display summary of detected objects
        st.write("### Object Detection Summary")
        df_summary = pd.DataFrame(object_summary)
        st.dataframe(df_summary)

        # Display extracted objects
        st.write("### Extracted Objects")
        output_dir = "output_objects"
        for filename in os.listdir(output_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(output_dir, filename)
                st.image(img_path, caption=filename, use_column_width=True)

if __name__ == "__main__":
    main()
