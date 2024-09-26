# Required Libraries
import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image, ImageDraw
import numpy as np
import cv2
import easyocr
import os
import sqlite3
from transformers import pipeline

# Load the pre-trained models
segmentation_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
segmentation_model.eval()

# Initialize EasyOCR reader and summarization pipeline
reader = easyocr.Reader(['en'])
summarizer = pipeline("summarization")

# COCO class names
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

# SQLite Database Setup
def setup_database():
    conn = sqlite3.connect('segmented_objects.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS objects (
        id INTEGER PRIMARY KEY,
        master_id INTEGER,
        label TEXT,
        filename TEXT
    )''')
    return conn

# Image Processing
def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(image)
    return image, img_tensor

# Object Segmentation and Extraction
def segment_and_extract_objects(image, img_tensor):
    with torch.no_grad():
        pred = segmentation_model([img_tensor])
    
    masks = pred[0]['masks']
    scores = pred[0]['scores']
    boxes = pred[0]['boxes']
    
    # Apply a lower confidence threshold
    threshold = 0.8  # Adjust this value as needed
    keep = scores > threshold

    masks = masks[keep]
    boxes = boxes[keep]
    
    return masks, boxes

# Draw Bounding Boxes
def draw_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes.tolist(), labels):
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), label, fill="red")
    return image

def extract_objects(image, masks):
    extracted_objects = []
    output_dir = 'extracted_objects'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(masks.shape[0]):
        mask = masks[i, 0].mul(255).byte().cpu().numpy()
        extracted_object = cv2.bitwise_and(np.array(image), np.array(image), mask=mask)
        object_filename = f'{output_dir}/object_{i+1}.png'
        cv2.imwrite(object_filename, extracted_object)
        extracted_objects.append(object_filename)
    return extracted_objects

# Object Identification and Text Extraction
def identify_and_extract_text(extracted_objects):
    extracted_data = []
    for object_img_path in extracted_objects:
        obj_image = cv2.imread(object_img_path)
        text = reader.readtext(obj_image)
        extracted_data.append({'path': object_img_path, 'text': " ".join([t[1] for t in text])})
    return extracted_data

# Summarize Attributes
def summarize_objects(extracted_data):
    summaries = []
    for data in extracted_data:
        summary_input = f"Text extracted: {data['text']}."
        summary = summarizer(summary_input, max_length=30, min_length=10, do_sample=False)
        summaries.append({'path': data['path'], 'summary': summary[0]['summary_text']})
    return summaries

# Streamlit Interface
def main():
    st.title("Image Object Detection & Segmentation")
    st.subheader("copyright Faraz Siddiqui")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        image, img_tensor = process_image(uploaded_file)
        
        # Step 1: Segment the image
        masks, boxes = segment_and_extract_objects(image, img_tensor)
        
        # Draw bounding boxes with labels
        labels = [coco_names[i] for i in range(len(boxes))]  # Assuming one label per box
        image_with_boxes = draw_boxes(image.copy(), boxes, labels)
        
        st.image(image_with_boxes, caption='Image with Detected Objects', use_column_width=True)

        # Step 2: Identify objects and extract text
        extracted_objects = extract_objects(image, masks)
        extracted_data = identify_and_extract_text(extracted_objects)
        
        # Step 3: Summarize attributes
        summaries = summarize_objects(extracted_data)
        
        # Display results
        for summary in summaries:
            st.image(summary['path'], caption='Extracted Object', use_column_width=True)
            st.write(f"Summary: {summary['summary']}")

if __name__ == "__main__":
    main()
