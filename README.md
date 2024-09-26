# AI Pipeline for Image Object Detection and Segmentation  (Video of appraoch is explained at the bottom of this readme file)
This project is a Streamlit application that leverages object detection and segmentation to analyze images. The app allows users to upload an image, detect and segment objects using a pre-trained Mask R-CNN and Faster R-CNN model. The results are displayed along with summaries of the detected attributes.

# Features
**Image Upload:** Users can upload images in JPG or PNG format.
**Object Detection:** Detects objects in the image using a Mask R-CNN model.
**Object Segmentation:** Segments the detected objects and extracts them as separate images.
**Attribute Summarization:** Summarizes extracted text using a transformer-based summarization model.
**User-Friendly Interface:** Streamlit provides an intuitive interface for seamless interaction.

# Installation
Prerequisites
Python 3.8 or higher
pip (Python package installer)

# Install Dependencies
Install the required Python packages using pip: streamlit torch torchvision easyocr transformers opencv-python

# Running the App
After installing the dependencies, you can start the Streamlit app by running: streamlit run app.py
(Replace app.py with the name of Python file in streamlit folder)

# Usage
**Upload an Image:** Click on "Choose an image..." and upload a .jpg or .png file.
**View Results:** The app will display the uploaded image with detected objects and their summaries.
**Extracted Objects:** Each detected object will be displayed with its corresponding extracted text summary.

# Image Processing Functions
**process_image:** Converts the uploaded image to a tensor.
**segment_and_extract_objects:** Segments the image and returns masks and bounding boxes.
**draw_boxes:** Draws bounding boxes around detected objects.
**extract_objects:** Saves segmented objects as separate images.
**identify_and_extract_text:** Extracts text from segmented objects.
**summarize_objects:** Summarizes extracted text.

# Streamlit Interface
The main function defines the Streamlit app's layout and functionality:

# Troubleshooting
**OSError**: Ensure your environment has write permissions for temporary files.
**Model Errors**: Verify that the required models and libraries are properly installed.
Future Enhancements

**Database Integration**: Store detected object information in the SQLite database.
**Improved Text Extraction**: Integrate more robust text extraction techniques.
**User Customization**: Allow users to modify detection thresholds and model parameters.

# Acknowledgments
Mask R-CNN model by Facebook AI Research
EasyOCR by Jaided AI
Streamlit for the user interface framework

# Video Explaination of Approach
https://www.loom.com/share/4b9cd17174e842858dee46e003f9f937?sid=e253a4ce-5a11-4384-86e3-b0db1e61e753
