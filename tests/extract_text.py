import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    print("Extracted Text:", text)

# Example usage
extract_text_from_image("path_to_object_image.png")
