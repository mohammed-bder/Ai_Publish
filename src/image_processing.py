from PIL import Image
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = 'tesseract'

def read_image(image_path: str) -> str:
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def clean_extracted_text_img(file_path):
    text = read_image(file_path)
    cleaned_text = re.sub(r"[\[\]{}(),:\*]", "",text) 
    
    return cleaned_text