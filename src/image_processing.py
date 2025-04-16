from PIL import Image
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image_path: str) -> str:
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def clean_extracted_text_img(file_path):
    text = extract_text(file_path)
    cleaned_text = re.sub(r"[\[\]{}(),:\*]", "",text) 
    
    return cleaned_text