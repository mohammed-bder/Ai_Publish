import os
import re
import pandas as pd
import joblib
import tempfile
from pathlib import Path
from src.image_processing import read_image, clean_extracted_text_img
from src.extract_values import extract_medical_data_from_full_blood_test
from src.text_processing import clean_extracted_text_txt
import PyPDF2  # For PDF extraction

# Load the trained model
model = joblib.load('decision_tree_model.pkl')

# Label mapping for predictions
label_mapping = ["anemic", "infection", "autoimmune disease", "injury"]

# Helper function for prediction
def predict_condition(clean_txt: str):
    df = extract_medical_data_from_full_blood_test(clean_txt)
    df_wide = df.set_index('Parameter')['Value'].transpose().to_frame().T
    df_wide.columns.name = None

    prediction = model.predict(df_wide)
    predicted_labels = [
        label_mapping[i] for i in range(len(prediction[0])) if prediction[0][i] == 1
    ] or ["normal"]
    return predicted_labels

# Function to process image files
def process_image_file(image_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_image_file:
        tmp_image_file.write(image_bytes)
        tmp_image_path = tmp_image_file.name

    image = read_image(tmp_image_path)
    clean_text = clean_extracted_text_img(image)
    os.remove(tmp_image_path)  # Clean up temporary file
    return clean_text

# Function to process PDF files
def process_pdf_file(pdf_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
        tmp_pdf_file.write(pdf_bytes)
        tmp_pdf_path = tmp_pdf_file.name

    # Read the PDF file and extract text
    with open(tmp_pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        clean_text = ""
        for page in reader.pages:
            clean_text += page.extract_text()
    
    os.remove(tmp_pdf_path)  # Clean up temporary file
    return clean_text

# Function to process text files
def process_txt_file(txt_bytes: bytes):
    # Decode bytes to string and clean the extracted text
    clean_text = txt_bytes.decode("utf-8")
    return clean_extracted_text_txt(clean_text)

# Main function to decide which processing function to call based on file type
def process_uploaded_file(file_bytes: bytes, file_type: str):
    if file_type == "image":
        return process_image_file(file_bytes)
    elif file_type == "pdf":
        return process_pdf_file(file_bytes)
    elif file_type == "text":
        return process_txt_file(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# Example of how the script might be executed with file paths
if __name__ == "__main__":
    # Example: how to handle different types of files
    file_path = "path/to/your/file"  # Change to your desired file
    file_type = "image"  # Change this to "pdf" or "text" based on the file you're testing

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    clean_text = process_uploaded_file(file_bytes, file_type)
    predictions = predict_condition(clean_text)
    
    print("Predicted Conditions:", predictions)
