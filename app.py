from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import joblib
import tempfile
from pathlib import Path
from src.image_processing import read_image, clean_extracted_text_img
from src.extract_values import extract_medical_data_from_full_blood_test
from src.text_processing import clean_extracted_text_txt
from pypdf import PdfReader


# Initialize FastAPI app
app = FastAPI()

# Load the trained model once when the app starts
model = joblib.load("decision_tree_model.pkl")

# Define label mapping
label_mapping = ["anemic", "infection", "autoimmune disease", "injury"]

# Predict condition based on extracted and cleaned text
def predict_condition(clean_txt: str):
    df = extract_medical_data_from_full_blood_test(clean_txt)
    df_wide = df.set_index("Parameter")["Value"].transpose().to_frame().T
    df_wide.columns.name = None
    prediction = model.predict(df_wide)
    predicted_labels = [label_mapping[i] for i in range(len(prediction[0])) if prediction[0][i] == 1] or ["normal"]
    return predicted_labels

# Process uploaded image file
def process_image_file(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    image = read_image(tmp_path)
    clean_text = clean_extracted_text_img(image)
    os.remove(tmp_path)
    return clean_text

# Process uploaded PDF file
def process_pdf_file(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        reader = PdfReader(f)
        clean_text = "".join([page.extract_text() or "" for page in reader.pages])
    os.remove(tmp_path)
    return clean_text

# Process uploaded TXT file
def process_txt_file(file_bytes: bytes):
    text = file_bytes.decode("utf-8")
    return clean_extracted_text_txt(text)

# Unified handler
def process_uploaded_file(file_bytes: bytes, filename: str):
    ext = Path(filename).suffix.lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        return process_image_file(file_bytes)
    elif ext == ".pdf":
        return process_pdf_file(file_bytes)
    elif ext == ".txt":
        return process_txt_file(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

# FastAPI endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        clean_text = process_uploaded_file(file_bytes, file.filename)
        predictions = predict_condition(clean_text)
        return {"extracted_text": clean_text, "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
