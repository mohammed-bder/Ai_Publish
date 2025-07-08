
from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import joblib
import tempfile
from pathlib import Path
from src.image_processing import read_image, clean_extracted_text_img
from src.extract_values import extract_medical_data_from_full_blood_test
from src.text_processing import clean_extracted_text_txt

# hr model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated
from heart_rate.ppg_model import predict_hr_condition  # ✅ Import your model logic
from heart_rate.onnx_hr_model import predict_from_onnx
from pypdf import PdfReader


# Initialize FastAPI app
app = FastAPI()

#print("app.py is running...")
# Load the trained model once when the app starts
model = joblib.load('main/decision_tree_model.pkl')

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
    clean_text = clean_extracted_text_img(tmp_path)
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
    return clean_extracted_text_txt(file_bytes)

# Unified handler
def process_uploaded_file(file_bytes: bytes, filename: str):
    ext = Path(filename).suffix.lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        return process_image_file(file_bytes)
    elif ext == ".pdf":
        return process_pdf_file(file_bytes)
    elif ext == ".txt":
        return clean_extracted_text_txt(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

# Pydantic model for request body
class HRInput(BaseModel):
    age: Annotated[int, Field(gt=0, lt=120)]
    hr: Annotated[float, Field(gt=0, lt=300)]
    gender: Annotated[int, Field(ge=0, le=1)]

class HRRInput(BaseModel):
    age: Annotated[int, Field(gt=1, lt=200)]
    hr: Annotated[float, Field(gt=0, lt=300)]

# FastAPI endpoint
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    #print("Endpoint hit!")
    try:
        file_bytes = await file.read()
        #print(f"Received filename: {file.filename}")
        #print(f"First 100 bytes: {file_bytes[:100]}")  # Just to inspect

        clean_text = process_uploaded_file(file_bytes, file.filename)
        predictions = predict_condition(clean_text)

        return {"predictions": predictions}
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict-hr")
async def predict_heart_rate(input: HRInput):
    try:
        result = predict_hr_condition(input.age, input.hr, input.gender)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict-hr-onnx")
async def predict_heart_rate_onnx(input: HRRInput):
    try:
        result = predict_from_onnx(input.age, input.hr)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))