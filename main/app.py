from fastapi import FastAPI, UploadFile, File
import os
import joblib
from src.image_processing import read_image, clean_extracted_text_img
from src.extract_values import extract_medical_data_from_full_blood_test

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

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Step 1: Save the uploaded image to a temporary file
    image_bytes = await file.read()
    image_path = "temp_image.png"
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # Step 2: Extract text from image using your function
    image = read_image(image_path)
    clean_text = clean_extracted_text_img(image)

    # Step 3: Get prediction from the cleaned text
    predictions = predict_condition(clean_text)

    # Clean up the temporary image file
    if os.path.exists(image_path):
        os.remove(image_path)

    return {"extracted_text": clean_text, "predictions": predictions}
