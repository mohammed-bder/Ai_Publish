from fastapi import FastAPI, UploadFile, File
from src.image_processing import read_image, clean_extracted_text_img

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_path = "temp_image.png"
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    image = read_image(image_path)
    clean_text = clean_extracted_text_img(image)
    return {"extracted_text": clean_text}
