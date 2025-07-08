# heart_rate/onnx_hr_model.py
import onnxruntime as ort
import numpy as np

# Load ONNX model and create session
model_path = "heart_rate/tfmodel.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

def scale_input(age: float, hr: float) -> np.ndarray:
    age_hr_interaction = age * hr
    age_normalized_by_hr = age / (hr + 1)
    extra_feature = age + hr

    scaled_age = (age - 40) / 30
    scaled_hr = (hr - 70) / 20
    scaled_interaction = (age_hr_interaction - 1000) / 5000
    scaled_normalized = (age_normalized_by_hr - 0.5) / 0.5
    scaled_extra = (extra_feature - 100) / 50

    return np.array([[scaled_age, scaled_hr, scaled_interaction, scaled_normalized, scaled_extra]], dtype=np.float32)

def predict_from_onnx(age: float, hr: float) -> str:
    input_data = scale_input(age, hr)
    result = session.run(None, {input_name: input_data})[0]

    if isinstance(result, np.ndarray):
        prediction = result[0] if result.ndim == 1 else result[0][0]
    else:
        raise ValueError("Unexpected output type from ONNX model.")

    return "Normal" if prediction == 0 else "Abnormal"
