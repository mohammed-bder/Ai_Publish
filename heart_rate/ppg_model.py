import onnxruntime as ort
import joblib
import pandas as pd
import os

# Construct full path to the model
BASE_DIR = os.path.dirname(__file__)

# Load ONNX model
onnx_model_path = os.path.join(BASE_DIR, 'tfmodel.onnx')
ort_session = ort.InferenceSession(onnx_model_path)
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler2.pkl'))

def data_process( age, hr, gender=1):
    sample = {'Age': [age], 'HR': [hr], 'Gender': [gender]}

    data = pd.DataFrame(sample)
    # data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

    # Create the new parameters
    data['age_hr_interaction'] = data['Age'] * data['HR']
    data['age_normalized_by_hr'] = data['Age'] / (data['HR'] + 1)

    # Step 3: Transform the single sample
    features = ['Age', 'HR', 'age_hr_interaction', 'age_normalized_by_hr']
    data[features] = scaler.transform(data[features])

    # Now data is ready to be used as input to your model
    # print(data)
    return(data)

def predict_hr_condition(age: int, hr: float, gender: int = 1) -> str:
    data = data_process(age, hr, gender)
    input_name = ort_session.get_inputs()[0].name
    input_data = data.values.astype('float32')
    prediction = ort_session.run(None, {input_name: input_data})[0]
    return "Normal" if prediction[0] == 0 else "Abnormal"


