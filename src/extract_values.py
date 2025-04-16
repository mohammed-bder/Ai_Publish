import re # for data cleaning and extraction
import pandas as pd  #

# extracting the values needed from the test using re library
def extract_value(parameter_variants, text):
    parameter_regex = "|".join(parameter_variants)  # Example: "Hemoglobin|HGB"
    # Define multiple possible patterns
    patterns = [
        fr"(?i)({parameter_regex})[^.^C][\s\D/]*([\d\.]+)\s*[\D\s]*\%?[\s\D\d/?]*\%?\d*\.?\d*\s*-\s*\d+\.?\d*", # (value at the middle)
        fr"(?i)({parameter_regex})[\s\D/]*\%?\w*\s*/?\w*\s*\d+\.?\d*\s*-\s*\d+\.?\d*\D*([\d\.]+)"  #  (value at end)
        
    ]
    # print(type(parameter_regex))
    for pattern in patterns:
        matches = re.finditer(pattern, text)  # Finds all matches in text
        # print(f"Testing pattern: {pattern}") 
        for match in matches:
            if match:
                for i in range(1, len(match.groups()) + 1):
                    if match.group(i) and re.finditer(r"\d+\.?\d*", match.group(i)):
                        # print(match.group(i+1))
                        result = float(match.group(i+1))
                        return(f"{result:.2f}")

            break # Stop checking other patterns once we find a match 
    # return "N/A"  # If no match found, return "N/A"



    # specifies the values to be extracted and returns a dataframe of them
def extract_medical_data_from_full_blood_test(pdf_text):
    # Extract values dynamically
    hemoglobin = extract_value(["hgb","Hemoglobin","hb","Haemoglobin"], pdf_text) 
    wbc = extract_value(["White Blood Cell","wbc Count","w\.?b\.?cs?"], pdf_text) 
    rbc = extract_value(["rbc Count","Red Blood Cell","r\.?b\.?cs?"], pdf_text) 
    hematocrit = extract_value(["Hematocrit","hct","Haematocrit"], pdf_text) 
    mcv = extract_value(["MCV","Mean Cell Volume","M.C.V"], pdf_text) 
    mch = extract_value(["MCH","Mean Cell Hemoglobin","M.C.H"], pdf_text) 
    mchc = extract_value(["MCHC","Mean Cell Hb Conc","M.C.H.C"], pdf_text) 
    rdw = extract_value(["RDW CV","Red Cell Dist Width"], pdf_text) 
    Neutrophils = extract_value(["Neutrophils?"], pdf_text) 
    NeutrophilsSegmented = extract_value(["Neutrophils-Segmented?","Segmented"], pdf_text) 
    Lymphocytes = extract_value(["Lymphocytes?"], pdf_text) 
    Eosinophils = extract_value(["Eosinophils?"], pdf_text) 
    Monocytes = extract_value(["Monocytes?"], pdf_text) 
    Basophils = extract_value(["Basophils?"], pdf_text) 
    Platelets = extract_value(["Platelet Count","Platelets"], pdf_text) 
    mpv =  extract_value(["Mean Platelet Volume"], pdf_text) 
    

    data = {
        "Parameter": ["Basophils","Neutrophils","Eosinophils","Hemoglobin","RDW", "Monocytes","MCH","Platelets","Hematocrit","Lymphocytes",
                      "MCHC","RBC","Neutrophils - Segmented %","MCV","WBC"],
        "Value": [Basophils, Neutrophils, Eosinophils, hemoglobin, rdw, Monocytes, mch, Platelets, hematocrit, Lymphocytes,
                    mchc, rbc,NeutrophilsSegmented, mcv, wbc],
       
    }

   
    
    df = pd.DataFrame(data)
    
    return df
