#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
#sys.path.append('D:/final_year_project/gitBackup/src')  # or wherever your src folder is
import os
sys.path.append(os.path.abspath(r'E:/Projects/cbc_repeort/src'))


# In[2]:


import re # for data cleaning and extraction
import pandas as pd  #
import pypdf # for reading pdfs


# In[19]:


import joblib

# Load the saved model
model = joblib.load('decision_tree_model.pkl')


# In[ ]:


# from src.image_processing import*
from image_processing import *
from extract_values import *
from text_processing import *
#from src.__init__ import*


# In[ ]:


image_path = r"C:/Users/ASUS/Downloads/IMG_20250416_181949.jpg" # replace with right image path
pdf_path = r"" # replace with right pdf path
txt_path = r"" # replace with right txt path


# In[ ]:


clean_txt = clean_extracted_text_img(image_path)


# In[ ]:


#clean_txt = clean_extracted_text_pdf(pdf_path)


# In[ ]:


#clean_txt = clean_extracted_text_txt(txt_path)


# In[33]:


df = extract_medical_data_from_full_blood_test(clean_txt)


# In[34]:


df_wide = df.set_index('Parameter')['Value'].transpose().to_frame().T

# Optionally reset column names if you want cleaner formatting
df_wide.columns.name = None

# View the result
# print(df_wide)


# In[35]:


# Get the model's prediction
prediction = model.predict(df_wide)

# Print the result
print("Predicted Result:", prediction[0])


# In[36]:


# Define label mapping
label_mapping = ["anemic", "infection", "autoimmune disease", "injury"]

# Get predicted labels
predicted_labels = [
    label_mapping[i] for i in range(len(prediction[0])) if prediction[0][i] == 1
] or ["normal"]
print("Predicted Conditions:", predicted_labels)

