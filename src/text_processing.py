import pypdf # for reading pdfs
import re

# extract text from a pdf
def read_pdf(pdf_path):
    pdf_reader = pypdf.PdfReader(pdf_path)
    pdf_text = ""
    
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() + "\n"
    return pdf_text

# extractin text from a txt file
def read_txt(txt_file):
    with open(txt_file, "r", encoding="utf-8") as file:
        content = file.read()
    return(content)

# removing {}[](),:* from a pdf
def clean_extracted_text_pdf(pdf_path):
    pdf_text = read_pdf(pdf_path)
    cleaned_text = re.sub(r"[\[\]{}(),:\*]", "", pdf_text)
    
    return cleaned_text

# removing {}[](),:* from a txt
def clean_extracted_text_txt(txt_path):
    text = read_txt(txt_path)
    cleaned_text = re.sub(r"[\[\]{}(),:\*]", "",text)
    
    return cleaned_text
