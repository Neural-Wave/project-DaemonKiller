# import json
# import re
# import nltk
# from nltk.corpus import stopwords
# from unidecode import unidecode
# import glob as glob
# import os

# # Download multilingual stopwords if not already done
# nltk.download("stopwords")

# # Combine stopwords from multiple languages
# stop_words = set(stopwords.words("english") + 
#                  stopwords.words("german") + 
#                  stopwords.words("italian") + 
#                  stopwords.words("french"))

# def clean_text_multilingual(text):
#     """
#     Clean text in English, German, Italian, and French for RAG fine-tuning.
#     Args:
#         text (str): The text content of a document.
#     Returns:
#         str: The cleaned text.
#     """
#     text = text.encode('utf-8').decode('unicode_escape')
    
#     # Remove unwanted characters and formatting
#     text = re.sub(r'\*+', '', text)  
#     text = re.sub(r'\n+', ' ', text) 
#     text = re.sub(r'\r+', ' ', text)  
#     text = re.sub(r'\s+', ' ', text)
#     text = text.strip()
#     text = unidecode(text)
#     text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)
#     text = text.lower()
    
#     # Remove multilingual stopwords
#     text = ' '.join(word for word in text.split() if word not in stop_words)
#     return text

# def clean_json_documents_multilingual(input_path, output_path):
#     """
#     Clean text documents in multiple languages within a JSON file.
    
#     Args:
#         input_path (str): Path to the input JSON file.
#         output_path (str): Path to save the cleaned JSON file.
#     """
#     # Load the JSON data
#     files = glob.glob(os.path.join(input_path, '*.json'))
#     for file in files:
#         out_fname = file.split('/')[-1]
#         with open(file, 'r') as f:
#             data = json.load(f)
#             if not data["content"]: data["content"] = clean_text_multilingual(data["content"])
#     # Save cleaned data to a new JSON file
#         with open(f"{output_path}/{out_fname}", 'w', encoding='utf-8') as f:
#             json.dump(data, f, ensure_ascii=False)
#         print(f"Cleaned multilingual dataset saved to {output_path}")

# # Usage
# input_path = "/teamspace/studios/this_studio/dataset/parsed_documents/"
# output_path = "/teamspace/studios/this_studio/dataset/cleandataset"
# clean_json_documents_multilingual(input_path, output_path)
import json
import re
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
import glob
import os

# Download multilingual stopwords if not already done
nltk.download("stopwords")

# Combine stopwords from multiple languages
stop_words = set(stopwords.words("english") + 
                 stopwords.words("german") + 
                 stopwords.words("italian") + 
                 stopwords.words("french"))

def replace_none_with_empty_string(data):
    return {k: (v if v is not None else "Null") for k, v in data.items()}

def clean_text_multilingual(text):
    """
    Clean text in English, German, Italian, and French for RAG fine-tuning.
    Args:
        text (str): The text content of a document.
    Returns:
        str: The cleaned text.
    """
    text = text.encode('utf-8').decode('unicode_escape')
    
    # Remove unwanted characters and formatting
    text = re.sub(r'\*+', '', text)  
    text = re.sub(r'\n+', ' ', text)  # Replace newline characters with space
    text = re.sub(r'\r+', ' ', text)  # Replace carriage returns with space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()
    text = unidecode(text)
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)  # Keep letters and spaces only
    text = text.lower()
    
    # Remove multilingual stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def clean_json_documents_multilingual(input_path, output_path):
    """
    Clean text documents in multiple languages within JSON files.
    
    Args:
        input_path (str): Path to the input JSON file directory.
        output_path (str): Path to save the cleaned JSON files.
    """
    # Load the JSON data
    files = glob.glob(os.path.join(input_path, '*.json'))
    for file in files:
        out_fname = os.path.basename(file)
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = replace_none_with_empty_string(data=data)
            # Clean "content" if it exists
            if "content" in data:
                data["content"] = clean_text_multilingual(data["content"])
            
        
        # Save cleaned data to a new JSON file
        os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
        with open(os.path.join(output_path, out_fname), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        
        print(f"Cleaned multilingual dataset saved to {output_path}/{out_fname}")

# Usage
input_path = "/teamspace/studios/this_studio/dataset/parsed_documents/"
output_path = "/teamspace/studios/this_studio/dataset/cleandataset"
clean_json_documents_multilingual(input_path, output_path)
