import os
import json

def extract_key_from_json(directory, key, output_directory):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # Open and read the JSON file
            with open(file_path, 'r', encoding='utf-8') as json_file:
                try:
                    data = json.load(json_file)
                    
                    # Extract content for the specified key
                    if key in data:
                        content = data[key]
                        
                        # Prepare output file path
                        output_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.txt")
                        
                        # Write content to text file
                        with open(output_file_path, 'w', encoding='utf-8') as txt_file:
                            txt_file.write(str(content))
                            print(f"Extracted content from '{filename}' to '{output_file_path}'")
                    else:
                        print(f"Key '{key}' not found in '{filename}'")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file '{filename}'")
                except Exception as e:
                    print(f"An error occurred with file '{filename}': {e}")

# Usage
directory = '/teamspace/studios/this_studio/dataset/parsed_documents/'  # Replace with your directory path
key = 'content'                       # Replace with the key you want to extract
output_directory = '/teamspace/studios/this_studio/dataset/rag_test/text'  # Replace with your output directory path

extract_key_from_json(directory, key, output_directory)
