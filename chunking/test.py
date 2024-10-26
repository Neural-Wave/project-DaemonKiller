import json 
import glob as glob
import os
input_path = '../dataset/raw_documents'
ps = glob.glob(os.path.join(input_path, '*.json'))
with open(ps[0], 'r') as f:
    data = json.load(f)

print(data.keys())



