import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify

# Load Excel Data
excel_folder = r"C:\Users\aspk1\OneDrive\Desktop\Gmail-Classification\Downloads"
excel_files = [f for f in os.listdir(excel_folder) if f.endswith('.xlsx')]

dataframes = []
for file in excel_files:
    df = pd.read_excel(os.path.join(excel_folder, file))
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Preprocessing function
def preprocess_text(text):
    encoding = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    return encoding

# Predict function
def classify_email(text):
    model.eval()
    inputs = preprocess_text(text)
    with torch.no_grad():
        outputs = model(**inputs)
    label = torch.argmax(outputs.logits, dim=1).item()
    return "Spam" if label == 1 else "Not Spam"

# Flask API
app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    email_text = request.json.get('email_text', '')
    classification = classify_email(email_text)
    return jsonify({"classification": classification})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
