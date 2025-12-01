from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pickle
from transformers import DistilBertTokenizer
from model import TransactionClassifier
import os 

if not os.path.exists('transaction_classifier.pth') or not os.path.exists('label_map.pkl'):
    print("Model files not found. Downloading from Google Drive...")
    from download_model import download_files
    download_files()

app = FastAPI(title="Transaction Classification API")

MAX_LENGTH = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)

NUM_CLASSES = len(label_map)
inverse_label_map = {v: k for k, v in label_map.items()}

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

model = TransactionClassifier(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('transaction_classifier.pth', map_location=DEVICE))
model.eval()

class TransactionRequest(BaseModel):
    description: str

class TransactionResponse(BaseModel):
    category: str
    confidence: float

@app.get("/")
def read_root():
    return {
        "message": "Transaction Classification API",
        "status": "running",
        "available_categories": list(label_map.keys())
    }

@app.post("/classify", response_model=TransactionResponse)
def classify_transaction(request: TransactionRequest):
    try:
        encoding = tokenizer(
            request.description,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_label = inverse_label_map[predicted.item()]
        
        return TransactionResponse(
            category=predicted_label,
            confidence=float(confidence.item())
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}