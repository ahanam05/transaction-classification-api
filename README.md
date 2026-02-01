# Transaction Classifier API

A small FastAPI service that classifies transaction descriptions into everyday spending categories using a fine‑tuned DistilBERT model.

## What this does
Given a short text like “uber to airport” or “paid rent for this month”, the API predicts a category (e.g., transportation, housing) and returns a confidence score.

The service loads a trained model and label map at startup, downloading them from Google Drive if they’re not present.

## Project layout
- [app.py](app.py) – FastAPI app and endpoints ([`app.classify_transaction`](app.py), [`app.read_root`](app.py), [`app.health_check`](app.py))
- [model.py](model.py) – PyTorch model definition ([`model.TransactionClassifier`](model.py))
- [download_model.py](download_model.py) – Model download helper ([`download_model.download_files`](download_model.py))
- [Dockerfile](Dockerfile) – Container build
- [requirements.txt](requirements.txt) – Python dependencies

## Quick start (local)

```bash
pip install -r requirements.txt
python app.py
```

Then open:

- `GET /` for a basic status response  
- `GET /health` for health checks  
- `POST /classify` to classify a transaction

### Example request

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"description":"uber to airport"}'
```

### Example response

```json
{
  "category": "transportation",
  "confidence": 0.93
}
```

## Running with Docker

```bash
docker build -t transaction-classifier .
docker run -p 8000:8000 transaction-classifier
```

The container downloads the model and label map during build via [`download_model.download_files`](download_model.py).

## Model details (from Google Colab)

The model was trained in Google Colab using DistilBERT (`distilbert-base-uncased`) with a lightweight classification head. The notebook used a curated dataset of short transaction descriptions across categories like **housing**, **transportation**, **investments**, **food**, **utilities**, **entertainment**, **healthcare**, and **miscellaneous**.

**Training setup:**
- Model: DistilBERT + dropout (0.3) + linear classifier
- Max sequence length: 32
- Optimizer: AdamW
- Learning rate: $2\times10^{-5}$
- Batch size: 8
- Epochs: 10
- Train/test split: 80/20

**Artifacts produced in Colab:**
- `transaction_classifier.pth` – model weights
- `label_map.pkl` – label map used at inference

These files are downloaded by the API on startup if missing and loaded in [`app.py`](app.py) using [`model.TransactionClassifier`](model.py).

## Notes
- The dataset is small and synthetic, designed to cover common transaction patterns.

## License
MIT