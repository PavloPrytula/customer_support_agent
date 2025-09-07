# api.py
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F

import infer as inf

app = FastAPI(title="Text Classifier API", version="1.0.0")

# --------- Pydantic models ---------
class PredictRequest(BaseModel):
    text: str
    top_k: int = 3
    temp: float = 0.8
    threshold: float = 0.25
    margin: float = 0.10

class PredictBatchRequest(BaseModel):
    texts: List[str]
    top_k: int = 3
    temp: float = 0.8
    threshold: float = 0.25
    margin: float = 0.10

# --------- Helpers ---------
def _rank_probs_row(probs_row) -> List[tuple]:
    ranked = sorted(enumerate(probs_row), key=lambda x: x[1], reverse=True)
    return [(inf.labels[i], float(p)) for i, p in ranked]

def _predict_single(text: str, top_k: int, temp: float, threshold: float, margin: float) -> Dict[str, Any]:
    t_clean = inf.clean_text(text)
    enc = inf._encode([t_clean])

    with torch.no_grad():
        logits = inf.model(**enc).logits  # [1, C]
        probs = F.softmax(logits / max(temp, 1e-6), dim=-1)[0].detach().cpu().numpy()

    ranked = _rank_probs_row(probs)
    top = ranked[:max(1, top_k)]
    best_label, best_prob = top[0]
    p1 = top[0][1]
    p2 = top[1][1] if len(top) > 1 else 0.0
    is_low_conf = (p1 < threshold) or ((p1 - p2) < margin)

    return {
        "top": top,
        "best_label": best_label,
        "best_prob": float(p1),
        "is_low_conf": bool(is_low_conf),
        "margin": float(p1 - p2),
    }

def _predict_batch(texts: List[str], top_k: int, temp: float, threshold: float, margin: float) -> List[Dict[str, Any]]:
    cleaned = [inf.clean_text(t) for t in texts]
    enc = inf._encode(cleaned)

    with torch.no_grad():
        logits = inf.model(**enc).logits  # [B, C]
        probs = F.softmax(logits / max(temp, 1e-6), dim=-1).detach().cpu().numpy()

    results = []
    for row in probs:
        ranked = _rank_probs_row(row)
        top = ranked[:max(1, top_k)]
        p1 = top[0][1]
        p2 = top[1][1] if len(top) > 1 else 0.0
        is_low_conf = (p1 < threshold) or ((p1 - p2) < margin)
        results.append({
            "top": top,
            "best_label": top[0][0],
            "best_prob": float(p1),
            "is_low_conf": bool(is_low_conf),
            "margin": float(p1 - p2),
        })
    return results

# --------- Routes ---------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "device": str(getattr(inf, "DEVICE", "cpu"))}

@app.get("/labels")
def get_labels() -> Dict[str, Any]:
    return {"num_labels": len(inf.labels), "labels": inf.labels}

@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    """
    :returns:
      {
        "top": [(label, prob), ...],
        "best_label": str,
        "best_prob": float,
        "is_low_conf": bool,
        "margin": float
      }
    """
    return _predict_single(
        text=req.text,
        top_k=req.top_k,
        temp=req.temp,
        threshold=req.threshold,
        margin=req.margin,
    )

@app.post("/predict-batch")
def predict_batch(req: PredictBatchRequest) -> List[Dict[str, Any]]:
    return _predict_batch(
        texts=req.texts,
        top_k=req.top_k,
        temp=req.temp,
        threshold=req.threshold,
        margin=req.margin,
    )