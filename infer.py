# infer.py
import os
import re
import sys
import argparse
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==== CONFIG ====
MODEL_DIR = "model_xlmr"
MAX_LEN = 256
T = 0.8
THRESHOLD = 0.25
MARGIN = 0.10

# ==== DEVICE ====
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ==== LOAD ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
model.to(DEVICE)

id2label = {int(k): v for k, v in model.config.id2label.items()}
labels = [id2label[i] for i in range(len(id2label))]

# ==== CLEANING ====
def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = re.sub(r'https?://\S+', ' ', t)
    t = re.sub(r'[“”]', '"', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# ==== CORE ====
def _encode(texts: List[str]):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
    )
    return {k: v.to(DEVICE) for k, v in enc.items()}

def _postprocess_probs(probs_row):
    ranked = sorted(enumerate(probs_row), key=lambda x: x[1], reverse=True)
    return [(labels[i], float(p)) for i, p in ranked]

def predict(text: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Returns:
      {
        "top": [(label, prob), ...],  # top_k
        "best_label": str,
        "best_prob": float,
        "is_low_conf": bool,          # best_prob < THRESHOLD or (p1-p2)<MARGIN
        "margin": float               # p1 - p2
      }
    """
    text = clean_text(text)
    enc = _encode([text])

    with torch.no_grad():
        logits = model(**enc).logits   # [1, C]
        probs = F.softmax(logits / T, dim=-1)[0].detach().cpu().numpy()

    ranked = _postprocess_probs(probs)
    top = ranked[:max(1, top_k)]
    best_label, best_prob = top[0]
    p1 = top[0][1]
    p2 = top[1][1] if len(top) > 1 else 0.0
    margin = p1 - p2
    is_low_conf = (best_prob < THRESHOLD) or (margin < MARGIN)

    return {
        "top": top,
        "best_label": best_label,
        "best_prob": best_prob,
        "is_low_conf": bool(is_low_conf),
        "margin": float(margin),
    }

def predict_batch(texts: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    texts = [clean_text(t) for t in texts]
    enc = _encode(texts)

    with torch.no_grad():
        logits = model(**enc).logits            # [B, C]
        probs = F.softmax(logits / T, dim=-1).detach().cpu().numpy()

    results = []
    for row in probs:
        ranked = _postprocess_probs(row)
        top = ranked[:max(1, top_k)]
        p1 = top[0][1]
        p2 = top[1][1] if len(top) > 1 else 0.0
        margin = p1 - p2
        is_low_conf = (p1 < THRESHOLD) or (margin < MARGIN)
        results.append({
            "top": top,
            "best_label": top[0][0],
            "best_prob": p1,
            "is_low_conf": bool(is_low_conf),
            "margin": float(margin),
        })
    return results

# ==== CSV UTILS ====
def run_csv(input_path: str, output_path: str, text_col: str = "text", top_k: int = 3):
    import csv
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if text_col not in reader.fieldnames:
            raise ValueError(f"Column '{text_col}' not found. Available: {reader.fieldnames}")
        for r in reader:
            rows.append(r)

    texts = [r.get(text_col, "") for r in rows]
    preds = predict_batch(texts, top_k=top_k)

    fieldnames = list(rows[0].keys()) if rows else [text_col]
    extra_cols = ["pred_label", "pred_prob", "low_conf", "margin"]
    extra_cols.append("topk")
    out_fields = fieldnames + extra_cols

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for r, p in zip(rows, preds):
            row = dict(r)
            row["pred_label"] = p["best_label"]
            row["pred_prob"]  = f'{p["best_prob"]:.4f}'
            row["low_conf"]   = int(p["is_low_conf"])
            row["margin"]     = f'{p["margin"]:.4f}'
            row["topk"]       = "; ".join([f"{lab}:{prob:.4f}" for lab, prob in p["top"]])
            writer.writerow(row)

    print(f"Saved predictions to {output_path}")

# ==== CLI ====
def main():
    global THRESHOLD, MARGIN, T

    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="Raw text to classify")
    ap.add_argument("--csv", type=str, help="Path to input CSV with a 'text' column (or set --text-col)")
    ap.add_argument("--text-col", type=str, default="text", help="Column name with text (for --csv)")
    ap.add_argument("--out", type=str, default="predictions.csv", help="Output CSV path (for --csv)")
    ap.add_argument("--top-k", type=int, default=3, help="Top-K probabilities to output")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="Low-confidence threshold for top-1 prob")
    ap.add_argument("--margin", type=float, default=MARGIN, help="Low-confidence margin = p1 - p2")
    ap.add_argument("--temp", type=float, default=T, help="Softmax temperature")
    args = ap.parse_args()

    THRESHOLD = args.threshold
    MARGIN = args.margin
    T = args.temp

    if args.csv:
        run_csv(args.csv, args.out, text_col=args.text_col, top_k=args.top_k)
        return

    if args.text:
        txt = args.text
    else:
        if not sys.stdin.isatty():
            txt = sys.stdin.read()
        else:
            txt = ("I'm having an issue with the Samsung Soundbar. "
                   "I've checked the device settings and made sure that everything is configured correctly.")
    res = predict(txt, top_k=args.top_k)
    print(res)

if __name__ == "__main__":
    main()