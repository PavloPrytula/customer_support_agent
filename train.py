# train.py
import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import re
import numpy as np
from collections import Counter

from datasets import load_dataset
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
)

# ======================
# CONFIG
# ======================
MODEL_NAME = "xlm-roberta-base"          # опционально: "microsoft/mdeberta-v3-base"
MAX_LEN = 256
BATCH_SIZE = 1
EPOCHS = 8
LR = 2e-5
SEED = 42
OUTPUT_DIR = "model_xlmr"
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
EARLY_STOP_PATIENCE = 5
GRAD_ACC_STEPS = 32

USE_FOCAL_LOSS = True          # фокальный лосс
FOCAL_GAMMA = 1.5
USE_WEIGHTED_SAMPLER = True    # взвешенный сэмплинг train-датасета

# фиксированный порядок меток (совместим с генератором CSV)
LABEL_LIST = ["Other", "Refund request", "Technical issue"]

set_seed(SEED)

# ======================
# DATA
# ======================
data_files = {
    "train": "ML CSVs/train.csv",
    "validation": "ML CSVs/validation.csv",
    "test": "ML CSVs/test.csv",
}
raw = load_dataset("csv", data_files=data_files)

# NaN -> ""
def filln(batch):
    batch["text"] = [x if isinstance(x, str) else "" for x in batch["text"]]
    batch["label"] = [y if isinstance(y, str) else "Other" for y in batch["label"]]
    return batch
raw = raw.map(filln, batched=True)

# Лёгкая очистка: не ломаем *_TOK и любые скобки — оставляем как есть
def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = re.sub(r"https?://\S+", " ", t)        # URL → пробел (в CSV уже есть URL_TOK, но на всякий)
    t = re.sub(r"[“”]", '"', t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def preprocess(batch):
    batch["text"] = [clean_text(x) for x in batch["text"]]
    return batch

raw = raw.map(preprocess, batched=True)

# ======================
# LABEL MAPPING (стабильный)
# ======================
# убеждаемся, что все метки известны; лишние — в Other (на всякий)
def normalize_label(y: str) -> str:
    return y if y in LABEL_LIST else "Other"

raw = raw.map(lambda ex: {"label": normalize_label(ex["label"])})

label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for l, i in label2id.items()}

def encode_label(example):
    example["labels"] = label2id[example["label"]]
    return example

raw = raw.map(encode_label)

# ======================
# TOKENIZER
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tok(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=False,
        max_length=MAX_LEN,
    )

tokenized = raw.map(tok, batched=True, remove_columns=["text", "label"])

# ======================
# COLLATOR
# ======================
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ======================
# CLASS WEIGHTS (по train-сплиту)
# ======================
train_labels = list(tokenized["train"]["labels"])
freq = Counter(train_labels)
num_classes = len(LABEL_LIST)
total = len(train_labels)

cw = np.zeros(num_classes, dtype=np.float32)
for i in range(num_classes):
    fi = freq.get(i, 0)
    cw[i] = total / (num_classes * (fi if fi > 0 else 1))
class_weights = torch.tensor(cw, dtype=torch.float32)
class_weights = class_weights / class_weights.mean()
class_weights = torch.clamp(class_weights, max=5.0)

# ======================
# MODEL (+ инициализация bias лог-приорами классов)
# ======================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
)
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# лог-приоры (по train)
counts = np.array([freq.get(i, 0) for i in range(num_classes)], dtype=np.float32)
priors = counts / max(counts.sum(), 1.0)
log_priors = np.log(priors + 1e-8)
# разные головы у разных моделей
classification_head = None
for name in ["classifier", "score"]:
    if hasattr(model, name):
        classification_head = getattr(model, name)
        break

with torch.no_grad():
    if classification_head is not None and hasattr(classification_head, "out_proj") and hasattr(classification_head.out_proj, "bias"):
        classification_head.out_proj.bias.copy_(torch.tensor(log_priors))
    elif classification_head is not None and hasattr(classification_head, "bias") and classification_head.bias is not None:
        classification_head.bias.copy_(torch.tensor(log_priors))

# ======================
# METRICS
# ======================
def compute_metrics(eval_pred):
    logits, labels_np = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels_np, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels_np, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "macro_f1": f1, "macro_precision": p, "macro_recall": r}

# ======================
# FOCAL LOSS
# ======================
def focal_loss(logits, targets, alpha, gamma=1.5, eps=1e-8):
    """
    logits: [B, C], targets: [B], alpha: [C]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=eps, max=1.0)  # p_t
    at = alpha[targets]  # class-wise alpha
    loss = -at * ((1 - pt) ** gamma) * torch.log(pt)
    return loss.mean()

# ======================
# Trainer с кастомным лоссом
# ======================
class BalancedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_t = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        if USE_FOCAL_LOSS:
            loss = focal_loss(
                logits,
                labels_t.long(),
                alpha=class_weights.to(logits.device),
                gamma=FOCAL_GAMMA
            )
        else:
            loss_fct = CrossEntropyLoss(
                weight=class_weights.to(logits.device),
                label_smoothing=0.05
            )
            loss = loss_fct(logits, labels_t.long())
        return (loss, outputs) if return_outputs else loss

# ======================
# MPS memory cleanup (macOS)
# ======================
class MPSMemoryCallback(TrainerCallback):
    def __init__(self, every=100): self.every = every
    def on_step_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available() and state.global_step % self.every == 0:
            torch.mps.empty_cache()
        return control
    def on_evaluate(self, args, state, control, **kwargs):
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return control

# ======================
# TRAINING ARGS
# ======================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    eval_accumulation_steps=20,
    report_to="none",
    fp16=False,                           # включи bf16/fp16 при наличии CUDA
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    seed=SEED,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
)

trainer = BalancedTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,                 # важно: используем tokenizer=..., а не deprecated processing_class
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE), MPSMemoryCallback(every=5)],
)

# ======================
# WeightedRandomSampler для train
# ======================
if USE_WEIGHTED_SAMPLER:
    train_y = np.array(train_labels)
    sample_w = np.array([1.0 / (freq[int(c)] if freq[int(c)] > 0 else 1.0) for c in train_y], dtype=np.float32)
    sample_w = torch.tensor(sample_w, dtype=torch.float32)

    def get_train_dataloader_patched(self):
        sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    trainer.get_train_dataloader = get_train_dataloader_patched.__get__(trainer, type(trainer))

# ======================
# TRAIN
# ======================
trainer.train()

# ======================
# VALIDATION REPORT
# ======================
pred_val = trainer.predict(tokenized["validation"])
y_true_val = pred_val.label_ids
y_pred_val = pred_val.predictions.argmax(axis=1)
print("\nValidation classification report:")
print(classification_report(
    y_true_val, y_pred_val,
    target_names=LABEL_LIST,
    zero_division=0
))

# ======================
# TEST
# ======================
test_metrics = trainer.evaluate(tokenized["test"])
print("\nTest metrics:", test_metrics)

pred_test = trainer.predict(tokenized["test"])
y_true_test = pred_test.label_ids
y_pred_test = pred_test.predictions.argmax(axis=1)
print("\nTest classification report:")
print(classification_report(
    y_true_test, y_pred_test,
    target_names=LABEL_LIST,
    zero_division=0
))
cm = confusion_matrix(y_true_test, y_pred_test, labels=list(range(len(LABEL_LIST))))
print("\nTest confusion matrix (rows=true, cols=pred):\n", cm)

# ======================
# SAVE
# ======================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w", encoding="utf-8") as f:
    for name in LABEL_LIST:
        f.write(name + "\n")

print("Done. Model saved to", OUTPUT_DIR)