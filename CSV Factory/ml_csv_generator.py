# SPDX-License-Identifier: MIT
"""
Data preparation utility for support-message classification.

Features:
- Cleans/normalizes raw text, detects "Technical issue"/"Refund request" signals
- Relabels inconsistent rows, optional class rebalance to target proportions
- Stratified train/val/test split with robust fallback
- Deterministic pipeline (seedable) and CLI flags

Expected input columns:
- "Ticket Description" (str)    — message text
- "Ticket Type" (str)           — raw label (mapped to {"Technical issue","Refund request","Other"})
- "Product Purchased" (str)     — optional; expands "{product_purchased}" placeholder
"""

from __future__ import annotations
import os
import re
import json
import argparse
import random
import logging
from math import ceil
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# ====== SPECIAL TOKENS (должны совпадать с train.py) ======
AMOUNT  = "AMOUNT_TOK"
PRODUCT = "PRODUCT_TOK"
URL     = "URL_TOK"
EMAIL   = "EMAIL_TOK"
ERR     = "ERRMSG_TOK"
DEVICES = "DEVICES_TOK"

# ====== REFUND SIGNALS ======
INTENT_REFUND_RE = re.compile(
    r"\b("
    r"i\s+(?:want|would\s+like|need|request|ask)\s+(?:a\s+)?refund|"
    r"please\s+refund|refund\s+me|money\s*back|"
    r"cancel(?:\s+my)?\s+order|order\s+cancell?ation|"
    r"(?:charged|billed)\s+twice|double\s+charge(?:d)?|over\s*charge(?:d)?|overcharged"
    r")\b", re.I
)
REFUND_TERMS_RE = re.compile(
    r"\b("
    r"refund(?:ed|s|ing)?|"
    r"charge\s*back|chargeback|"
    r"rma|warranty\s+claim|restocking\s+fee|return\s+label|"
    r"not\s+as\s+described|wrong\s+item|missing\s+item|damaged\s+item"
    r")\b", re.I
)
RETURN_WORD_RE = re.compile(r"\breturn(?:s|ed|ing)?\b", re.I)
BILLING_SIDE = re.compile(r"\b(billing|payment|invoice|charge|transaction|card|credit|debit)\b", re.I)
BILLING_PROB = re.compile(r"\b(issue|error|problem|fail(?:ed|ure)?|dispute)\b", re.I)
EXCLUDE_RETURN_PHRASES = re.compile(
    r"\b("
    r"return\s+instructions?|return\s+policy|returns?\s+page|shipping\s+page|"
    r"return\s+address|how\s+to\s+return|contact\s+customer\s+service\s+to\s+return"
    r")\b", re.I
)

def _near(tokens, i_list, j_list, window=8):
    for i in i_list:
        for j in j_list:
            if abs(i - j) <= window:
                return True
    return False

def _token_idxs(tokens, words):
    s = set(words)
    return [i for i, t in enumerate(tokens) if t in s]

def has_refund(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if INTENT_REFUND_RE.search(text):
        return True
    if REFUND_TERMS_RE.search(text):
        return True
    if RETURN_WORD_RE.search(text):
        if not EXCLUDE_RETURN_PHRASES.search(text):
            toks = re.findall(r"[a-z']+", text.lower())
            if toks:
                ret_idxs   = _token_idxs(toks, {"return","returns","returned","returning"})
                item_idxs  = _token_idxs(toks, {"order","item","package","parcel"})
                verb_idxs  = _token_idxs(toks, {"cancel","refund","exchange"})
                if _near(toks, ret_idxs, item_idxs, window=6) or _near(toks, ret_idxs, verb_idxs, window=6):
                    return True
    # billing/payment … (issue/error/problem/dispute) в радиусе <= 8 слов
    if BILLING_SIDE.search(text) and BILLING_PROB.search(text):
        toks = re.findall(r"[a-z']+", text.lower())
        if toks:
            side = _token_idxs(toks, {"billing","payment","invoice","charge","transaction","card","credit","debit"})
            prob = _token_idxs(toks, {"issue","error","problem","fail","failed","failure","dispute"})
            if _near(toks, side, prob, window=8):
                return True
    return False

# ====== TECH SIGNALS ======
NEG_RE = re.compile(r"\b(no|not|never|cannot|can't|won't|does(?:n't)?|did(?:n't)?|failed|unable|stopp?ed|missing)\b", re.I)
STRONG_TECH_RE = re.compile(
    r"\b(?:"
    r"login|log\s?in|password|credential(?:s)?|2fa|mfa|otp|"
    r"(?:not\s+)?connect(?:ing|ion)?|disconnect(?:ing)?|no\s+signal|pair(?:ing)?|"
    r"wifi|bluetooth|network|"
    r"update(?:d|s)?|software\s+update|install(?:ation)?|firmware|driver(?:s)?|"
    r"crash(?:es|ed|ing)?|freeze(?:s|d|ing)?|flicker(?:ing)?|"
    r"overheat(?:ing)?|battery|bootloop|bsod|kernel\s+panic|safe\s+mode|"
    r"factory\s+reset|reset(?:ted|ting)?|"
    r"malfunction(?:s|ed|ing)?|unresponsive|"
    r"repair|replacement|warranty|rma|"
    r"account\s+locked|locked\s+account|unlock|"
    r"diagnostic(?:s)?|troubleshoot(?:ing)?|"
    r"configuration|settings|driver|firmware"
    r")\b",
    re.I,
)
WEAK_TECH_RE = re.compile(r"\b(?:issue(?:s)?|problem(?:s)?|error(?:s)?)\b", re.I)

def has_tech(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if STRONG_TECH_RE.search(text):
        return True
    if WEAK_TECH_RE.search(text) and NEG_RE.search(text):
        return True
    return False

# ====== CLEANING ======
DEMOGRAPHIC_RE = re.compile(r"\bi\s*am\s+a?n?\s*AMOUNT_TOK\s*year[-\s]*old\b.*?(male|female)?", re.I)
REPEAT_SPACE_RE = re.compile(r"\s{2,}")

JUNK_PATTERNS = [
    r"\bplease\s+assist\b",
    r"\bthank(s| you)\b",
    r"\bwe'?re\s+sorry\b",
    r"\bcreate\s+a\s+customer\s+account\b",
    r"\bpowered\s+by\s+vbulletin\b",
    r"\ball\s+rights\s+reserved\b",
    r"\bprivacy\s+policy\b",
    r"\bterms\s+of\s+service\b",
    r"(?:^|\s)(product\s+(name|id|purchase|rating)\s*:?.*?)(?=$|\.)",
    r"`{1,3}[^`]{0,400}`{1,3}",
    r"</?[^>]{1,200}>",
    r"(^|\s)[\-\*\•·●]\s+",
    r"\b0x[0-9a-fA-F]{4,}\b",
    r"([A-Za-z]:)?[\\/][\w\-/\.]+",
    r"@\w+|#\w+",
    r"^all requests .*",
    r"^please email .*",
    r"^if you have any questions.*",
]
JUNK_RE = re.compile("|".join(JUNK_PATTERNS), re.I)

def normalize_contractions(t: str) -> str:
    pairs = {
        r"\bI s\b": "it's", r"\bIt s\b": "It's", r"\bLet t\b": "Let's",
        r"\bI m\b": "I'm",  r"\bi m\b": "I'm", r"\bit s\b": "it's", r"\bisn t\b": "isn't",
        r"\bdon t\b": "don't", r"\bcan t\b": "can't", r"\bwon t\b": "won't",
        r"\bI ve\b": "I've", r"\bI d\b": "I'd", r"\byou re\b": "you're", r"\bwe re\b": "we're",
        r"\bthey re\b": "they're", r"\bI ll\b": "I'll", r"\bwe ll\b": "we'll",
    }
    for pat, sub in pairs.items():
        t = re.sub(pat, sub, t, flags=re.I)
    t = re.sub(r"\ban\s+(Lenovo|DEVICES_TOK)", r"a \1", t, flags=re.I)
    return t

def normalize_placeholders(t: str) -> str:
    t = re.sub(r"\+\s*product[_\s]*name\s*\+", PRODUCT, t, flags=re.I)
    t = re.sub(r"https?://\S+|\b\w+\.(com|net|org|io|co|ai)\b\S*", URL, t, flags=re.I)
    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", EMAIL, t)
    t = re.sub(r"\$?\b\d{1,3}(?:[,\.\s]\d{3})*(?:\.\d+)?\b", AMOUNT, t)
    t = re.sub(r"\berror(_message)?\b", ERR, t, flags=re.I)
    t = re.sub(r"\berror\s*:\s*[^\.\!\?]{3,80}", ERR, t, flags=re.I)
    return t

def strip_markup_code_lists(t: str) -> str:
    t = re.sub(r"</?[^>]+>", " ", t)
    t = re.sub(r"`{1,3}.*?`{1,3}", " ", t)
    t = re.sub(r"(^|\s)[\-\*\•·●]\s+", " ", t)
    t = re.sub(r"\b0x[0-9a-fA-F]{4,}\b", " ", t)
    t = re.sub(r"([A-Za-z]:)?[\\/][\w\-/\.]+", " ", t)
    t = re.sub(r"@\w+|#\w+", " ", t)
    t = re.sub(
        r"\b(iphone|ipad|android|windows phone|macbook|thinkpad|galaxy|pixel|"
        r"ipad pro|airpods|fitbit|gopro|echo|alexa)\b(,\s*\b[\w\s]+\b){0,6}",
        DEVICES, t, flags=re.I
    )
    return t

def split_sents(t: str):
    return [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", t) if s.strip()]

def select_signal_sents(t: str) -> str:
    sents = split_sents(t)
    if not sents:
        return ""
    tech_s = [s for s in sents if has_tech(s)]
    ref_s  = [s for s in sents if has_refund(s)]
    picked = []
    for s in tech_s:
        if s not in picked:
            picked.append(s)
        if len(picked) == 3:
            break
    if len(picked) < 3:
        for s in ref_s:
            if s not in picked:
                picked.append(s)
            if len(picked) == 3:
                break
    if len(picked) < 3:
        for s in sents:
            if s not in picked:
                picked.append(s)
            if len(picked) == 3:
                break
    return " ".join(picked[:3])

def strip_junk_with_score(text: str):
    if not isinstance(text, str):
        return "", 1.0
    orig_len = len(text)
    removed = []
    def _repl(m):
        removed.append(m.group(0))
        return " "
    cleaned = JUNK_RE.sub(_repl, text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    junk_chars = sum(len(x) for x in removed)
    junk_ratio = junk_chars / max(orig_len, 1)
    return cleaned, junk_ratio

def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = normalize_contractions(t)
    t = normalize_placeholders(t)
    t = strip_markup_code_lists(t)

    t = DEMOGRAPHIC_RE.sub(" ", t)
    t = re.sub(r"\b(thank(s| you)|we'?re sorry|please\s+(assist|help|advise))\b.*?$", " ", t, flags=re.I)

    t = re.sub(r"[“”]", "\"", t)
    t = re.sub(r"\s*([\,\.\!\?])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()

    t = select_signal_sents(t)

    t = re.sub(r"\b(contact\s+us|support\s+page|issue\s+tracker)\b.*?$", " ", t, flags=re.I)
    t = re.sub(r"\b(click\s+\"?submit\"?)\b.*?$", " ", t, flags=re.I)
    t = re.sub(r"\{[^}]{0,200}\}", " ", t)
    t = re.sub(r"\b(?:make|run|install|bin|pip|npm|yarn)[^\.\!\?]{0,80}", " ", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip()

    return t

def is_low_content(text: str) -> bool:
    if not text or len(text) < 30:
        return True
    letters = len(re.findall(r"[A-Za-zА-Яа-яЁё\s]", text))
    if letters / max(len(text), 1) < 0.7:
        return True
    words = re.findall(r"[A-Za-zА-Яа-яЁё']+", text)
    stop = {"the","a","an","and","or","if","to","of","in","on","for","by","with","that","this","it","is","are","was","were"}
    content = [w for w in words if w.lower() not in stop]
    if len(content) < 7:
        return True
    if len(re.findall(r"\b[A-Z]{3,}_TOK\b", text)) >= 4 and len(content) < 12:
        return True
    return False

def relabel_consistent(orig_label: str, text: str) -> str:
    if isinstance(orig_label, str):
        trimmed = orig_label.strip()
        if trimmed in ("Refund request", "Technical issue"):
            if trimmed == "Technical issue" and not has_tech(text):
                return "Other"
            if trimmed == "Refund request" and not has_refund(text):
                return "Other"
            return trimmed
    has_ref = has_refund(text)
    has_tec = has_tech(text)
    if has_ref and has_tec:
        return "Technical issue"
    if has_ref:
        return "Refund request"
    if has_tec:
        return "Technical issue"
    return "Other"

# ====== SPLIT & REBALANCE ======
def robust_three_way_split(df, test_size=0.30, val_from_temp=0.50, random_state=42):
    """70/15/15 by default via stratified split; uses non-stratified fallback if needed."""
    y = df["label"]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, temp_idx = next(sss.split(df, y))
    train = df.iloc[train_idx].copy()
    temp  = df.iloc[temp_idx].copy()
    y_temp = temp["label"]
    test_from_temp = 1.0 - val_from_temp
    if y_temp.value_counts().min() >= 2:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_from_temp, random_state=random_state)
        val_idx, test_idx = next(sss2.split(temp, y_temp))
        val  = temp.iloc[val_idx].copy()
        test = temp.iloc[test_idx].copy()
        fallback = False
    else:
        val, test = train_test_split(temp, test_size=test_from_temp, random_state=random_state, shuffle=True, stratify=None)
        fallback = True
    return train, val, test, fallback

def rebalance_to_props(df, props, random_state=42):
    """Re-sample dataframe to approximate desired class proportions."""
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    counts = df['label'].value_counts()
    smallest_label = counts.idxmin()
    smallest = counts.min()
    denom = max(props.get(smallest_label, 1e-6), 1e-6)
    target_total = max(len(df), ceil(smallest / denom))
    parts = []
    for lab, p in props.items():
        target_n = int(round(p * target_total))
        cur = df[df['label'] == lab]
        if len(cur) == 0:
            continue
        if len(cur) >= target_n:
            parts.append(cur.sample(n=target_n, random_state=random_state))
        else:
            parts.append(cur.sample(n=target_n, replace=True, random_state=random_state))
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out

# ---------------------- CLI / UTILS ----------------------
def parse_desired(desired_str: str) -> Dict[str, float]:
    """Parse 'label=p,label2=q,...' into dict, verifying sum≈1."""
    out: Dict[str, float] = {}
    for part in desired_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad desired format part: {part!r}. Use Label=0.6,Other=0.2,...")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    s = sum(out.values())
    if not (0.99 <= s <= 1.01):
        raise ValueError(f"Desired proportions must sum to 1.0 (got {s}).")
    return out

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def require_columns(df: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)[:10]}...")

def positive_int(value: str) -> int:
    iv = int(value)
    if iv <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return iv

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ml_csv_generator", description="Prepare CSVs for model training/validation/testing.")
    p.add_argument("--input", "-i", default="main.csv", help="Path to input CSV (default: main.csv)")
    p.add_argument("--outdir", "-o", default="ML CSVs", help="Output directory (default: 'ML CSVs')")
    p.add_argument("--max-len", type=positive_int, default=1200, help="Truncate text to this length (default: 1200)")
    p.add_argument("--test-size", type=float, default=0.30, help="Test+Val share (default: 0.30)")
    p.add_argument("--val-from-temp", type=float, default=0.50, help="Val share from temp (default: 0.50 → 15/15 split)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--desired", default="Technical issue=0.6,Refund request=0.2,Other=0.2",
                   help="Target class proportions summing to 1.0")
    p.add_argument("--no-rebalance", action="store_true", help="Disable class rebalance step")
    p.add_argument("--dry-run", action="store_true", help="Do everything but don't write CSVs")
    p.add_argument("--preview", type=int, default=0, help="Print up to N sample rows per split")
    p.add_argument("--report", default="", help="Path to save JSON report (metrics, counts, warnings)")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Verbosity: -v (info), -vv (debug), none=warn")
    return p

# ---------------------- MAIN ----------------------
def main():
    args = build_argparser().parse_args()
    setup_logging(args.verbose)
    set_global_seed(args.seed)

    desired = parse_desired(args.desired)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Read
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    df = pd.read_csv(args.input)
    require_columns(df, ["Ticket Description", "Ticket Type"])

    # 2) Placeholder expansion (optional col)
    if "Product Purchased" in df.columns:
        df["Ticket Description"] = df.apply(
            lambda r: str(r.get("Ticket Description", "")).replace("{product_purchased}", str(r.get("Product Purchased", ""))),
            axis=1
        )

    # 3) Map labels to 3 classes
    KEEP = {"Refund request", "Technical issue"}
    df["Ticket Type"] = df["Ticket Type"].apply(lambda x: x if str(x) in KEEP else "Other")

    # 4) Clean & filter
    before = len(df)
    df["Ticket Description"] = df["Ticket Description"].apply(clean_text)

    def keep_row(t):
        if not isinstance(t, str):
            return False
        if has_refund(t) or has_tech(t):
            return True
        return not is_low_content(t)

    df = df[df["Ticket Description"].apply(keep_row)].copy()
    after_quality = len(df)

    # 5) Consistent relabel by signals
    df["Ticket Type"] = df.apply(lambda r: relabel_consistent(r["Ticket Type"], r["Ticket Description"]), axis=1)

    # 6) Final shape
    df = df.rename(columns={"Ticket Type": "label", "Ticket Description": "text"})[["label", "text"]]
    df.drop_duplicates(subset=["text"], inplace=True)
    df = df[df["text"].str.len() > 0].copy()
    df["text"] = df["text"].str.slice(0, args.max_len)

    # 7) Fix remaining label inconsistencies
    mask_tech   = df["text"].apply(has_tech)
    mask_refund = df["text"].apply(has_refund)
    df.loc[(df["label"] == "Other") & mask_tech, "label"] = "Technical issue"
    df.loc[(df["label"] == "Other") & ~mask_tech & mask_refund, "label"] = "Refund request"

    leftover = df[(df["label"] == "Other") & df["text"].apply(has_tech)]
    warnings_list = []
    if len(leftover) > 0:
        msg = f"Leftover 'Other' with tech signals: {len(leftover)}"
        logging.warning(msg)
        warnings_list.append(msg)

    # 8) Optional rebalance
    if not args.no_rebalance:
        df = rebalance_to_props(df, desired, random_state=args.seed)

    # 9) Split
    train, val, test, fallback_used = robust_three_way_split(
        df, test_size=args.test_size, val_from_temp=args.val_from_temp, random_state=args.seed
    )

    # 10) Preview
    if args.preview > 0:
        for name, part in [("train", train), ("val", val), ("test", test)]:
            logging.info(f"Preview {name}:")
            for _, r in part.head(args.preview).iterrows():
                logging.info(f"  · [{r['label']}] {r['text'][:160].strip()}")

    # 11) Report
    def counts(df_):
        return df_["label"].value_counts(normalize=True).round(3).to_dict()

    report = {
        "input": os.path.abspath(args.input),
        "outdir": os.path.abspath(args.outdir),
        "rows_before": before,
        "rows_after_quality": after_quality,
        "fallback_used": bool(fallback_used),
        "counts": {
            "train": counts(train),
            "val": counts(val),
            "test": counts(test),
        },
        "warnings": warnings_list,
        "config": {
            "max_len": args.max_len,
            "test_size": args.test_size,
            "val_from_temp": args.val_from_temp,
            "seed": args.seed,
            "desired": desired,
            "rebalance": (not args.no_rebalance)
        }
    }

    # 12) Save
    train_path = os.path.join(args.outdir, "train.csv")
    val_path   = os.path.join(args.outdir, "validation.csv")
    test_path  = os.path.join(args.outdir, "test.csv")

    if not args.dry_run:
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

    # 13) Console summary
    logging.info(f"Dropped by quality filter: {before - after_quality}")
    if fallback_used:
        logging.warning("Stratified val/test split was not possible (min class < 2 in temp). Used non-stratified fallback.")
    logging.info(f"Train proportions: {report['counts']['train']}")
    logging.info(f"Val proportions:   {report['counts']['val']}")
    logging.info(f"Test proportions:  {report['counts']['test']}")
    if not args.dry_run:
        logging.info(f"Saved to:\n  {os.path.abspath(train_path)}\n  {os.path.abspath(val_path)}\n  {os.path.abspath(test_path)}")
    if args.report and not args.dry_run:
        logging.info(f"Report saved to: {os.path.abspath(args.report)}")

if __name__ == "__main__":
    main()