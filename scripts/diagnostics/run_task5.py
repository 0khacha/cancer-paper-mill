import json
import re
import unicodedata
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import random
import time
import shutil

# --- 1. NORMALIZATION ---
def aggressive_normalize(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    headers = ["materials and methods", "patients and methods", "background", "objective", "objectives", "aim", "aims", "introduction", "purpose", "method", "methods", "result", "results", "conclusion", "conclusions", "discussion", "significance", "design"]
    headers_pat = "|".join(headers)
    header_regex = re.compile(rf'\b({headers_pat})\b\s*:', re.IGNORECASE)
    text = header_regex.sub(' ', text)
    text = re.sub(r'[.,;:!?\"\'\(\)\[\]\{\}\-_/\\|]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_split(in_path, out_path):
    print(f"Aggressively normalizing {in_path} -> {out_path}...")
    with open(in_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    new_records = []
    for r in records:
        r_new = r.copy()
        r_new["title"] = aggressive_normalize(r.get("title", ""))
        r_new["abstract"] = aggressive_normalize(r.get("abstract", ""))
        new_records.append(r_new)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_records, f, indent=2, ensure_ascii=False)
    return new_records

# --- 2. TRAINING SETUP ---
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LENGTH = 512
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
EPOCHS = 3
SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AbstractDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        title = rec.get("title", "") or ""
        abstract = rec.get("abstract", "") or ""
        text = title + " " + abstract
        label = int(rec["label"])
        return {
            "input_ids": self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")["input_ids"].squeeze(0),
            "attention_mask": self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
            "pmid": rec.get("pmid", str(idx))
        }

class BERTClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, 1))

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    pmids = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            with autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(input_ids, mask)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            pmids.extend(batch["pmid"])
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return {"auc": auc, "best_f1": best_f1, "best_thresh": best_thresh, "probs": all_probs, "labels": all_labels, "pmids": pmids}

def train_model(train_records, val_records, out_path, device):
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(AbstractDataset(train_records, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AbstractDataset(val_records, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    
    model = BERTClassifier(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_FRACTION), total_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
    
    best_f1 = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            with autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(input_ids, mask)
                loss = criterion(logits.squeeze(-1), labels) / GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            if step % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
        val_res = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: Val AUC={val_res['auc']:.4f}, F1={val_res['best_f1']:.4f}")
        
        if val_res['best_f1'] > best_f1 or epoch == 1:
            best_f1 = val_res['best_f1']
            torch.save({"model_state_dict": model.state_dict()}, out_path)
            
    print(f"Training complete. Best model saved to {out_path}")
    return out_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Normalize datasets
    splits = ["train", "val", "holdout"]
    aggr_paths = {}
    orig_paths = {}
    for s in splits:
        in_p = f"data/final/cancer_pm_{s}.json"
        out_p = f"data/final/cancer_pm_{s}_aggressive.json"
        aggr_paths[s] = out_p
        orig_paths[s] = in_p
        normalize_split(in_p, out_p)
        
    with open(orig_paths["train"], "r", encoding="utf-8") as f: train_orig = json.load(f)
    with open(orig_paths["val"], "r", encoding="utf-8") as f: val_orig = json.load(f)
    with open(aggr_paths["train"], "r", encoding="utf-8") as f: train_aggr = json.load(f)
    with open(aggr_paths["val"], "r", encoding="utf-8") as f: val_aggr = json.load(f)
    
    # 2. Train Control
    print("\n--- Training Control Model ---")
    control_model_path = "models/pubmedbert_control_best.pt"
    train_model(train_orig, val_orig, control_model_path, device)
    
    # 3. Train Aggressive (Test)
    print("\n--- Training Aggressive Model ---")
    aggr_model_path = "models/pubmedbert_aggressive_best.pt"
    train_model(train_aggr, val_aggr, aggr_model_path, device)
    
    # 4. Evaluate on Holdout
    print("\n--- Evaluating on Holdout ---")
    with open(orig_paths["holdout"], "r", encoding="utf-8") as f: holdout_orig = json.load(f)
    with open(aggr_paths["holdout"], "r", encoding="utf-8") as f: holdout_aggr = json.load(f)
    
    # Filter only Hindawi for the report
    def get_publisher(rec):
        j = rec.get('journal', '').lower()
        if "computational and mathematical methods in medicine" in j or "biomed research international" in j:
            return "Hindawi"
        return "Other"
        
    hindawi_orig = [r for r in holdout_orig if get_publisher(r) == "Hindawi"]
    hindawi_aggr = [r for r in holdout_aggr if get_publisher(r) == "Hindawi"]
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Evaluate Control
    print("\nControl Model on Hindawi Holdout (Original text):")
    model_ctrl = BERTClassifier(MODEL_NAME).to(device)
    model_ctrl.load_state_dict(torch.load(control_model_path)["model_state_dict"])
    loader_ctrl = DataLoader(AbstractDataset(hindawi_orig, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    res_ctrl = evaluate(model_ctrl, loader_ctrl, device)
    print(f"AUC: {res_ctrl['auc']:.4f}, F1: {res_ctrl['best_f1']:.4f}")
    
    # Evaluate Aggressive
    print("\nAggressive Model on Hindawi Holdout (Aggressive text):")
    model_aggr = BERTClassifier(MODEL_NAME).to(device)
    model_aggr.load_state_dict(torch.load(aggr_model_path)["model_state_dict"])
    loader_aggr = DataLoader(AbstractDataset(hindawi_aggr, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    res_aggr = evaluate(model_aggr, loader_aggr, device)
    print(f"AUC: {res_aggr['auc']:.4f}, F1: {res_aggr['best_f1']:.4f}")
    
    # 5. Save 40 individual predictions
    print("\n--- Saving 40 individual predictions ---")
    with open(r"C:\Users\moham\.gemini\antigravity\brain\c22a0783-bc26-4f1e-820b-3c6515e0eaa8\scratch\task0_sample.json", "r", encoding="utf-8") as f:
        samples_40 = json.load(f)
        
    # Get original and aggressive versions for these 40
    pmids_40 = [s["pmid"] for s in samples_40]
    
    # Run control on original 40
    loader_40_ctrl = DataLoader(AbstractDataset(samples_40, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    res_40_ctrl = evaluate(model_ctrl, loader_40_ctrl, device)
    
    # Run aggressive on aggressive 40
    samples_40_aggr = [r for r in holdout_aggr if r["pmid"] in pmids_40]
    # Ensure order is same
    samples_40_aggr_dict = {r["pmid"]: r for r in samples_40_aggr}
    samples_40_aggr_ordered = [samples_40_aggr_dict.get(pmid) for pmid in pmids_40]
    # Some might be missing if there was a pmid mismatch, but shouldn't be
    samples_40_aggr_ordered = [x for x in samples_40_aggr_ordered if x is not None]
    
    loader_40_aggr = DataLoader(AbstractDataset(samples_40_aggr_ordered, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    res_40_aggr = evaluate(model_aggr, loader_40_aggr, device)
    
    # Combine into CSV
    df_ctrl = pd.DataFrame({"pmid": res_40_ctrl["pmids"], "label": res_40_ctrl["labels"], "prob_control": res_40_ctrl["probs"]})
    df_aggr = pd.DataFrame({"pmid": res_40_aggr["pmids"], "label": res_40_aggr["labels"], "prob_aggressive": res_40_aggr["probs"]})
    
    df_combined = pd.merge(df_ctrl, df_aggr, on=["pmid", "label"], how="inner")
    
    out_csv = r"C:\Users\moham\.gemini\antigravity\brain\c22a0783-bc26-4f1e-820b-3c6515e0eaa8\scratch\task5_predictions.csv"
    df_combined.to_csv(out_csv, index=False)
    print(f"Saved combined 40 predictions to {out_csv}")

if __name__ == "__main__":
    main()
