import json
import os
import sys
import time
import random
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding='utf-8')

# Config
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
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32)
        }

class BERTClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.encoder.config.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits

def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
            with autocast("cuda", enabled=True):
                logits = model(input_ids, mask)
            
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    ap = average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    preds_best = (all_probs >= best_thresh).astype(int)
    prec_best = precision_score(all_labels, preds_best, zero_division=0)
    rec_best = recall_score(all_labels, preds_best, zero_division=0)
    
    return {
        "auc": auc,
        "pr_auc": ap,
        "best_f1": best_f1,
        "best_thresh": best_thresh,
        "precision": prec_best,
        "recall": rec_best,
        "probs": all_probs,
        "labels": all_labels
    }

# Helper to load publisher info
journal_to_publisher = {}
if os.path.exists('data/raw/rwdb/retraction_watch.csv'):
    with open('data/raw/rwdb/retraction_watch.csv', 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            j = row.get('Journal', '').strip()
            p = row.get('Publisher', '').strip()
            if j and p:
                if "computational and mathematical methods in medicine" in j.lower():
                    p = "Hindawi"
                elif "biomed research international" in j.lower():
                    p = "Hindawi"
                journal_to_publisher[j] = p
                
with open('data/final/journal_to_nlm.json', 'r', encoding='utf-8') as f:
    j2nlm = json.load(f)
nlm2raw = {v: k for k, v in j2nlm.items()}

def get_publisher(rec):
    j = rec['journal']
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() or "comput math methods med" in j.lower():
        pub = "Hindawi"
    elif "biomed research international" in raw_j.lower() or "biomed res int" in j.lower():
        pub = "Hindawi"
    return pub

def train_control_model(train_recs, val_recs, save_name, device):
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = AbstractDataset(train_recs, tokenizer, MAX_LENGTH)
    val_dataset = AbstractDataset(val_recs, tokenizer, MAX_LENGTH)
    
    # Adjust batch size for very small training sets if necessary, but keep hyperparameters same
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    model = BERTClassifier(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRACTION)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda", enabled=True)
    
    best_f1 = 0.0
    best_metrics = None
    best_state_dict = None
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            with autocast("cuda", enabled=True):
                logits = model(input_ids, mask)
                loss = criterion(logits.squeeze(-1), labels)
                loss = loss / GRAD_ACCUM_STEPS
                
            scaler.scale(loss).backward()
            
            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
        # Evaluate
        val_res = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch}/{EPOCHS} -> Val AUC: {val_res['auc']:.4f}, Val F1: {val_res['best_f1']:.4f}")
        
        if val_res["best_f1"] >= best_f1:
            best_f1 = val_res["best_f1"]
            best_metrics = val_res
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Save the best model state dict
    save_path = f"models/{save_name}_best.pt"
    torch.save({
        "model_state_dict": best_state_dict,
        "metrics": {k: v for k, v in best_metrics.items() if k not in ["probs", "labels"]}
    }, save_path)
    print(f"  Saved best model to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training controls on device: {device}")
    
    # 1. Load datasets
    # Pre-normalized
    with open("data/final/cancer_pm_train.json", "r", encoding="utf-8") as f:
        train_orig = json.load(f)
    with open("data/final/cancer_pm_val.json", "r", encoding="utf-8") as f:
        val_orig = json.load(f)
        
    # Normalized
    with open("data/final/cancer_pm_train_normalized.json", "r", encoding="utf-8") as f:
        train_norm = json.load(f)
    with open("data/final/cancer_pm_val_normalized.json", "r", encoding="utf-8") as f:
        val_norm = json.load(f)
        
    # 2. Filter datasets for Hindawi and Non-Hindawi
    # Hindawi subset
    train_orig_h = [r for r in train_orig if get_publisher(r) == 'Hindawi']
    val_orig_h = [r for r in val_orig if get_publisher(r) == 'Hindawi']
    train_norm_h = [r for r in train_norm if get_publisher(r) == 'Hindawi']
    val_norm_h = [r for r in val_norm if get_publisher(r) == 'Hindawi']
    
    # Non-Hindawi subset
    train_orig_nh = [r for r in train_orig if get_publisher(r) != 'Hindawi']
    val_orig_nh = [r for r in val_orig if get_publisher(r) != 'Hindawi']
    train_norm_nh = [r for r in train_norm if get_publisher(r) != 'Hindawi']
    val_norm_nh = [r for r in val_norm if get_publisher(r) != 'Hindawi']
    
    # Check sizes
    print(f"Original Hindawi train size: {len(train_orig_h)}, val: {len(val_orig_h)}")
    print(f"Normalized Hindawi train size: {len(train_norm_h)}, val: {len(val_norm_h)}")
    print(f"Original Non-Hindawi train size: {len(train_orig_nh)}, val: {len(val_orig_nh)}")
    print(f"Normalized Non-Hindawi train size: {len(train_norm_nh)}, val: {len(val_norm_nh)}")
    
    # 3. Train all 4 control models
    # Control 1: Pre-normalized, Hindawi-only
    print("\n--- Training Control 1: Pre-normalized Hindawi-only ---")
    train_control_model(train_orig_h, val_orig_h, "pubmedbert_orig_hindawi_only", device)
    
    # Control 2: Normalized, Hindawi-only
    print("\n--- Training Control 2: Normalized Hindawi-only ---")
    train_control_model(train_norm_h, val_norm_h, "pubmedbert_norm_hindawi_only", device)
    
    # Control 3: Pre-normalized, Non-Hindawi only
    print("\n--- Training Control 3: Pre-normalized Non-Hindawi only ---")
    train_control_model(train_orig_nh, val_orig_nh, "pubmedbert_orig_non_hindawi", device)
    
    # Control 4: Normalized, Non-Hindawi only
    print("\n--- Training Control 4: Normalized Non-Hindawi only ---")
    train_control_model(train_norm_nh, val_norm_nh, "pubmedbert_norm_non_hindawi", device)

if __name__ == "__main__":
    main()
