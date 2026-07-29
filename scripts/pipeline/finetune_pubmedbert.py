import json
import os
import sys
import time
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Config
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LENGTH = 512
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2  # Effective batch size = 16
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
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (batch, hidden_size)
        logits = self.classifier(cls_embedding)
        return logits

def get_publisher(rec, journal_to_publisher, nlm2raw):
    j = rec['journal']
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() or "comput math methods med" in j.lower():
        pub = "Hindawi"
    return pub

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
    
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    
    # Standard threshold 0.5 metrics
    preds_05 = (all_probs >= 0.5).astype(int)
    f1_05 = f1_score(all_labels, preds_05, zero_division=0)
    
    # Search for best F1 threshold on validation
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
        "f1_05": f1_05,
        "best_f1": best_f1,
        "best_thresh": best_thresh,
        "precision": prec_best,
        "recall": rec_best,
        "probs": all_probs,
        "labels": all_labels
    }

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    
    # Load data
    print("Loading data...")
    with open("data/final/cancer_pm_train.json", "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open("data/final/cancer_pm_val.json", "r", encoding="utf-8") as f:
        val_records = json.load(f)
        
    # Load publisher dictionaries for stratification
    import csv
    journal_to_publisher = {}
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
    
    # Tokenizer & Datasets
    print(f"Initializing tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = AbstractDataset(train_records, tokenizer, MAX_LENGTH)
    val_dataset = AbstractDataset(val_records, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # Model
    model = BERTClassifier(MODEL_NAME).to(device)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRACTION)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda", enabled=True)
    
    # Ensure save directory exists
    os.makedirs("models", exist_ok=True)
    
    epoch_metrics = []
    
    # Training Loop
    print("\nStarting Fine-tuning...")
    t_start = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
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
            
            if step % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            n_batches += 1
            
        avg_loss = epoch_loss / max(n_batches, 1)
        
        # Evaluate after epoch
        val_res = evaluate(model, val_loader, device)
        
        epoch_metrics.append({
            "epoch": epoch,
            "loss": avg_loss,
            "auc": val_res["auc"],
            "best_f1": val_res["best_f1"],
            "best_thresh": val_res["best_thresh"],
            "f1_05": val_res["f1_05"]
        })
        
        print(f"Epoch {epoch}/{EPOCHS} complete:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val AUC:    {val_res['auc']:.4f}")
        print(f"  Val F1@0.5: {val_res['f1_05']:.4%}")
        print(f"  Val BestF1: {val_res['best_f1']:.4%} (threshold={val_res['best_thresh']:.2f})")
        
        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": val_res
        }, f"models/pubmedbert_epoch_{epoch}.pt")
        
    t_total = time.time() - t_start
    print(f"\nTraining completed in {t_total/60:.2f} minutes.")
    
    # Select best epoch by F1
    best_epoch_idx = np.argmax([m["best_f1"] for m in epoch_metrics])
    best_epoch = epoch_metrics[best_epoch_idx]["epoch"]
    print(f"\nSelecting Epoch {best_epoch} as the best model (F1={epoch_metrics[best_epoch_idx]['best_f1']:.4%}).")
    
    # Load best checkpoint
    checkpoint = torch.load(f"models/pubmedbert_epoch_{best_epoch}.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Run final evaluation
    val_res = evaluate(model, val_loader, device)
    best_thresh = val_res["best_thresh"]
    
    # Stratified breakdown
    print("\n" + "="*80)
    print("FINAL EVALUATION METRICS (Best Checkpoint) ON VALIDATION")
    print("="*80)
    print(f"Decision Threshold: {best_thresh:.2f}")
    print(f"Overall Precision:  {val_res['precision']:.4%}")
    print(f"Overall Recall:     {val_res['recall']:.4%}")
    print(f"Overall F1 Score:   {val_res['best_f1']:.4%}")
    print(f"Overall ROC-AUC:    {val_res['auc']:.4%}")
    print(f"Overall PR-AUC:     {val_res['pr_auc']:.4%}")
    
    # Compute stratified metrics
    val_probs = val_res["probs"]
    val_labels = val_res["labels"]
    
    hindawi_probs, hindawi_labels = [], []
    spandidos_probs, spandidos_labels = [], []
    others_probs, others_labels = [], []
    
    for idx, rec in enumerate(val_records):
        pub = get_publisher(rec, journal_to_publisher, nlm2raw)
        prob = val_probs[idx]
        lbl = val_labels[idx]
        
        if pub == "Hindawi":
            hindawi_probs.append(prob)
            hindawi_labels.append(lbl)
        elif pub == "Spandidos":
            spandidos_probs.append(prob)
            spandidos_labels.append(lbl)
        else:
            others_probs.append(prob)
            others_labels.append(lbl)
            
    def get_stratified_metrics(lbls, prbs, thresh, name):
        lbls = np.array(lbls)
        prbs = np.array(prbs)
        preds = (prbs >= thresh).astype(int)
        
        f1 = f1_score(lbls, preds, zero_division=0)
        prec = precision_score(lbls, preds, zero_division=0)
        rec = recall_score(lbls, preds, zero_division=0)
        auc = roc_auc_score(lbls, prbs) if len(np.unique(lbls)) > 1 else 0.5
        
        n_pos = int(lbls.sum())
        n_neg = len(lbls) - n_pos
        
        print(f"Strate {name:<15} (Size: {n_pos} pos, {n_neg} neg):")
        print(f"  F1 Score:   {f1:.4%}")
        print(f"  Precision:  {prec:.4%}")
        print(f"  Recall:     {rec:.4%}")
        print(f"  ROC-AUC:    {auc:.4%}")
        return {"f1": f1, "prec": prec, "rec": rec, "auc": auc, "n_pos": n_pos, "n_neg": n_neg}

    print("\n--- STRATIFIED BREAKDOWN ---")
    h_m = get_stratified_metrics(hindawi_labels, hindawi_probs, best_thresh, "Hindawi")
    s_m = get_stratified_metrics(spandidos_labels, spandidos_probs, best_thresh, "Spandidos")
    o_m = get_stratified_metrics(others_labels, others_probs, best_thresh, "Pooled Others")
    
    # Save best predictions and metrics
    save_results = {
        "epoch_history": epoch_metrics,
        "best_epoch": best_epoch,
        "best_threshold": best_thresh,
        "overall": {
            "precision": val_res["precision"],
            "recall": val_res["recall"],
            "f1": val_res["best_f1"],
            "auc": val_res["auc"],
            "pr_auc": val_res["pr_auc"]
        },
        "stratified": {
            "hindawi": h_m,
            "spandidos": s_m,
            "others": o_m
        }
    }
    
    # Serialize results to json
    with open("models/pubmedbert_finetuned_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print("\nResults saved to models/pubmedbert_finetuned_results.json")
    
    # Copy checkpoint to models/pubmedbert_finetuned_best.pt
    import shutil
    shutil.copyfile(f"models/pubmedbert_epoch_{best_epoch}.pt", "models/pubmedbert_finetuned_best.pt")
    print("Best model copied to models/pubmedbert_finetuned_best.pt")

if __name__ == "__main__":
    main()
