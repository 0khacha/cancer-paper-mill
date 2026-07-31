import json
import os
import sys
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding='utf-8')

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LENGTH = 512
BATCH_SIZE = 8

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

def evaluate_model(model_path, dataset_recs, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = AbstractDataset(dataset_recs, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    model = BERTClassifier(MODEL_NAME).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
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
        "auc": float(auc),
        "pr_auc": float(ap),
        "f1": float(best_f1),
        "threshold": float(best_thresh),
        "precision": float(prec_best),
        "recall": float(rec_best)
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    # 1. Load all evaluation splits (both original and normalized)
    splits = {}
    for name in ["val", "test", "holdout"]:
        # Original
        with open(f"data/final/cancer_pm_{name}.json", "r", encoding="utf-8") as f:
            splits[f"{name}_orig"] = json.load(f)
        # Normalized
        with open(f"data/final/cancer_pm_{name}_normalized.json", "r", encoding="utf-8") as f:
            splits[f"{name}_norm"] = json.load(f)
            
    # Pre-split subsets for controls
    # Hindawi holdout is already 100% Hindawi
    # For val and test: filter out Hindawi vs Non-Hindawi
    for suffix in ["orig", "norm"]:
        val_recs = splits[f"val_{suffix}"]
        test_recs = splits[f"test_{suffix}"]
        
        splits[f"val_{suffix}_hindawi"] = [r for r in val_recs if get_publisher(r) == 'Hindawi']
        splits[f"val_{suffix}_non_hindawi"] = [r for r in val_recs if get_publisher(r) != 'Hindawi']
        splits[f"test_{suffix}_hindawi"] = [r for r in test_recs if get_publisher(r) == 'Hindawi']
        splits[f"test_{suffix}_non_hindawi"] = [r for r in test_recs if get_publisher(r) != 'Hindawi']

    # 2. Define models to evaluate
    models_to_eval = [
        # (Model display name, Model path, Test set key, result key)
        
        # Pooled Models
        ("Pooled (Pre-Norm)", "models/pubmedbert_finetuned_best.pt", "val_orig", "val"),
        ("Pooled (Pre-Norm)", "models/pubmedbert_finetuned_best.pt", "test_orig", "test"),
        ("Pooled (Pre-Norm)", "models/pubmedbert_finetuned_best.pt", "holdout_orig", "holdout"),
        
        ("Pooled (Post-Norm)", "models/pubmedbert_normalized_best.pt", "val_norm", "val"),
        ("Pooled (Post-Norm)", "models/pubmedbert_normalized_best.pt", "test_norm", "test"),
        ("Pooled (Post-Norm)", "models/pubmedbert_normalized_best.pt", "holdout_norm", "holdout"),
        
        # Hindawi-Only Controls (evaluated on Hindawi holdout)
        ("Hindawi-Only (Pre-Norm)", "models/pubmedbert_orig_hindawi_only_best.pt", "holdout_orig", "holdout"),
        ("Hindawi-Only (Post-Norm)", "models/pubmedbert_norm_hindawi_only_best.pt", "holdout_norm", "holdout"),
        
        # Cross-Transfer: Train Non-Hindawi, Eval Hindawi Holdout
        ("Non-Hindawi Trained (Pre-Norm)", "models/pubmedbert_orig_non_hindawi_best.pt", "holdout_orig", "holdout"),
        ("Non-Hindawi Trained (Post-Norm)", "models/pubmedbert_norm_non_hindawi_best.pt", "holdout_norm", "holdout"),
        
        # Cross-Transfer: Train Hindawi-Only, Eval Non-Hindawi Test
        ("Hindawi Trained -> Non-Hindawi Test (Pre-Norm)", "models/pubmedbert_orig_hindawi_only_best.pt", "test_orig_non_hindawi", "test_non_hindawi"),
        ("Hindawi Trained -> Non-Hindawi Test (Post-Norm)", "models/pubmedbert_norm_hindawi_only_best.pt", "test_norm_non_hindawi", "test_non_hindawi"),
    ]
    
    results = {}
    
    for display_name, model_path, test_set_key, result_key in models_to_eval:
        if not os.path.exists(model_path):
            print(f"Skipping evaluation of {display_name} because model file {model_path} does not exist.")
            continue
            
        print(f"Evaluating model '{display_name}' on '{test_set_key}'...")
        recs = splits[test_set_key]
        res = evaluate_model(model_path, recs, device)
        print(f"  Result -> AUC: {res['auc']:.4f}, F1: {res['f1']:.4f}")
        
        if display_name not in results:
            results[display_name] = {}
        results[display_name][result_key] = res
        
    # Write results to json
    out_path = "models/all_evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll evaluation results written to {out_path}")

if __name__ == "__main__":
    main()
