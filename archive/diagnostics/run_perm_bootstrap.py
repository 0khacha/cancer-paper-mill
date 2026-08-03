import json
import random
import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from scipy.stats import bootstrap

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

class AbstractDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        text = (rec.get("title", "") or "") + " " + (rec.get("abstract", "") or "")
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {"input_ids": encoding["input_ids"].squeeze(0), "attention_mask": encoding["attention_mask"].squeeze(0), "label": torch.tensor(int(rec["label"]), dtype=torch.float32)}

class BERTClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, 1))
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            with autocast("cuda", enabled=True):
                logits = model(input_ids, mask)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_probs), np.array(all_labels)

def train_permutation(seed, train_records, val_records, test_records, holdout_records):
    print(f"Starting training for seed {seed}...", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
    # Shuffle train labels
    labels = [r["label"] for r in train_records]
    random.shuffle(labels)
    perm_train = [{"title": r.get("title"), "abstract": r.get("abstract"), "label": l} for r, l in zip(train_records, labels)]
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(AbstractDataset(perm_train, tokenizer), batch_size=2, shuffle=True)
    val_loader = DataLoader(AbstractDataset(val_records, tokenizer), batch_size=2, shuffle=False)
    test_loader = DataLoader(AbstractDataset(test_records, tokenizer), batch_size=2, shuffle=False)
    holdout_loader = DataLoader(AbstractDataset(holdout_records, tokenizer), batch_size=2, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda", enabled=True)
    
    total_steps = (len(train_loader) // 8) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*0.1), total_steps)
    
    for epoch in range(1, 4):
        print(f"Epoch {epoch}/3...", flush=True)
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            with autocast("cuda", enabled=True):
                logits = model(input_ids, mask)
                loss = criterion(logits.squeeze(-1), labels) / 8
            scaler.scale(loss).backward()
            if step % 8 == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
    print(f"Generating predictions for seed {seed}...", flush=True)
    val_p, val_y = get_predictions(model, val_loader, device)
    test_p, test_y = get_predictions(model, test_loader, device)
    holdout_p, holdout_y = get_predictions(model, holdout_loader, device)
    
    return (val_p, val_y), (test_p, test_y), (holdout_p, holdout_y)

def compute_combined_bootstrap_ci(probs_1, y_1, probs_2, y_2, n_boot=2000):
    mean_probs = (probs_1 + probs_2) / 2.0
    n = len(y_1)
    boot_aucs = []
    np.random.seed(42)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_1[idx])) > 1:
            boot_aucs.append(roc_auc_score(y_1[idx], mean_probs[idx]))
    
    ci_lo = np.percentile(boot_aucs, 2.5)
    ci_hi = np.percentile(boot_aucs, 97.5)
    return roc_auc_score(y_1, mean_probs), ci_lo, ci_hi

def main():
    print("Loading data...", flush=True)
    with open("data/final_v2_period_stripped/cancer_pm_train.json", "r", encoding="utf-8") as f: train_records = json.load(f)
    with open("data/final_v2_period_stripped/cancer_pm_val.json", "r", encoding="utf-8") as f: val_records = json.load(f)
    with open("data/final_v2_period_stripped/cancer_pm_test.json", "r", encoding="utf-8") as f: test_records = json.load(f)
    with open("data/final_v2_period_stripped/cancer_pm_holdout.json", "r", encoding="utf-8") as f: holdout_records = json.load(f)
    
    torch.cuda.empty_cache()
    (v_p1, v_y1), (t_p1, t_y1), (h_p1, h_y1) = train_permutation(123, train_records, val_records, test_records, holdout_records)
    torch.cuda.empty_cache()
    (v_p2, v_y2), (t_p2, t_y2), (h_p2, h_y2) = train_permutation(456, train_records, val_records, test_records, holdout_records)
    
    print("Computing CIs...", flush=True)
    v_auc, v_lo, v_hi = compute_combined_bootstrap_ci(v_p1, v_y1, v_p2, v_y2)
    t_auc, t_lo, t_hi = compute_combined_bootstrap_ci(t_p1, t_y1, t_p2, t_y2)
    h_auc, h_lo, h_hi = compute_combined_bootstrap_ci(h_p1, h_y1, h_p2, h_y2)
    
    print(f"Val Bootstrap CI: {v_auc*100:.2f}% [{v_lo*100:.2f}%, {v_hi*100:.2f}%]", flush=True)
    print(f"Test Bootstrap CI: {t_auc*100:.2f}% [{t_lo*100:.2f}%, {t_hi*100:.2f}%]", flush=True)
    print(f"Holdout Bootstrap CI: {h_auc*100:.2f}% [{h_lo*100:.2f}%, {h_hi*100:.2f}%]", flush=True)

if __name__ == "__main__": main()
