import json
import random
import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score

# Config
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 3
SEED = 456

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

def evaluate_auc(model, loader, device):
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
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    return auc

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    with open("data/final_v2_period_stripped/cancer_pm_train.json", "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open("data/final_v2_period_stripped/cancer_pm_val.json", "r", encoding="utf-8") as f:
        val_records = json.load(f)
    with open("data/final_v2_period_stripped/cancer_pm_test.json", "r", encoding="utf-8") as f:
        test_records = json.load(f)
    with open("data/final_v2_period_stripped/cancer_pm_holdout.json", "r", encoding="utf-8") as f:
        holdout_records = json.load(f)
        
    # Shuffle training labels
    labels = [r["label"] for r in train_records]
    random.shuffle(labels)
    for r, l in zip(train_records, labels):
        r["label"] = l
        
    print("Labels permuted.")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = AbstractDataset(train_records, tokenizer)
    val_dataset = AbstractDataset(val_records, tokenizer)
    test_dataset = AbstractDataset(test_records, tokenizer)
    holdout_dataset = AbstractDataset(holdout_records, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    holdout_loader = DataLoader(holdout_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = BERTClassifier(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda", enabled=True)
    
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*0.1), total_steps)
    
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
            
            if step % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
        val_auc = evaluate_auc(model, val_loader, device)
        test_auc = evaluate_auc(model, test_loader, device)
        holdout_auc = evaluate_auc(model, holdout_loader, device)
        print(f"Permutation Test - Epoch {epoch}")
        print(f"  Val AUC:     {val_auc*100:.2f}%")
        print(f"  Test AUC:    {test_auc*100:.2f}%")
        print(f"  Holdout AUC: {holdout_auc*100:.2f}%")

if __name__ == "__main__":
    main()
