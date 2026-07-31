import json
import csv
import sys
import os
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

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
        pmid = rec.get("pmid", str(idx))
        
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
            "label": torch.tensor(label, dtype=torch.float32),
            "pmid": pmid
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

def get_publisher(rec, journal_to_publisher, nlm2raw):
    j = rec.get('journal', '')
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() or "comput math methods med" in j.lower():
        pub = "Hindawi"
    elif "biomed research international" in raw_j.lower() or "biomed res int" in j.lower():
        pub = "Hindawi"
    return pub

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load publisher info
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
    
    # Load holdout data
    with open("data/final/cancer_pm_holdout.json", "r", encoding="utf-8") as f:
        holdout = json.load(f)
        
    hindawi_recs = [r for r in holdout if get_publisher(r, journal_to_publisher, nlm2raw) == 'Hindawi']
    hindawi_pos = [r for r in hindawi_recs if r['label'] == 1][:20]
    hindawi_neg = [r for r in hindawi_recs if r['label'] == 0][:20]
    
    test_recs = hindawi_pos + hindawi_neg
    print(f"Selected {len(hindawi_pos)} pos and {len(hindawi_neg)} neg Hindawi examples.")
    
    # Load model
    model_path = "models/pubmedbert_finetuned_best.pt"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = AbstractDataset(test_recs, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = BERTClassifier(MODEL_NAME).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()
            pmids = batch["pmid"]
            
            with autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(input_ids, mask)
            
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            
            for pmid, label, prob in zip(pmids, labels, probs):
                results.append({
                    "pmid": pmid,
                    "label": int(label),
                    "pred_prob": float(prob)
                })
                
    # Save to CSV
    os.makedirs(r"C:\Users\moham\.gemini\antigravity\brain\c22a0783-bc26-4f1e-820b-3c6515e0eaa8\scratch", exist_ok=True)
    out_csv = r"C:\Users\moham\.gemini\antigravity\brain\c22a0783-bc26-4f1e-820b-3c6515e0eaa8\scratch\task0_baseline.csv"
    
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["pmid", "label", "pred_prob"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Saved baseline predictions to {out_csv}")
    
    # Select the same 40 for task 1
    out_json = r"C:\Users\moham\.gemini\antigravity\brain\c22a0783-bc26-4f1e-820b-3c6515e0eaa8\scratch\task0_sample.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(test_recs, f, indent=2)

if __name__ == "__main__":
    main()
