import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
import numpy as np

class BERTClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, 1))
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

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

def eval_model(model_path, model_name, records, device, thresh=0.36):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BERTClassifier(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["model_state_dict"])
    model.to(device)
    model.eval()
    
    loader = DataLoader(AbstractDataset(records, tokenizer), batch_size=16, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids, mask)
            probs.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
    
    y = np.array([r["label"] for r in records])
    p = np.array(probs)
    preds = (p >= thresh).astype(int)
    
    auc = roc_auc_score(y, p)
    f1 = f1_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    pr_auc = average_precision_score(y, p)
    
    return {"prec": prec, "rec": rec, "f1": f1, "auc": auc, "pr_auc": pr_auc}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("data/final_v2_period_stripped/cancer_pm_test.json", "r", encoding="utf-8") as f: test_records = json.load(f)
    with open("data/final_v2_period_stripped/cancer_pm_holdout.json", "r", encoding="utf-8") as f: holdout_records = json.load(f)

    # For stratified test
    with open("data/final_v2_period_stripped/journal_to_nlm.json", "r", encoding="utf-8") as f: nlm2raw = {v: k for k, v in json.load(f).items()}
    import csv
    j2p = {}
    with open("data/raw/rwdb/retraction_watch.csv", "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            j, p = row.get("Journal", "").strip(), row.get("Publisher", "").strip()
            if j and p: j2p[j] = p
            
    def get_publisher(rec):
        j = rec['journal']
        raw_j = nlm2raw.get(j, j)
        pub = j2p.get(raw_j, 'Direct PubMed')
        if 'computational and mathematical methods in medicine' in raw_j.lower() or 'comput math methods med' in j.lower(): pub = 'Hindawi'
        elif 'biomed research international' in raw_j.lower() or 'biomed res int' in j.lower(): pub = 'Hindawi'
        return pub

    test_h = [r for r in test_records if get_publisher(r) == 'Hindawi']
    test_s = [r for r in test_records if get_publisher(r) == 'Spandidos']
    test_o = [r for r in test_records if get_publisher(r) not in ['Hindawi', 'Spandidos']]

    print("--- PubMedBERT (Epoch 3) ---")
    pm_path = "models/pubmedbert_epoch_3.pt"
    pm_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    
    res = eval_model(pm_path, pm_name, test_records, device)
    print(f"Test Global: Prec={res['prec']:.4f}, Rec={res['rec']:.4f}, F1={res['f1']:.4f}, AUC={res['auc']:.4f}, PR-AUC={res['pr_auc']:.4f}")
    
    res = eval_model(pm_path, pm_name, test_h, device)
    print(f"Test Hindawi: Prec={res['prec']:.4f}, Rec={res['rec']:.4f}, F1={res['f1']:.4f}, AUC={res['auc']:.4f}")
    
    res = eval_model(pm_path, pm_name, test_s, device)
    print(f"Test Spandidos: Prec={res['prec']:.4f}, Rec={res['rec']:.4f}, F1={res['f1']:.4f}, AUC={res['auc']:.4f}")
    
    res = eval_model(pm_path, pm_name, test_o, device)
    print(f"Test Others: Prec={res['prec']:.4f}, Rec={res['rec']:.4f}, F1={res['f1']:.4f}, AUC={res['auc']:.4f}")
    
    res = eval_model(pm_path, pm_name, holdout_records, device)
    print(f"Holdout Hindawi: Prec={res['prec']:.4f}, Rec={res['rec']:.4f}, F1={res['f1']:.4f}, AUC={res['auc']:.4f}, PR-AUC={res['pr_auc']:.4f}")
    
    print("\n--- SciBERT (Epoch 3) ---")
    sci_path = "models/scibert_epoch_3.pt"
    sci_name = "allenai/scibert_scivocab_uncased"
    
    res = eval_model(sci_path, sci_name, test_records, device)
    print(f"Test Global: Prec={res['prec']:.4f}, Rec={res['rec']:.4f}, F1={res['f1']:.4f}, AUC={res['auc']:.4f}, PR-AUC={res['pr_auc']:.4f}")
    
    res = eval_model(sci_path, sci_name, test_h, device)
    print(f"Test Hindawi: Prec={res['prec']:.4f}, Rec={res['rec']:.4f}, F1={res['f1']:.4f}, AUC={res['auc']:.4f}")
    
    res = eval_model(sci_path, sci_name, test_s, device)
    print(f"Test Spandidos: Prec={res['prec']:.4f}, Rec={res['rec']:.4f}, F1={res['f1']:.4f}, AUC={res['auc']:.4f}")
    
    res = eval_model(sci_path, sci_name, test_o, device)
    print(f"Test Others: Prec={res['prec']:.4f}, Rec={res['rec']:.4f}, F1={res['f1']:.4f}, AUC={res['auc']:.4f}")
    
    res = eval_model(sci_path, sci_name, holdout_records, device)
    print(f"Holdout Hindawi: Prec={res['prec']:.4f}, Rec={res['rec']:.4f}, F1={res['f1']:.4f}, AUC={res['auc']:.4f}, PR-AUC={res['pr_auc']:.4f}")

if __name__ == "__main__": main()
