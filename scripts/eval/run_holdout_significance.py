import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import chi2
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset

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
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, 1))
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

def get_bert_probs(model_path, model_name, records, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BERTClassifier(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["model_state_dict"])
    model.to(device)
    model.eval()
    
    loader = DataLoader(AbstractDataset(records, tokenizer), batch_size=16, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids, mask)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)

def main():
    with open("data/final_v2_period_stripped/cancer_pm_train.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open("data/final_v2_period_stripped/cancer_pm_holdout.json", "r", encoding="utf-8") as f:
        holdout = json.load(f)

    train_texts = [r["title"] + " " + r["abstract"] for r in train]
    train_labels = np.array([r["label"] for r in train])
    holdout_texts = [r["title"] + " " + r["abstract"] for r in holdout]
    holdout_labels = np.array([r["label"] for r in holdout])

    print("Training TF-IDF Combiné (min_df=5, C=10.0)...")
    vec_comb = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    X_train_comb = vec_comb.fit_transform(train_texts)
    X_holdout_comb = vec_comb.transform(holdout_texts)
    clf_comb = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced", random_state=42)
    clf_comb.fit(X_train_comb, train_labels)
    probs_comb = clf_comb.predict_proba(X_holdout_comb)[:, 1]
    preds_comb = (probs_comb >= 0.36).astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Getting PubMedBERT Epoch 3 probs on Holdout...")
    probs_pm = get_bert_probs("models/pubmedbert_epoch_3.pt", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", holdout, device)
    preds_pm = (probs_pm >= 0.36).astype(int)

    # McNemar Test
    b = int(((preds_pm == holdout_labels) & (preds_comb != holdout_labels)).sum())
    c = int(((preds_pm != holdout_labels) & (preds_comb == holdout_labels)).sum())
    if b + c > 0:
        chi2_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p_val = chi2.sf(chi2_stat, 1)
    else:
        chi2_stat, p_val = 0.0, 1.0
    print(f"\nMcNemar: PubMedBERT vs TF-IDF Combiné (Holdout)")
    print(f"  p-value: {p_val:.6f} (Significant: {p_val < 0.05})")

    np.random.seed(42)
    n_samples = len(holdout_labels)
    diff_f1 = []
    
    for i in range(2000):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_y = holdout_labels[idx]
        if len(np.unique(boot_y)) < 2: continue
        f1_c = f1_score(boot_y, preds_comb[idx], zero_division=0)
        f1_p = f1_score(boot_y, preds_pm[idx], zero_division=0)
        diff_f1.append(f1_p - f1_c)
        
    mean_pm = np.mean(diff_f1)
    lo_pm = np.percentile(diff_f1, 2.5)
    hi_pm = np.percentile(diff_f1, 97.5)
    
    print(f"\nBootstrap dF1 (PubMedBERT - TF-IDF Combiné):")
    print(f"  Mean: {mean_pm*100:+.2f}%, 95% CI: [{lo_pm*100:+.2f}%, {hi_pm*100:+.2f}%]")
    print(f"  Distinguishable from 0: {not (lo_pm <= 0 <= hi_pm)}")

if __name__ == "__main__": main()
