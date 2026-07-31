import json
import csv
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import sys
import os

# --- 1. Replication of Extraction Pipeline ---
def fetch_pubmed_abstracts_batch(pmids):
    if not pmids:
        return {}
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(pmids)}&retmode=xml"
    headers = {'User-Agent': 'CancerPaperMill/1.0'}
    req = urllib.request.Request(url, headers=headers)
    results = {}
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            xml_text = r.read().decode('utf-8', errors='replace')
            root = ET.fromstring(xml_text)
            for article in root.findall('.//PubmedArticle'):
                pmid_el = article.find('.//PMID')
                pmid = pmid_el.text.strip() if pmid_el is not None else ""
                if not pmid:
                    continue
                    
                ab_parts = []
                for ab_el in article.findall('.//AbstractText'):
                    if ab_el.text:
                        label = ab_el.attrib.get('Label')
                        if label:
                            ab_parts.append(f"{label}: {ab_el.text.strip()}")
                        else:
                            ab_parts.append(ab_el.text.strip())
                            
                abstract = " ".join(ab_parts) if ab_parts else ""
                results[pmid] = abstract
    except Exception as e:
        print(f"Batch fetch error: {e}")
    return results

# --- 2. Evaluation Setup ---
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the 40 samples from Task 0
    with open(r"C:\Users\moham\.gemini\antigravity\brain\c22a0783-bc26-4f1e-820b-3c6515e0eaa8\scratch\task0_sample.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    negatives = [s for s in samples if s['label'] == 0]
    pmids = [n['pmid'] for n in negatives]
    
    print(f"Fetching {len(pmids)} negatives from PubMed API...")
    fetched_abstracts = fetch_pubmed_abstracts_batch(pmids)
    
    # Update negatives with new abstract
    for n in negatives:
        pmid = n['pmid']
        if pmid in fetched_abstracts:
            n['abstract'] = fetched_abstracts[pmid]
        else:
            print(f"Warning: could not refetch pmid {pmid}")
            
    # Load model and evaluate
    model_path = "models/pubmedbert_finetuned_best.pt"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = AbstractDataset(negatives, tokenizer, MAX_LENGTH)
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
            pmids_batch = batch["pmid"]
            
            with autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(input_ids, mask)
            
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            
            for pmid, label, prob in zip(pmids_batch, labels, probs):
                results.append({
                    "pmid": pmid,
                    "label": int(label),
                    "pred_prob": float(prob)
                })
                
    out_csv = r"C:\Users\moham\.gemini\antigravity\brain\c22a0783-bc26-4f1e-820b-3c6515e0eaa8\scratch\task3_predictions.csv"
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["pmid", "label", "pred_prob"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Saved refetched negative predictions to {out_csv}")
    
    # --- 3. Statistical Analysis ---
    df0 = pd.read_csv(r"C:\Users\moham\.gemini\antigravity\brain\c22a0783-bc26-4f1e-820b-3c6515e0eaa8\scratch\task0_baseline.csv")
    df3 = pd.read_csv(out_csv)
    
    # Force pmids to be strings for safe comparison
    df0['pmid'] = df0['pmid'].astype(str)
    df3['pmid'] = df3['pmid'].astype(str)
    
    # Filter df0 to just these negatives
    df0_neg = df0[df0['pmid'].isin(df3['pmid'])].sort_values('pmid')
    df3 = df3.sort_values('pmid')
    
    # Compute differences (Task 3 - Task 0)
    # If the format leak pushes them to look like positives, prob will go up.
    diffs = df3['pred_prob'].values - df0_neg['pred_prob'].values
    mean_diff = np.mean(diffs)
    
    # Bootstrap CI
    n_bootstraps = 1000
    rng = np.random.default_rng(42)
    bootstrap_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_bootstraps)]
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    print("\n--- Statistical Results ---")
    print(f"Mean predicted prob (Baseline): {np.mean(df0_neg['pred_prob'].values):.4f}")
    print(f"Mean predicted prob (Task 3):   {np.mean(df3['pred_prob'].values):.4f}")
    print(f"Mean Difference: {mean_diff:.4f}")
    print(f"95% CI of Difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    if ci_lower > 0:
        print("Verdict: Pipeline leak STATISTICALLY CONFIRMED (95% CI excludes 0 and is positive)")
    else:
        print("Verdict: No statistically significant shift towards positive class.")

if __name__ == "__main__":
    main()
