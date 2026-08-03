"""
Follow-up Analysis for Subgroup Effects & Crossref Interpretation
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
import csv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout.reconfigure(encoding='utf-8')

# Publisher mapping
journal_to_publisher = {}
rwdb_path = os.path.join(project_root, 'data', 'raw', 'rwdb', 'retraction_watch.csv')
with open(rwdb_path, 'r', encoding='utf-8', errors='replace') as f:
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

with open(os.path.join(project_root, 'data', 'final', 'journal_to_nlm.json'), 'r', encoding='utf-8') as f:
    j2nlm = json.load(f)
nlm2raw = {v: k for k, v in j2nlm.items()}

def get_publisher(rec):
    j = rec.get('journal', '')
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if 'computational and mathematical methods in medicine' in raw_j.lower() or 'comput math methods med' in j.lower():
        pub = 'Hindawi'
    elif 'biomed research international' in raw_j.lower() or 'biomed res int' in j.lower():
        pub = 'Hindawi'
    return pub

# Model components
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LENGTH = 512
BATCH_SIZE = 8
FROZEN_THRESHOLD = 0.10

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
        
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.utils.stats import bootstrap_ci

def main():
    print("=" * 80)
    print("SUBGROUP FOLLOW-UP ANALYSIS")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BERTClassifier(MODEL_NAME).to(device)
    model_path = os.path.join(project_root, 'models', 'pubmedbert_finetuned_best.pt')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    def score_records(records):
        if not records:
            return np.array([])
        dataset = AbstractDataset(records, tokenizer, MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                probs = torch.sigmoid(model(input_ids, mask).squeeze(-1)).cpu().numpy()
                all_probs.extend(probs)
                if (i+1) % 50 == 0:
                    print(f"  Scoring batch {i+1}/{len(loader)}")
        return np.array(all_probs)

    # 1. Load Hindawi holdout data
    with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_holdout.json'), 'r', encoding='utf-8') as f:
        holdout = json.load(f)
    
    hindawi_holdout = [r for r in holdout if get_publisher(r) == 'Hindawi']
    total_pos = sum(1 for r in hindawi_holdout if r['label'] == 1)
    total_neg = sum(1 for r in hindawi_holdout if r['label'] == 0)
    
    print(f"\nTotal Hindawi Holdout: Pos={total_pos}, Neg={total_neg}")
    
    # Stratify by journal
    FLAGGED_JOURNALS = ['Evid Based Complement Alternat Med', 'Contrast Media Mol Imaging']
    
    flagged_recs = [r for r in hindawi_holdout if r.get('journal', '') in FLAGGED_JOURNALS]
    unflagged_recs = [r for r in hindawi_holdout if r.get('journal', '') not in FLAGGED_JOURNALS]
    
    flagged_pos = sum(1 for r in flagged_recs if r['label'] == 1)
    flagged_neg = sum(1 for r in flagged_recs if r['label'] == 0)
    unflagged_pos = sum(1 for r in unflagged_recs if r['label'] == 1)
    unflagged_neg = sum(1 for r in unflagged_recs if r['label'] == 0)
    
    print("\n--- Composition ---")
    print(f"Flagged Journals (EBCAM, CMMI):")
    print(f"  Positives: {flagged_pos} ({flagged_pos/total_pos*100:.1f}% of Hindawi pos)")
    print(f"  Negatives: {flagged_neg} ({flagged_neg/total_neg*100:.1f}% of Hindawi neg)")
    
    print(f"\nUnflagged Journals (Rest of Hindawi):")
    print(f"  Positives: {unflagged_pos} ({unflagged_pos/total_pos*100:.1f}% of Hindawi pos)")
    print(f"  Negatives: {unflagged_neg} ({unflagged_neg/total_neg*100:.1f}% of Hindawi neg)")
    
    # 2. Score and evaluate
    print("\nScoring Flagged Journals...")
    flagged_scores = score_records(flagged_recs)
    flagged_y = np.array([r['label'] for r in flagged_recs])
    
    if len(np.unique(flagged_y)) > 1:
        flagged_auc = roc_auc_score(flagged_y, flagged_scores)
    else:
        flagged_auc = np.nan
    
    flagged_neg_idx = (flagged_y == 0)
    if np.sum(flagged_neg_idx) > 0:
        flagged_fpr = np.mean(flagged_scores[flagged_neg_idx] >= FROZEN_THRESHOLD)
    else:
        flagged_fpr = np.nan
        
    print("\nScoring Unflagged Journals...")
    unflagged_scores = score_records(unflagged_recs)
    unflagged_y = np.array([r['label'] for r in unflagged_recs])
    
    if len(np.unique(unflagged_y)) > 1:
        unflagged_auc = roc_auc_score(unflagged_y, unflagged_scores)
    else:
        unflagged_auc = np.nan
        
    unflagged_neg_idx = (unflagged_y == 0)
    if np.sum(unflagged_neg_idx) > 0:
        unflagged_fpr = np.mean(unflagged_scores[unflagged_neg_idx] >= FROZEN_THRESHOLD)
    else:
        unflagged_fpr = np.nan

    print("\n--- Original Holdout Performance by Subgroup ---")
    print(f"Flagged Journals AUC: {flagged_auc:.4f}")
    print(f"Flagged Journals FPR@0.10: {flagged_fpr*100:.2f}%")
    print(f"Unflagged Journals AUC: {unflagged_auc:.4f}")
    print(f"Unflagged Journals FPR@0.10: {unflagged_fpr*100:.2f}%")
    
    # 3. Bootstrap CI on Provenance-Matched Negatives (EBCAM & CMMI)
    print("\n--- Bootstrap CI on Provenance-Matched Negatives FPR ---")
    with open(os.path.join(project_root, 'data', 'final', 'provenance_matched_negatives.json'), 'r', encoding='utf-8') as f:
        prov_neg = json.load(f)
    
    for journal in FLAGGED_JOURNALS:
        j_recs = [r for r in prov_neg if r.get('journal', '') == journal or r.get('target_journal', '') == journal]
        if not j_recs:
            print(f"No provenance-matched records found for {journal}")
            continue
            
        print(f"Scoring {len(j_recs)} provenance-matched negatives for {journal}...")
        j_scores = score_records(j_recs)
        
        obs_fpr = np.mean(j_scores >= FROZEN_THRESHOLD)
        
        def fpr_stat(scores):
            return np.mean(scores >= FROZEN_THRESHOLD)
            
        lower, upper = bootstrap_ci(j_scores, fpr_stat, n_resamples=2000)
        
        print(f"  {journal}: FPR = {obs_fpr*100:.1f}% [95% CI: {lower*100:.1f}% - {upper*100:.1f}%]")
    
if __name__ == "__main__":
    main()
