import json
import csv
import sys
import re
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Redirect output to file
output_path = r"C:\projects\cancer-paper-mill\leakage_check_results.txt"
sys.stdout = open(output_path, "w", encoding="utf-8")
sys.stderr = sys.stdout

# Define model class
class BERTClassifier(nn.Module):
    def __init__(self, model_name):
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

class AbstractDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=128):
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

def get_3grams(text):
    text_clean = re.sub(r'\s+', ' ', text.lower().strip())
    if len(text_clean) < 3:
        return set()
    return set(text_clean[i:i+3] for i in range(len(text_clean) - 2))

def get_publisher(rec, journal_to_publisher, nlm2raw):
    j = rec['journal']
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, 'Direct PubMed')
    if 'computational and mathematical methods in medicine' in raw_j.lower() or 'comput math methods med' in j.lower():
        pub = 'Hindawi'
    return pub

def main():
    print("="*80)
    print("DIAGNOSTIC TEST: HINDAWI HOLDOUT LEAKAGE AND SCORE SPACE SEPARATION")
    print("="*80)
    
    # -------------------------------------------------------------
    # 1. Check Leakage: Holdout positives vs. Train positives (Hindawi)
    # -------------------------------------------------------------
    with open("data/final/cancer_pm_train.json", "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open("data/final/cancer_pm_holdout.json", "r", encoding="utf-8") as f:
        holdout_records = json.load(f)
    with open("data/final/cancer_pm_val.json", "r", encoding="utf-8") as f:
        val_records = json.load(f)

    with open('data/final/journal_to_nlm.json', 'r', encoding='utf-8') as f:
        j2nlm = json.load(f)
    nlm2raw = {v: k for k, v in j2nlm.items()}

    # Initialize publisher mapper
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

    # Filter train positives that are Hindawi
    train_hindawi_pos = [r for r in train_records if r["label"] == 1 and get_publisher(r, journal_to_publisher, nlm2raw) == 'Hindawi']
    holdout_hindawi_pos = [r for r in holdout_records if r["label"] == 1]
    
    print(f"\nNumber of Hindawi positives in Train: {len(train_hindawi_pos)}")
    print(f"Number of Hindawi positives in Holdout: {len(holdout_hindawi_pos)}")

    # Jaccard title similarity check (similar to split_dataset.py)
    max_jaccards = []
    near_dups_jaccard = []

    train_titles_grams = [(r["doi"], r["title"], get_3grams(r["title"])) for r in train_hindawi_pos]

    for h_rec in holdout_hindawi_pos:
        h_title = h_rec["title"]
        h_doi = h_rec["doi"]
        h_grams = get_3grams(h_title)
        
        best_jaccard = 0.0
        best_match_title = ""
        best_match_doi = ""
        
        if h_grams:
            for t_doi, t_title, t_grams in train_titles_grams:
                if not t_grams:
                    continue
                inter = len(h_grams & t_grams)
                union_len = len(h_grams | t_grams)
                jac = inter / union_len if union_len > 0 else 0
                if jac > best_jaccard:
                    best_jaccard = jac
                    best_match_title = t_title
                    best_match_doi = t_doi
                    
        max_jaccards.append(best_jaccard)
        if best_jaccard >= 0.7:
            near_dups_jaccard.append({
                "holdout_doi": h_doi,
                "holdout_title": h_title,
                "train_doi": best_match_doi,
                "train_title": best_match_title,
                "jaccard": best_jaccard
            })

    # Cosine similarity check on Titles + Abstracts
    # Fit vectorizer on all train + holdout texts
    train_texts = [r["title"] + " " + r["abstract"] for r in train_hindawi_pos]
    holdout_texts = [r["title"] + " " + r["abstract"] for r in holdout_hindawi_pos]
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    train_tfidf = vectorizer.fit_transform(train_texts)
    holdout_tfidf = vectorizer.transform(holdout_texts)
    
    cos_sims = cosine_similarity(holdout_tfidf, train_tfidf)
    max_cosines = np.max(cos_sims, axis=1)
    
    near_dups_cosine = []
    for idx, best_cos in enumerate(max_cosines):
        if best_cos >= 0.7:
            train_idx = np.argmax(cos_sims[idx])
            near_dups_cosine.append({
                "holdout_doi": holdout_hindawi_pos[idx]["doi"],
                "holdout_title": holdout_hindawi_pos[idx]["title"],
                "train_doi": train_hindawi_pos[train_idx]["doi"],
                "train_title": train_hindawi_pos[train_idx]["title"],
                "cosine": best_cos
            })

    print(f"\nDistribution of Nearest-Neighbor Title 3-gram Jaccard similarities:")
    print(f"  Min:  {np.min(max_jaccards):.4f}")
    print(f"  25%:  {np.percentile(max_jaccards, 25):.4f}")
    print(f"  50%:  {np.median(max_jaccards):.4f}")
    print(f"  75%:  {np.percentile(max_jaccards, 75):.4f}")
    print(f"  Max:  {np.max(max_jaccards):.4f}")
    print(f"  Mean: {np.mean(max_jaccards):.4f}")
    print(f"  Count of holdout positives with Jaccard >= 0.7 overlap in training: {len(near_dups_jaccard)}")

    print(f"\nDistribution of Nearest-Neighbor Title+Abstract TF-IDF Cosine similarities:")
    print(f"  Min:  {np.min(max_cosines):.4f}")
    print(f"  25%:  {np.percentile(max_cosines, 25):.4f}")
    print(f"  50%:  {np.median(max_cosines):.4f}")
    print(f"  75%:  {np.percentile(max_cosines, 75):.4f}")
    print(f"  Max:  {np.max(max_cosines):.4f}")
    print(f"  Mean: {np.mean(max_cosines):.4f}")
    print(f"  Count of holdout positives with Cosine >= 0.7 overlap in training: {len(near_dups_cosine)}")

    if len(near_dups_jaccard) > 0:
        print("\n--- SAMPLE JACCARD NEAR-DUPLICATES LINKING TRAIN AND HOLDOUT ---")
        for i, item in enumerate(near_dups_jaccard[:10]):
            print(f"Match {i+1}:")
            print(f"  Holdout DOI:   {item['holdout_doi']}")
            print(f"  Holdout Title: {repr(item['holdout_title'])}")
            print(f"  Train DOI:     {item['train_doi']}")
            print(f"  Train Title:   {repr(item['train_title'])}")
            print(f"  Jaccard:       {item['jaccard']:.4f}")

    if len(near_dups_cosine) > 0:
        print("\n--- SAMPLE COSINE NEAR-DUPLICATES LINKING TRAIN AND HOLDOUT ---")
        for i, item in enumerate(near_dups_cosine[:10]):
            print(f"Match {i+1}:")
            print(f"  Holdout DOI:   {item['holdout_doi']}")
            print(f"  Holdout Title: {repr(item['holdout_title'])}")
            print(f"  Train DOI:     {item['train_doi']}")
            print(f"  Train Title:   {repr(item['train_title'])}")
            print(f"  Cosine:        {item['cosine']:.4f}")

    # -------------------------------------------------------------
    # 2. Get Raw Prediction Scores for Validation Sous-groupe B vs Negatives
    # -------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice for BERT Inference: {device}")

    # Read RWDB reasons to identify validation subgroup B
    doi_to_reasons = {}
    pmid_to_reasons = {}
    title_to_reasons = {}

    with open('data/raw/rwdb/retraction_watch.csv', 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doi = row.get('OriginalPaperDOI', '').strip().lower()
            pmid = row.get('OriginalPaperPubMedID', '').strip()
            title = row.get('Title', '').strip().lower()
            reasons = row.get('Reason', '').strip()
            if doi and doi != "0" and doi != "nan":
                doi_to_reasons[doi] = reasons
            if pmid and pmid != "0" and pmid != "nan":
                pmid_to_reasons[pmid] = reasons
            if title:
                title_to_reasons[title] = reasons

    # Filter val records for Hindawi
    val_hindawi = [r for r in val_records if get_publisher(r, journal_to_publisher, nlm2raw) == 'Hindawi']
    positives_val = [r for r in val_hindawi if r["label"] == 1]
    negatives_val = [r for r in val_hindawi if r["label"] == 0]

    sg_b = []
    for r in positives_val:
        doi = (r.get("doi", "") or "").strip().lower()
        pmid = str(r.get("pmid", "") or "").strip()
        title = (r.get("title", "") or "").strip().lower()
        
        reasons = ""
        if doi in doi_to_reasons:
            reasons = doi_to_reasons[doi]
        elif pmid in pmid_to_reasons:
            reasons = pmid_to_reasons[pmid]
        elif title in title_to_reasons:
            reasons = title_to_reasons[title]
        else:
            for t, reas in title_to_reasons.items():
                if t in title or title in t:
                    reasons = reas
                    break
        
        is_computer_generated = "computer-aided" in reasons.lower() or "computer-generated" in reasons.lower()
        if not is_computer_generated:
            sg_b.append(r)

    print(f"\nValidation Hindawi Negatives Count: {len(negatives_val)}")
    print(f"Validation Hindawi Sous-groupe B Positives Count: {len(sg_b)}")

    # Load Model
    pm_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    tokenizer_pm = AutoTokenizer.from_pretrained(pm_model_name)
    checkpoint_pm = torch.load("models/pubmedbert_finetuned_best.pt", map_location=device, weights_only=False)
    model_pm = BERTClassifier(pm_model_name)
    model_pm.load_state_dict(checkpoint_pm["model_state_dict"])
    model_pm.to(device)
    model_pm.eval()

    # Predict on SG B
    dataset_sg_b = AbstractDataset(sg_b, tokenizer_pm)
    loader_sg_b = DataLoader(dataset_sg_b, batch_size=8, shuffle=False)
    
    probs_sg_b = []
    with torch.no_grad():
        for batch in loader_sg_b:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model_pm(input_ids, mask)
            prbs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            probs_sg_b.extend(prbs)

    # Predict on Negatives
    dataset_neg = AbstractDataset(negatives_val, tokenizer_pm)
    loader_neg = DataLoader(dataset_neg, batch_size=8, shuffle=False)
    
    probs_neg = []
    with torch.no_grad():
        for batch in loader_neg:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model_pm(input_ids, mask)
            prbs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            probs_neg.extend(prbs)

    probs_sg_b = np.array(probs_sg_b)
    probs_neg = np.array(probs_neg)

    print("\n--- RAW PREDICTION SCORE DISTRIBUTION (PubMedBERT on SG B vs Negatives) ---")
    print(f"Sous-groupe B Positives (N={len(sg_b)}):")
    print(f"  Min score: {np.min(probs_sg_b):.8f}")
    print(f"  Max score: {np.max(probs_sg_b):.8f}")
    print(f"  Median   : {np.median(probs_sg_b):.8f}")
    print(f"  Mean     : {np.mean(probs_sg_b):.8f}")
    print(f"  Quantile 10%: {np.percentile(probs_sg_b, 10):.8f}")
    print(f"  Quantile 25%: {np.percentile(probs_sg_b, 25):.8f}")

    print(f"\nHindawi Negatives (N={len(negatives_val)}):")
    print(f"  Min score: {np.min(probs_neg):.8f}")
    print(f"  Max score: {np.max(probs_neg):.8f}")
    print(f"  Median   : {np.median(probs_neg):.8f}")
    print(f"  Mean     : {np.mean(probs_neg):.8f}")
    print(f"  Quantile 90%: {np.percentile(probs_neg, 90):.8f}")
    print(f"  Quantile 75%: {np.percentile(probs_neg, 75):.8f}")

    gap = np.min(probs_sg_b) - np.max(probs_neg)
    print(f"\nDecision Margin (Separation Gap: Min_Pos - Max_Neg): {gap:.8f}")
    if gap > 0:
        print("SUCCESS: Clear and perfect separation gap between SG B positives and negatives!")
    else:
        print("OVERLAP DETECTED.")
        
    print("\nAll done.")

if __name__ == "__main__":
    main()
