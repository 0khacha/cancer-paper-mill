import json
import csv
import sys
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score

sys.stdout.reconfigure(encoding='utf-8')

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

def get_publisher(rec, journal_to_publisher, nlm2raw):
    j = rec['journal']
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, 'Direct PubMed')
    if 'computational and mathematical methods in medicine' in raw_j.lower() or 'comput math methods med' in j.lower():
        pub = 'Hindawi'
    return pub

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load validation data
    with open("data/final_v2_period_stripped/cancer_pm_val.json", "r", encoding="utf-8") as f:
        val_records = json.load(f)

    with open('data/final_v2_period_stripped/journal_to_nlm.json', 'r', encoding='utf-8') as f:
        j2nlm = json.load(f)
    nlm2raw = {v: k for k, v in j2nlm.items()}

    # Load RWDB reason mapping
    doi_to_reasons = {}
    pmid_to_reasons = {}
    title_to_reasons = {}
    journal_to_publisher = {}

    with open('data/raw/rwdb/retraction_watch.csv', 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doi = row.get('OriginalPaperDOI', '').strip().lower()
            pmid = row.get('OriginalPaperPubMedID', '').strip()
            title = row.get('Title', '').strip().lower()
            reasons = row.get('Reason', '').strip()
            
            j = row.get('Journal', '').strip()
            p = row.get('Publisher', '').strip()
            if j and p:
                if "computational and mathematical methods in medicine" in j.lower():
                    p = "Hindawi"
                elif "biomed research international" in j.lower():
                    p = "Hindawi"
                journal_to_publisher[j] = p

            if doi and doi != "0" and doi != "nan":
                doi_to_reasons[doi] = reasons
            if pmid and pmid != "0" and pmid != "nan":
                pmid_to_reasons[pmid] = reasons
            if title:
                title_to_reasons[title] = reasons

    # Filter validation papers for Hindawi
    hindawi_val = [r for r in val_records if get_publisher(r, journal_to_publisher, nlm2raw) == 'Hindawi']
    print(f"Total Hindawi Validation Records: {len(hindawi_val)}")
    
    positives = [r for r in hindawi_val if r["label"] == 1]
    negatives = [r for r in hindawi_val if r["label"] == 0]
    print(f"Hindawi Positives: {len(positives)}, Negatives: {len(negatives)}")

    # Classify positives into Sous-groupe A and B
    sg_a = []
    sg_b = []
    not_matched = 0

    for r in positives:
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
            # Try soft title matching
            for t, reas in title_to_reasons.items():
                if t in title or title in t:
                    reasons = reas
                    break
        
        if not reasons:
            not_matched += 1
            sg_b.append(r)
            continue

        is_computer_generated = "computer-aided" in reasons.lower() or "computer-generated" in reasons.lower()
        if is_computer_generated:
            sg_a.append(r)
        else:
            sg_b.append(r)

    print(f"Matched positives -> Sous-groupe A (IA): {len(sg_a)}, Sous-groupe B (Autre): {len(sg_b)} (including {not_matched} unmatched)")

    # Evaluation helper
    def eval_subgroup(model, tokenizer, pos_records, neg_records):
        records = pos_records + neg_records
        dataset = AbstractDataset(records, tokenizer)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        probs = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbls = batch["label"]
                logits = model(input_ids, mask)
                prbs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                probs.extend(prbs)
                labels.extend(lbls.numpy())
        
        probs = np.array(probs)
        labels = np.array(labels)
        
        auc = roc_auc_score(labels, probs)
        
        # Calculate F1 using frozen 0.36 threshold (which was used everywhere else)
        preds_36 = (probs >= 0.36).astype(int)
        f1_36 = f1_score(labels, preds_36, zero_division=0)
        
        return auc, f1_36

    # 1. Evaluate PubMedBERT fine-tuned model
    print("\nEvaluating PubMedBERT...")
    pm_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    tokenizer_pm = AutoTokenizer.from_pretrained(pm_model_name)
    checkpoint_pm = torch.load("models/pubmedbert_finetuned_best.pt", map_location=device, weights_only=False)
    model_pm = BERTClassifier(pm_model_name)
    model_pm.load_state_dict(checkpoint_pm["model_state_dict"])
    model_pm.to(device)
    model_pm.eval()

    auc_pm_a, f1_36_pm_a = eval_subgroup(model_pm, tokenizer_pm, sg_a, negatives)
    auc_pm_b, f1_36_pm_b = eval_subgroup(model_pm, tokenizer_pm, sg_b, negatives)

    print(f"PubMedBERT on SG A (IA vs Neg): AUC = {auc_pm_a:.4%}, F1@0.36 = {f1_36_pm_a:.4%}")
    print(f"PubMedBERT on SG B (Autre vs Neg): AUC = {auc_pm_b:.4%}, F1@0.36 = {f1_36_pm_b:.4%}")

    # 2. Evaluate SciBERT fine-tuned model
    print("\nEvaluating SciBERT...")
    sci_model_name = "allenai/scibert_scivocab_uncased"
    tokenizer_sci = AutoTokenizer.from_pretrained(sci_model_name)
    checkpoint_sci = torch.load("models/scibert_finetuned_best.pt", map_location=device, weights_only=False)
    model_sci = BERTClassifier(sci_model_name)
    model_sci.load_state_dict(checkpoint_sci["model_state_dict"])
    model_sci.to(device)
    model_sci.eval()

    auc_sci_a, f1_36_sci_a = eval_subgroup(model_sci, tokenizer_sci, sg_a, negatives)
    auc_sci_b, f1_36_sci_b = eval_subgroup(model_sci, tokenizer_sci, sg_b, negatives)

    print(f"SciBERT on SG A (IA vs Neg): AUC = {auc_sci_a:.4%}, F1@0.36 = {f1_36_sci_a:.4%}")
    print(f"SciBERT on SG B (Autre vs Neg): AUC = {auc_sci_b:.4%}, F1@0.36 = {f1_36_sci_b:.4%}")

if __name__ == "__main__":
    main()
