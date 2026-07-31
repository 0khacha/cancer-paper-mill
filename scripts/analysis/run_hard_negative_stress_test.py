import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re

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

def find_split_for_dois(dois):
    splits = {
        "train": "data/final/cancer_pm_train.json",
        "val": "data/final/cancer_pm_val.json",
        "test": "data/final/cancer_pm_test.json",
        "holdout": "data/final/cancer_pm_holdout.json"
    }
    results = {}
    for doi in dois:
        results[doi] = "Not found"
        
    for split_name, path in splits.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for r in data:
                doi = str(r.get("doi", "")).lower()
                if doi in dois:
                    results[doi] = split_name
    return results

def main():
    print("="*80)
    print("HARD NEGATIVE STRESS TEST & ATTRIBUTION QUANTIFICATION")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Split location for 3 adversarial DOIs
    target_dois = ["10.1155/2020/8891876", "10.1155/2020/8124570", "10.1002/iub.2012"]
    split_locations = find_split_for_dois(target_dois)
    print("\n--- 1. SPLIT LOCATIONS FOR ADVERSARIAL DOIs ---")
    for doi, split in split_locations.items():
        print(f"DOI: {doi} -> Split: {split}")

    # 2. Hard Negative Stress Test
    with open("data/final/cancer_pm_holdout_normalized.json", "r", encoding="utf-8") as f:
        holdout_records = json.load(f)
        
    negatives = [r for r in holdout_records if r["label"] == 0]
    
    # TF-IDF false positive driver terms
    driver_terms = ["microrna", "mir-", "mirna", "lncrna", "apoptosis", "noncoding", "circrna", "ce-rna", "cerna"]
    
    def score_density(text):
        t = text.lower()
        return sum(t.count(term) for term in driver_terms)
        
    for r in negatives:
        r["density_score"] = score_density(r["title"] + " " + r["abstract"])
        
    negatives.sort(key=lambda x: x["density_score"], reverse=True)
    hard_negatives = negatives[:100]
    
    print(f"\n--- 2. HARD NEGATIVE STRESS TEST ---")
    print(f"Selected top {len(hard_negatives)} negatives by density of molecular biology terms.")
    
    # Load Model
    pm_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(pm_model_name)
    model_path = "models/pubmedbert_norm_non_hindawi_best.pt"
    
    print(f"Loading cross-transfer model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = BERTClassifier(pm_model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    def predict_prob(title_text, abstract_text):
        full_text = title_text + " " + abstract_text
        encoding = tokenizer(
            full_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            logit = model(input_ids, mask)
            prob = torch.sigmoid(logit.squeeze(-1)).item()
        return prob
        
    fp_count = 0
    probs = []
    for r in hard_negatives:
        prob = predict_prob(r["title"], r["abstract"])
        probs.append(prob)
        if prob >= 0.5:
            fp_count += 1
            
    fpr = fp_count / len(hard_negatives) * 100
    print(f"False Positive Rate on Top 100 Hard Negatives: {fp_count} / {len(hard_negatives)} ({fpr:.2f}%)")
    print(f"Mean predicted probability among hard negatives: {sum(probs)/len(probs):.6f}")
    
    # 3. Quantify random-sample attribution phrases
    print("\n--- 3. ATTRIBUTION PHRASE PREVALENCE (HINDAWI HOLDOUT) ---")
    phrases_to_test = [
        "breast cancer frequently experience psychological",
        "Renal cell carcinoma (RCC) accounts",
        "OBJECTIVE: Identifying gene mutation signatures",
        "The goal of this study",
        "One of the primary causes",
        "OBJECTIVE: The change of bacterial",
        "diagnosis of a liver tumor",
        "automatic biopsy guided by ultrasound",
        "Thyroid cancer (TC) is the",
        "OBJECTIVE: To determine the effects"
    ]
    
    hindawi_pos = [r for r in holdout_records if r["label"] == 1]
    hindawi_neg = [r for r in holdout_records if r["label"] == 0]
    
    for phrase in phrases_to_test:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        pos_match = sum(1 for r in hindawi_pos if pattern.search(r["abstract"]))
        neg_match = sum(1 for r in hindawi_neg if pattern.search(r["abstract"]))
        pos_pct = pos_match / len(hindawi_pos) * 100 if hindawi_pos else 0
        neg_pct = neg_match / len(hindawi_neg) * 100 if hindawi_neg else 0
        
        print(f"\nPhrase: '{phrase}'")
        print(f"  Positives: {pos_match} ({pos_pct:.2f}%)")
        print(f"  Negatives: {neg_match} ({neg_pct:.2f}%)")
        
if __name__ == '__main__':
    main()
