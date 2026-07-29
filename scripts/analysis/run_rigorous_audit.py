import json
import csv
import sys
import os
import re
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict

# Redirect output to file
output_path = r"C:\projects\cancer-paper-mill\rigorous_audit_results.txt"
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
    print("RIGOROUS AUDIT: HISTOGRAMS, JOURNAL SKUES, AND DIRECT ATTRIBUTION")
    print("="*80)

    # Load partitions
    with open("data/final/cancer_pm_train.json", "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open("data/final/cancer_pm_holdout.json", "r", encoding="utf-8") as f:
        holdout_records = json.load(f)
    with open("data/final/cancer_pm_val.json", "r", encoding="utf-8") as f:
        val_records = json.load(f)
    with open("data/final/cancer_pm_test.json", "r", encoding="utf-8") as f:
        test_records = json.load(f)

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

    # -------------------------------------------------------------
    # PART 1: FULL SIMILARITY DISTRIBUTION (Nearest Neighbor Bins)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("1. FULL NEAREST-NEIGHBOR SIMILARITY DISTRIBUTION BETWEEN HOLDOUT & TRAIN")
    print("="*70)

    train_hindawi_pos = [r for r in train_records if r["label"] == 1 and get_publisher(r, journal_to_publisher, nlm2raw) == 'Hindawi']
    holdout_hindawi_pos = [r for r in holdout_records if r["label"] == 1]

    # Character 3-gram Jaccard
    max_jaccards = []
    train_titles_grams = [get_3grams(r["title"]) for r in train_hindawi_pos]

    for h_rec in holdout_hindawi_pos:
        h_grams = get_3grams(h_rec["title"])
        best_jac = 0.0
        if h_grams:
            for t_grams in train_titles_grams:
                if not t_grams:
                    continue
                inter = len(h_grams & t_grams)
                union_len = len(h_grams | t_grams)
                jac = inter / union_len if union_len > 0 else 0
                if jac > best_jac:
                    best_jac = jac
        max_jaccards.append(best_jac)

    # Word TF-IDF Cosine
    train_texts = [r["title"] + " " + r["abstract"] for r in train_hindawi_pos]
    holdout_texts = [r["title"] + " " + r["abstract"] for r in holdout_hindawi_pos]
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    train_tfidf = vectorizer.fit_transform(train_texts)
    holdout_tfidf = vectorizer.transform(holdout_texts)
    
    cos_sims = cosine_similarity(holdout_tfidf, train_tfidf)
    max_cosines = np.max(cos_sims, axis=1)

    # Helper function to print histogram bin counts
    def print_histogram(scores, title_name):
        bins = np.arange(0.0, 1.1, 0.1)
        counts, edges = np.histogram(scores, bins=bins)
        print(f"Bins and Counts for {title_name}:")
        for i in range(len(counts)):
            print(f"  [{edges[i]:.1f} - {edges[i+1]:.1f}): {counts[i]} records  ({counts[i]/len(scores)*100:.2f}%)")
        print(f"  Total checked: {len(scores)}")

    print_histogram(max_jaccards, "Nearest-Neighbor Title 3-gram Jaccard")
    print()
    print_histogram(max_cosines, "Nearest-Neighbor Title+Abstract TF-IDF Cosine (Unigrams + Bigrams)")

    # -------------------------------------------------------------
    # PART 2: NEGATIVES JOURNAL composition skew check
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("2. DEVIATION SKEW CHECK: HOLDOUT VS. IN-POOL HINDAWI NEGATIVE JOURNAL COMPOSITION")
    print("="*70)

    # Hindawi negatives in partition list
    holdout_hindawi_neg = [r for r in holdout_records if r["label"] == 0]
    
    # In-pool Hindawi negatives can be found in train, val, and test partitions
    in_pool_recs = train_records + val_records + test_records
    in_pool_hindawi_neg = [r for r in in_pool_recs if r["label"] == 0 and get_publisher(r, journal_to_publisher, nlm2raw) == 'Hindawi']

    print(f"Total negatives in Hindawi Holdout: {len(holdout_hindawi_neg)}")
    print(f"Total negatives in Hindawi In-Pool: {len(in_pool_hindawi_neg)}")

    holdout_journals = [r["journal"] for r in holdout_hindawi_neg]
    in_pool_journals = [r["journal"] for r in in_pool_hindawi_neg]

    holdout_counts = Counter(holdout_journals)
    in_pool_counts = Counter(in_pool_journals)
    
    all_uniq_journals = sorted(list(set(holdout_journals + in_pool_journals)))
    
    print(f"\n{'Journal Name':<45} | {'Holdout Neg Count (Pct)':<25} | {'In-Pool Neg Count (Pct)':<25} | {'Skew (Holdout% - Pool%)':<25}")
    print("-" * 125)
    for j in all_uniq_journals:
        h_cnt = holdout_counts.get(j, 0)
        p_cnt = in_pool_counts.get(j, 0)
        h_pct = h_cnt / len(holdout_journals) * 100 if len(holdout_journals) > 0 else 0.0
        p_pct = p_cnt / len(in_pool_journals) * 100 if len(in_pool_journals) > 0 else 0.0
        skew = h_pct - p_pct
        raw_j_name = nlm2raw.get(j, j)
        print(f"{raw_j_name[:43]:<45} | {h_cnt:>3} ({h_pct:>5.2f}%)            | {p_cnt:>3} ({p_pct:>5.2f}%)            | {skew:>+6.2f}%")

    # -------------------------------------------------------------
    # PART 3: DIRECT ATTRIBUTION/INTERPRETABILITY LOGS
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("3. DIRECT ATTRIBUTION/INTERPRETABILITY CHECK (LEAVE-ONE-SENTENCE-OUT)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {device}")

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

    val_hindawi = [r for r in val_records if get_publisher(r, journal_to_publisher, nlm2raw) == 'Hindawi']
    positives_val = [r for r in val_hindawi if r["label"] == 1]

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

    # Load Model
    pm_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    tokenizer_pm = AutoTokenizer.from_pretrained(pm_model_name)
    checkpoint_pm = torch.load("models/pubmedbert_finetuned_best.pt", map_location=device, weights_only=False)
    model_pm = BERTClassifier(pm_model_name)
    model_pm.load_state_dict(checkpoint_pm["model_state_dict"])
    model_pm.to(device)
    model_pm.eval()

    def predict_prob(title_text, abstract_text):
        full_text = title_text + " " + abstract_text
        encoding = tokenizer_pm(
            full_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            logit = model_pm(input_ids, mask)
            prob = torch.sigmoid(logit.squeeze(-1)).item()
        return prob

    # Find highest confidence SG B papers
    sg_b_predictions = []
    for r in sg_b:
        prob = predict_prob(r["title"], r["abstract"])
        sg_b_predictions.append((prob, r))
    
    # Sort by confidence descending
    sg_b_predictions.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Top 5 most confident true positives of Sous-groupe B (probabilities from {sg_b_predictions[0][0]:.8f} to {sg_b_predictions[4][0]:.8f}):")

    for k in range(5):
        orig_prob, r = sg_b_predictions[k]
        title = r["title"]
        abstract = r["abstract"]
        
        print("\n" + "-"*80)
        print(f"PAPER {k+1}: DOI: {r['doi']} | CONFIDENCE: {orig_prob:.8f}")
        print(f"TITLE: {title}")
        print("-"*80)
        
        # 1. Sentence-level perturbation
        # Splitting sentences by period followed by whitespace
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', abstract) if s.strip()]
        
        sentence_attributions = []
        for i, s in enumerate(sentences):
            # Omit sentence s
            omitted_abstract = " ".join([sentences[j] for j in range(len(sentences)) if j != i])
            prob_without = predict_prob(title, omitted_abstract)
            drop = orig_prob - prob_without
            sentence_attributions.append((drop, s, prob_without))
        
        # Sort by drop descending
        sentence_attributions.sort(key=lambda x: x[0], reverse=True)
        
        print("\nSentence Attributions:")
        for idx, (drop, s, p_w) in enumerate(sentence_attributions[:4]):
            print(f"  Rank {idx+1}: [Delta Prob: {drop:>+.8f} | Prob without: {p_w:.8f}]")
            print(f"    Text: \"{s}\"")

        # 2. Phrase-level perturbation (N-gram sliding window of length 5 words)
        words = abstract.split()
        phrase_attributions = []
        
        # Check sliding windows of 5 words
        W = 5
        if len(words) >= W:
            for i in range(len(words) - W + 1):
                phrase = " ".join(words[i:i+W])
                # Omit this phrase
                omitted_words = words[:i] + words[i+W:]
                omitted_abstract = " ".join(omitted_words)
                prob_without = predict_prob(title, omitted_abstract)
                drop = orig_prob - prob_without
                phrase_attributions.append((drop, phrase, prob_without))
                
            phrase_attributions.sort(key=lambda x: x[0], reverse=True)
            print("\nTop 5-Word Phrase Attributions:")
            for idx, (drop, phr, p_w) in enumerate(phrase_attributions[:5]):
                print(f"  Rank {idx+1}: [Delta Prob: {drop:>+.8f}]  \"{phr}\"")
        else:
            print("\nAbstract too short for sliding window.")

    print("\nRigorous audit finished.")

if __name__ == "__main__":
    main()
