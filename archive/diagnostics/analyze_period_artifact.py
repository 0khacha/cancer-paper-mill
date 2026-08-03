import json
import csv
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

def get_publisher(rec, journal_to_publisher, nlm2raw):
    j = rec.get('journal', '')
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, 'Direct PubMed')
    if 'computational and mathematical methods in medicine' in raw_j.lower() or 'comput math methods med' in j.lower():
        pub = 'Hindawi'
    elif 'biomed research international' in raw_j.lower():
        pub = 'Hindawi'
        
    # Group into Hindawi, Spandidos, Others
    if pub == 'Hindawi':
        return 'Hindawi'
    elif pub == 'Spandidos Publications':
        return 'Spandidos'
    else:
        return 'Others'

def analyze():
    # Load journal mappings
    with open('data/final/journal_to_nlm.json', 'r', encoding='utf-8') as f:
        j2nlm = json.load(f)
    nlm2raw = {v: k for k, v in j2nlm.items()}
    
    journal_to_publisher = {}
    with open('data/raw/rwdb/retraction_watch.csv', 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            j = row.get('Journal', '').strip()
            p = row.get('Publisher', '').strip()
            if j and p:
                journal_to_publisher[j] = p

    splits = ['train', 'val', 'test', 'holdout']
    
    # Data structures for modeling
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    X_holdout, y_holdout = [], []
    
    def extract_feature(title, abstract):
        text = title + " " + abstract
        if len(title) > 0 and text[len(title)-1] == '.':
            return [1]
        return [0]
    
    print("="*60)
    print("1. ARTIFACT QUANTIFICATION PER SPLIT AND STRATUM")
    print("="*60)
    
    for split in splits:
        with open(f'data/final/cancer_pm_{split}.json', 'r', encoding='utf-8') as f:
            records = json.load(f)
            
        pos = [r for r in records if r['label'] == 1]
        neg = [r for r in records if r['label'] == 0]
        
        # Calculate split stats
        pos_dots = sum(1 for r in pos if r.get('title', '').strip().endswith('.'))
        neg_dots = sum(1 for r in neg if r.get('title', '').strip().endswith('.'))
        
        print(f"\nSPLIT: {split.upper()} (Total: {len(records)})")
        print(f"  Positives ending in '.': {pos_dots} / {len(pos)} ({pos_dots/max(1, len(pos))*100:.2f}%)")
        print(f"  Negatives ending in '.': {neg_dots} / {len(neg)} ({neg_dots/max(1, len(neg))*100:.2f}%)")
        
        # Calculate stratum stats
        strata = {'Hindawi': {'pos': [], 'neg': []}, 
                  'Spandidos': {'pos': [], 'neg': []}, 
                  'Others': {'pos': [], 'neg': []}}
                  
        for r in records:
            pub = get_publisher(r, journal_to_publisher, nlm2raw)
            if r['label'] == 1:
                strata[pub]['pos'].append(r)
            else:
                strata[pub]['neg'].append(r)
                
            # Build dataset for Logistic Regression
            feat = extract_feature(r.get('title', ''), r.get('abstract', ''))
            label = r['label']
            if split == 'train':
                X_train.append(feat)
                y_train.append(label)
            elif split == 'val':
                X_val.append(feat)
                y_val.append(label)
            elif split == 'test':
                X_test.append(feat)
                y_test.append(label)
            elif split == 'holdout':
                X_holdout.append(feat)
                y_holdout.append(label)
                
        for pub_name, pub_data in strata.items():
            p_recs = pub_data['pos']
            n_recs = pub_data['neg']
            if len(p_recs) == 0 and len(n_recs) == 0:
                continue
            
            p_dots = sum(1 for r in p_recs if r.get('title', '').strip().endswith('.'))
            n_dots = sum(1 for r in n_recs if r.get('title', '').strip().endswith('.'))
            
            print(f"    Stratum: {pub_name}")
            if len(p_recs) > 0:
                print(f"      Positives w/ dot: {p_dots} / {len(p_recs)} ({p_dots/len(p_recs)*100:.2f}%)")
            if len(n_recs) > 0:
                print(f"      Negatives w/ dot: {n_dots} / {len(n_recs)} ({n_dots/len(n_recs)*100:.2f}%)")
                
        # 4. Check for other systemic differences in this split
        # We'll just check train split for other differences
        if split == 'train':
            print("\n" + "="*60)
            print("4. OTHER FORMATTING DIFFERENCES (TRAIN SPLIT)")
            print("="*60)
            
            p_titles_upper = sum(1 for r in pos if r.get('title', '').strip() and r.get('title', '').strip()[0].isupper())
            n_titles_upper = sum(1 for r in neg if r.get('title', '').strip() and r.get('title', '').strip()[0].isupper())
            
            p_abs_upper = sum(1 for r in pos if r.get('abstract', '').strip() and r.get('abstract', '').strip()[0].isupper())
            n_abs_upper = sum(1 for r in neg if r.get('abstract', '').strip() and r.get('abstract', '').strip()[0].isupper())
            
            print(f"  Pos titles starting w/ Uppercase: {p_titles_upper} / {len(pos)} ({p_titles_upper/max(1, len(pos))*100:.2f}%)")
            print(f"  Neg titles starting w/ Uppercase: {n_titles_upper} / {len(neg)} ({n_titles_upper/max(1, len(neg))*100:.2f}%)")
            
            print(f"  Pos abstracts starting w/ Uppercase: {p_abs_upper} / {len(pos)} ({p_abs_upper/max(1, len(pos))*100:.2f}%)")
            print(f"  Neg abstracts starting w/ Uppercase: {n_abs_upper} / {len(neg)} ({n_abs_upper/max(1, len(neg))*100:.2f}%)")

    print("\n" + "="*60)
    print("3. ISOLATION PROBE (LOGISTIC REGRESSION ON 1 FEATURE)")
    print("="*60)
    
    # Note: we predict POSITIVE (label 1), but the period means NEGATIVE (label 0).
    # So the logistic regression will learn a strongly negative weight for the feature.
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    print(f"Model trained on {len(X_train)} samples. Feature weight: {clf.coef_[0][0]:.4f}")
    
    def eval_probe(name, X, y):
        probs = clf.predict_proba(X)[:, 1]
        preds = clf.predict(X)
        auc = roc_auc_score(y, probs)
        f1 = f1_score(y, preds)
        print(f"  {name} - AUC: {auc:.4%}, F1: {f1:.4%}")
        
    eval_probe("Train", X_train, y_train)
    eval_probe("Val", X_val, y_val)
    eval_probe("Test", X_test, y_test)
    eval_probe("Holdout", X_holdout, y_holdout)

if __name__ == "__main__":
    analyze()
