"""
train_tfidf_combined.py
-----------------------
This configuration was inferred by matching previously-published results; it is not independently verified against a recovered original script.
"""
import json
import os
import sys
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding='utf-8')

def get_publisher(rec, journal_to_publisher, nlm2raw):
    j = rec['journal']
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() or "comput math methods med" in j.lower():
        pub = "Hindawi"
    elif "biomed research international" in raw_j.lower() or "biomed res int" in j.lower():
        pub = "Hindawi"
    return pub

def evaluate_predictions(probs, labels, threshold_range=np.arange(0.1, 0.9, 0.01)):
    best_thresh = 0.5
    best_f1 = 0.0
    for t in threshold_range:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(round(t, 2))
            
    preds_best = (probs >= best_thresh).astype(int)
    prec = precision_score(labels, preds_best, zero_division=0)
    rec = recall_score(labels, preds_best, zero_division=0)
    auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    
    return {
        "best_threshold": best_thresh,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(best_f1),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "preds": preds_best,
        "probs": probs
    }

def get_stratified_metrics(lbls, prbs, thresh, name):
    lbls = np.array(lbls)
    prbs = np.array(prbs)
    preds = (prbs >= thresh).astype(int)
    
    f1 = f1_score(lbls, preds, zero_division=0)
    prec = precision_score(lbls, preds, zero_division=0)
    rec = recall_score(lbls, preds, zero_division=0)
    auc = roc_auc_score(lbls, prbs) if len(np.unique(lbls)) > 1 else 0.5
    
    n_pos = int(lbls.sum())
    n_neg = len(lbls) - n_pos
    
    print(f"Strate {name:<15} (Size: {n_pos} pos, {n_neg} neg):")
    print(f"  F1 Score:   {f1:.4%}")
    print(f"  Precision:  {prec:.4%}")
    print(f"  Recall:     {rec:.4%}")
    print(f"  ROC-AUC:    {auc:.4%}")
    return {"f1": float(f1), "prec": float(prec), "rec": float(rec), "auc": float(auc), "n_pos": n_pos, "n_neg": n_neg}

def main():
    print("Loading data...")
    with open("data/processed/cancer_pm_train.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open("data/processed/cancer_pm_val.json", "r", encoding="utf-8") as f:
        val = json.load(f)

    train_texts = [r["title"] + " " + r["abstract"] for r in train]
    train_labels = np.array([int(r["label"]) for r in train])
    val_texts = [r["title"] + " " + r["abstract"] for r in val]
    val_labels = np.array([int(r["label"]) for r in val])

    import csv
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
                
    with open('data/processed/journal_to_nlm.json', 'r', encoding='utf-8') as f:
        j2nlm = json.load(f)
    nlm2raw = {v: k for k, v in j2nlm.items()}

    print("Extracting TF-IDF features (min_df=5, ngram_range=(1,2), sublinear_tf=True)...")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)
    X_val = vec.transform(val_texts)

    print("\nTraining LogisticRegression (C=10.0, class_weight='balanced', max_iter=1000, random_state=123)...")
    clf = LogisticRegression(C=10.0, class_weight="balanced", max_iter=1000, random_state=123, solver="lbfgs")
    clf.fit(X_train, train_labels)
    
    probs_val = clf.predict_proba(X_val)[:, 1]
    res = evaluate_predictions(probs_val, val_labels, threshold_range=np.arange(0.1, 0.9, 0.01))
    
    print("\n" + "="*50)
    print("TF-IDF COMBINÉ VALIDATION METRICS")
    print("="*50)
    print(f"Optimal Threshold:  {res['best_threshold']:.2f}")
    print(f"Overall Precision:  {res['precision']:.4%}")
    print(f"Overall Recall:     {res['recall']:.4%}")
    print(f"Overall F1 Score:   {res['f1']:.4%}")
    print(f"Overall ROC-AUC:    {res['auc']:.4%}")
    print(f"Overall PR-AUC:     {res['pr_auc']:.4%}")
    
    print("\n--- STRATIFIED BREAKDOWN ---")
    h_labels = [val_labels[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) == "Hindawi"]
    h_probs = [probs_val[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) == "Hindawi"]
    h_m = get_stratified_metrics(h_labels, h_probs, res["best_threshold"], "Hindawi")
    
    s_labels = [val_labels[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) == "Spandidos"]
    s_probs = [probs_val[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) == "Spandidos"]
    s_m = get_stratified_metrics(s_labels, s_probs, res["best_threshold"], "Spandidos")
    
    o_labels = [val_labels[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) not in ["Hindawi", "Spandidos"]]
    o_probs = [probs_val[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) not in ["Hindawi", "Spandidos"]]
    o_m = get_stratified_metrics(o_labels, o_probs, res["best_threshold"], "Pooled Others")

    os.makedirs("models", exist_ok=True)
    
    model_data = {
        "vectorizer": vec,
        "classifier": clf,
        "threshold": res["best_threshold"]
    }
    joblib.dump(model_data, "models/tfidf_combined.joblib")
    print("\nModel and vectorizer saved to models/tfidf_combined.joblib")

if __name__ == "__main__":
    main()
