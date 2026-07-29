import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import xgboost as xgb

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def get_publisher(rec, journal_to_publisher, nlm2raw):
    j = rec['journal']
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() or "comput math methods med" in j.lower():
        pub = "Hindawi"
    return pub

def evaluate_predictions(probs, labels, threshold_range=np.arange(0.1, 0.9, 0.01)):
    # If probs are decision function values (for LinearSVC), we sweep thresholds over the decision function range
    # Or we can calibrate them to probabilities. For LinearSVC, decision function is standard.
    # Let's search threshold that maximizes F1
    best_thresh = 0.5
    best_f1 = 0.0
    for t in threshold_range:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    preds_best = (probs >= best_thresh).astype(int)
    prec = precision_score(labels, preds_best, zero_division=0)
    rec = recall_score(labels, preds_best, zero_division=0)
    auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    
    return {
        "best_threshold": best_thresh,
        "precision": prec,
        "recall": rec,
        "f1": best_f1,
        "auc": auc,
        "pr_auc": pr_auc,
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
    return {"f1": f1, "prec": prec, "rec": rec, "auc": auc, "n_pos": n_pos, "n_neg": n_neg}

def main():
    print("Loading data...")
    with open("data/final/cancer_pm_train.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open("data/final/cancer_pm_val.json", "r", encoding="utf-8") as f:
        val = json.load(f)

    train_texts = [r["title"] + " " + r["abstract"] for r in train]
    train_labels = np.array([r["label"] for r in train])
    val_texts = [r["title"] + " " + r["abstract"] for r in val]
    val_labels = np.array([r["label"] for r in val])

    # Load publisher dictionaries for stratification
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
                
    with open('data/final/journal_to_nlm.json', 'r', encoding='utf-8') as f:
        j2nlm = json.load(f)
    nlm2raw = {v: k for k, v in j2nlm.items()}

    # TF-IDF Extraction (min_df=5, sublinear_tf=True)
    print("Extracting TF-IDF features (min_df=5)...")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)
    X_val = vec.transform(val_texts)

    # 1. Linear SVM
    print("\nTraining Linear SVM (LinearSVC)...")
    svm = LinearSVC(class_weight="balanced", random_state=42, dual="auto")
    svm.fit(X_train, train_labels)
    # Get decision function values (analogous to probabilities for thresholding)
    svm_dec = svm.decision_function(X_val)
    # Map decision values to probabilities using sigmoid for better metric alignment (or just threshold raw decision function)
    # We can threshold the decision function itself. Threshold 0.0 is the standard SVM boundary.
    svm_res = evaluate_predictions(svm_dec, val_labels, threshold_range=np.arange(-2.0, 2.0, 0.05))
    
    print("\n" + "="*50)
    print("LINEAR SVM VALIDATION METRICS")
    print("="*50)
    print(f"Optimal Threshold:  {svm_res['best_threshold']:.2f}")
    print(f"Overall Precision:  {svm_res['precision']:.4%}")
    print(f"Overall Recall:     {svm_res['recall']:.4%}")
    print(f"Overall F1 Score:   {svm_res['f1']:.4%}")
    print(f"Overall ROC-AUC:    {svm_res['auc']:.4%}")
    print(f"Overall PR-AUC:     {svm_res['pr_auc']:.4%}")
    
    print("\n--- STRATIFIED BREAKDOWN (Linear SVM) ---")
    h_labels_svm = [val_labels[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) == "Hindawi"]
    h_probs_svm = [svm_dec[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) == "Hindawi"]
    h_m_svm = get_stratified_metrics(h_labels_svm, h_probs_svm, svm_res["best_threshold"], "Hindawi")
    
    s_labels_svm = [val_labels[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) == "Spandidos"]
    s_probs_svm = [svm_dec[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) == "Spandidos"]
    s_m_svm = get_stratified_metrics(s_labels_svm, s_probs_svm, svm_res["best_threshold"], "Spandidos")
    
    o_labels_svm = [val_labels[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) not in ["Hindawi", "Spandidos"]]
    o_probs_svm = [svm_dec[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) not in ["Hindawi", "Spandidos"]]
    o_m_svm = get_stratified_metrics(o_labels_svm, o_probs_svm, svm_res["best_threshold"], "Pooled Others")

    # 2. XGBoost
    print("\nTraining XGBoost...")
    # Calculate scale_pos_weight
    n_neg = (train_labels == 0).sum()
    n_pos = (train_labels == 1).sum()
    scale_pos = n_neg / n_pos
    print(f"XGBoost scale_pos_weight: {scale_pos:.4f}")
    
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    xgb_clf.fit(X_train, train_labels)
    xgb_probs = xgb_clf.predict_proba(X_val)[:, 1]
    xgb_res = evaluate_predictions(xgb_probs, val_labels, threshold_range=np.arange(0.1, 0.9, 0.01))
    
    print("\n" + "="*50)
    print("XGBOOST VALIDATION METRICS")
    print("="*50)
    print(f"Optimal Threshold:  {xgb_res['best_threshold']:.2f}")
    print(f"Overall Precision:  {xgb_res['precision']:.4%}")
    print(f"Overall Recall:     {xgb_res['recall']:.4%}")
    print(f"Overall F1 Score:   {xgb_res['f1']:.4%}")
    print(f"Overall ROC-AUC:    {xgb_res['auc']:.4%}")
    print(f"Overall PR-AUC:     {xgb_res['pr_auc']:.4%}")
    
    print("\n--- STRATIFIED BREAKDOWN (XGBoost) ---")
    h_labels_xgb = [val_labels[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) == "Hindawi"]
    h_probs_xgb = [xgb_probs[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) == "Hindawi"]
    h_m_xgb = get_stratified_metrics(h_labels_xgb, h_probs_xgb, xgb_res["best_threshold"], "Hindawi")
    
    s_labels_xgb = [val_labels[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) == "Spandidos"]
    s_probs_xgb = [xgb_probs[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) == "Spandidos"]
    s_m_xgb = get_stratified_metrics(s_labels_xgb, s_probs_xgb, xgb_res["best_threshold"], "Spandidos")
    
    o_labels_xgb = [val_labels[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) not in ["Hindawi", "Spandidos"]]
    o_probs_xgb = [xgb_probs[idx] for idx, r in enumerate(val) if get_publisher(r, journal_to_publisher, nlm2raw) not in ["Hindawi", "Spandidos"]]
    o_m_xgb = get_stratified_metrics(o_labels_xgb, o_probs_xgb, xgb_res["best_threshold"], "Pooled Others")

    # Save results
    save_results = {
        "svm": {
            "overall": {
                "threshold": svm_res["best_threshold"],
                "precision": svm_res["precision"],
                "recall": svm_res["recall"],
                "f1": svm_res["f1"],
                "auc": svm_res["auc"],
                "pr_auc": svm_res["pr_auc"]
            },
            "stratified": {
                "hindawi": h_m_svm,
                "spandidos": s_m_svm,
                "others": o_m_svm
            }
        },
        "xgboost": {
            "overall": {
                "threshold": xgb_res["best_threshold"],
                "precision": xgb_res["precision"],
                "recall": xgb_res["recall"],
                "f1": xgb_res["f1"],
                "auc": xgb_res["auc"],
                "pr_auc": xgb_res["pr_auc"]
            },
            "stratified": {
                "hindawi": h_m_xgb,
                "spandidos": s_m_xgb,
                "others": o_m_xgb
            }
        }
    }
    
    os.makedirs("models", exist_ok=True)
    with open("models/classical_baselines_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print("\nResults saved to models/classical_baselines_results.json")

if __name__ == "__main__":
    main()
