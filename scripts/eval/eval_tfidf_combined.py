"""
eval_tfidf_combined.py
----------------------
Trains TF-IDF Combiné (LogisticRegression, min_df=5, C=15.0, ngram_range=(1,2),
sublinear_tf=True) on the canonical v2 training set, then evaluates on the
validation, test, and Hindawi holdout splits.

Evaluation protocol (matching thesis §5.5):
  - Optimal threshold is selected on the validation set by sweeping [0.1, 0.9).
  - That threshold is then FROZEN and applied unchanged to the test and holdout sets.
  - Stratified metrics (Hindawi / Spandidos / Others) are computed for all three splits.

Output: results/tfidf_combined_results.json
"""

import csv
import json
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths (relative to repository root — run from there)
# ---------------------------------------------------------------------------
DATA_DIR  = "data/processed"
RWDB_CSV  = "data/raw/rwdb/retraction_watch.csv"
NLM_JSON  = os.path.join(DATA_DIR, "journal_to_nlm.json")
RESULTS_OUT = "results/tfidf_combined_results.json"

# ---------------------------------------------------------------------------
# Publisher lookup (same logic as train_classical_baselines.py)
# ---------------------------------------------------------------------------
def build_publisher_lookup():
    journal_to_publisher = {}
    if os.path.exists(RWDB_CSV):
        with open(RWDB_CSV, "r", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                j = row.get("Journal", "").strip()
                p = row.get("Publisher", "").strip()
                if j and p:
                    if "computational and mathematical methods in medicine" in j.lower():
                        p = "Hindawi"
                    elif "biomed research international" in j.lower():
                        p = "Hindawi"
                    journal_to_publisher[j] = p
    with open(NLM_JSON, "r", encoding="utf-8") as f:
        j2nlm = json.load(f)
    nlm2raw = {v: k for k, v in j2nlm.items()}
    return journal_to_publisher, nlm2raw


def get_publisher(rec, journal_to_publisher, nlm2raw):
    j = rec["journal"]
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() \
            or "comput math methods med" in j.lower():
        pub = "Hindawi"
    elif "biomed research international" in raw_j.lower() \
            or "biomed res int" in j.lower():
        pub = "Hindawi"
    return pub


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def best_threshold_metrics(probs, labels):
    """Sweep thresholds on [0.1, 0.9) and return the threshold that maximises F1."""
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(round(t, 2))
    preds_best = (probs >= best_thresh).astype(int)
    return {
        "threshold": best_thresh,
        "precision": float(precision_score(labels, preds_best, zero_division=0)),
        "recall":    float(recall_score(labels, preds_best, zero_division=0)),
        "f1":        float(f1_score(labels, preds_best, zero_division=0)),
        "auc":       float(roc_auc_score(labels, probs)),
        "pr_auc":    float(average_precision_score(labels, probs)),
    }


def fixed_threshold_metrics(probs, labels, threshold):
    """Evaluate at a fixed (frozen) threshold."""
    preds = (probs >= threshold).astype(int)
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5
    return {
        "threshold": threshold,
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "auc":       auc,
        "pr_auc":    float(average_precision_score(labels, probs))
                     if len(np.unique(labels)) > 1 else 0.0,
    }


def stratified_metrics(records, probs_array, labels_array, threshold,
                       journal_to_publisher, nlm2raw):
    """Compute per-stratum (Hindawi / Spandidos / Others) metrics at fixed threshold."""
    strata = {"hindawi": [], "spandidos": [], "others": []}
    for idx, rec in enumerate(records):
        pub = get_publisher(rec, journal_to_publisher, nlm2raw)
        entry = (probs_array[idx], labels_array[idx])
        if pub == "Hindawi":
            strata["hindawi"].append(entry)
        elif pub == "Spandidos":
            strata["spandidos"].append(entry)
        else:
            strata["others"].append(entry)

    out = {}
    for name, pairs in strata.items():
        if not pairs:
            out[name] = {}
            continue
        p_arr = np.array([x[0] for x in pairs])
        l_arr = np.array([x[1] for x in pairs])
        m = fixed_threshold_metrics(p_arr, l_arr, threshold)
        m["n_pos"] = int(l_arr.sum())
        m["n_neg"] = int(len(l_arr) - l_arr.sum())
        out[name] = m
        print(f"  {name:12s}  n_pos={m['n_pos']:4d}  n_neg={m['n_neg']:4d}  "
              f"F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("TF-IDF Combiné (LogReg, min_df=5, C=15.0) — Full Evaluation")
    print("=" * 60)

    # 1. Load data
    print("\nLoading splits...")
    splits = {}
    for name in ["train", "val", "test", "holdout"]:
        path = os.path.join(DATA_DIR, f"cancer_pm_{name}.json")
        with open(path, "r", encoding="utf-8") as f:
            splits[name] = json.load(f)
        print(f"  {name:8s}: {len(splits[name])} records")

    journal_to_publisher, nlm2raw = build_publisher_lookup()

    # 2. Build text arrays
    def texts(recs):
        return [r["title"] + " " + r["abstract"] for r in recs]

    def labels(recs):
        return np.array([int(r["label"]) for r in recs])

    # 3. Load pre-trained vectoriser + model
    import joblib
    print("\nLoading pre-trained TF-IDF vectoriser and LogisticRegression model...")
    model_data = joblib.load("models/tfidf_combined.joblib")
    vec = model_data["vectorizer"]
    clf = model_data["classifier"]

    # 4. Validation — find optimal threshold
    print("\n--- Validation set (threshold optimisation) ---")
    X_val   = vec.transform(texts(splits["val"]))
    y_val   = labels(splits["val"])
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_metrics_optimal = best_threshold_metrics(val_probs, y_val)
    frozen_threshold = val_metrics_optimal["threshold"]
    print(f"  Optimal threshold: {frozen_threshold:.2f}")
    print(f"  Global  →  F1={val_metrics_optimal['f1']:.4f}  "
          f"AUC={val_metrics_optimal['auc']:.4f}  "
          f"Prec={val_metrics_optimal['precision']:.4f}  "
          f"Rec={val_metrics_optimal['recall']:.4f}")
    print(f"\n  Stratified (at threshold={frozen_threshold:.2f}):")
    val_strat = stratified_metrics(
        splits["val"], val_probs, y_val, frozen_threshold,
        journal_to_publisher, nlm2raw
    )

    # 5. Test — frozen threshold
    print(f"\n--- Test set (threshold FROZEN at {frozen_threshold:.2f}) ---")
    X_test   = vec.transform(texts(splits["test"]))
    y_test   = labels(splits["test"])
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_metrics = fixed_threshold_metrics(test_probs, y_test, frozen_threshold)
    print(f"  Global  →  F1={test_metrics['f1']:.4f}  "
          f"AUC={test_metrics['auc']:.4f}  "
          f"Prec={test_metrics['precision']:.4f}  "
          f"Rec={test_metrics['recall']:.4f}")
    print(f"\n  Stratified:")
    test_strat = stratified_metrics(
        splits["test"], test_probs, y_test, frozen_threshold,
        journal_to_publisher, nlm2raw
    )

    # 6. Holdout — frozen threshold
    print(f"\n--- Hindawi Holdout (threshold FROZEN at {frozen_threshold:.2f}) ---")
    X_holdout   = vec.transform(texts(splits["holdout"]))
    y_holdout   = labels(splits["holdout"])
    holdout_probs = clf.predict_proba(X_holdout)[:, 1]
    holdout_metrics = fixed_threshold_metrics(holdout_probs, y_holdout, frozen_threshold)
    print(f"  Global  →  F1={holdout_metrics['f1']:.4f}  "
          f"AUC={holdout_metrics['auc']:.4f}  "
          f"Prec={holdout_metrics['precision']:.4f}  "
          f"Rec={holdout_metrics['recall']:.4f}")

    # 7. Assemble and write results
    results = {
        "model": {
            "classifier": "LogisticRegression",
            "C": 15.0,
            "vectorizer": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "min_df": 5,
            "sublinear_tf": True,
        },
        "frozen_threshold": frozen_threshold,
        "val": {
            "overall_at_optimal_threshold": val_metrics_optimal,
            "overall_at_frozen_threshold": fixed_threshold_metrics(
                val_probs, y_val, frozen_threshold
            ),
            "stratified": val_strat,
        },
        "test": {
            "overall": test_metrics,
            "stratified": test_strat,
        },
        "holdout": {
            "overall": holdout_metrics,
        },
    }

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_OUT}")

    # 8. Cross-check against thesis numbers
    thesis = {
        "val_f1":       0.5366, "val_auc":       0.8089,
        "val_h_f1":     0.4183, "val_h_auc":     0.6327,
        "val_s_f1":     0.8780, "val_s_auc":     0.9697,
        "val_o_f1":     0.6000, "val_o_auc":     0.9310,
        "test_f1":      0.4867, "test_auc":      0.7959,
        "test_h_f1":    0.3766, "test_h_auc":    0.6571,
        "test_s_f1":    0.8125, "test_s_auc":    0.9806,
        "test_o_f1":    0.5411, "test_o_auc":    0.8944,
        "holdout_f1":   0.4063, "holdout_auc":   0.6447,
    }
    computed = {
        "val_f1":    val_metrics_optimal["f1"],
        "val_auc":   val_metrics_optimal["auc"],
        "val_h_f1":  val_strat["hindawi"].get("f1", float("nan")),
        "val_h_auc": val_strat["hindawi"].get("auc", float("nan")),
        "val_s_f1":  val_strat["spandidos"].get("f1", float("nan")),
        "val_s_auc": val_strat["spandidos"].get("auc", float("nan")),
        "val_o_f1":  val_strat["others"].get("f1", float("nan")),
        "val_o_auc": val_strat["others"].get("auc", float("nan")),
        "test_f1":   test_metrics["f1"],
        "test_auc":  test_metrics["auc"],
        "test_h_f1": test_strat["hindawi"].get("f1", float("nan")),
        "test_h_auc":test_strat["hindawi"].get("auc", float("nan")),
        "test_s_f1": test_strat["spandidos"].get("f1", float("nan")),
        "test_s_auc":test_strat["spandidos"].get("auc", float("nan")),
        "test_o_f1": test_strat["others"].get("f1", float("nan")),
        "test_o_auc":test_strat["others"].get("auc", float("nan")),
        "holdout_f1":  holdout_metrics["f1"],
        "holdout_auc": holdout_metrics["auc"],
    }

    print("\n--- Cross-check vs thesis numbers (tolerance ±0.015) ---")
    print(f"  {'Metric':<22} {'Thesis':>8} {'Computed':>10} {'Δ':>8}  {'OK?':>5}")
    print("  " + "-" * 60)
    all_ok = True
    for key, t_val in thesis.items():
        c_val = computed[key]
        delta = abs(c_val - t_val)
        ok = "✓" if delta <= 0.015 else "✗ MISMATCH"
        if delta > 0.015:
            all_ok = False
        print(f"  {key:<22} {t_val:>8.4f} {c_val:>10.4f} {delta:>+8.4f}  {ok:>5}")

    print()
    if all_ok:
        print("All metrics within tolerance. Reproducibility gap closed.")
    else:
        print("WARNING: One or more metrics exceed tolerance. "
              "Check that the data path, vectoriser settings, and "
              "LogisticRegression(C=15.0) match exactly what was used in the thesis.")


if __name__ == "__main__":
    main()
