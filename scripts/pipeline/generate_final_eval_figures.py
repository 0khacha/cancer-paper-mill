import json
import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(project_root, 'data', 'final')
models_dir = os.path.join(project_root, 'models')
figures_dir = os.path.join(project_root, 'docs', 'figures')
os.makedirs(figures_dir, exist_ok=True)

# 1. Verification of data files
required_files = [
    os.path.join(data_dir, 'cancer_pm_train.json'),
    os.path.join(data_dir, 'cancer_pm_val.json'),
    os.path.join(models_dir, 'classical_baselines_results.json'),
    os.path.join(models_dir, 'pubmedbert_finetuned_results.json'),
    os.path.join(models_dir, 'scibert_finetuned_results.json'),
    os.path.join(models_dir, 'significance_results.json')
]

for f in required_files:
    if not os.path.exists(f):
        print(f"ERROR: Required file {f} is missing on disk!")
        sys.exit(1)

def main():
    # Load dataset split files
    with open(os.path.join(data_dir, 'cancer_pm_train.json'), "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(os.path.join(data_dir, 'cancer_pm_val.json'), "r", encoding="utf-8") as f:
        val = json.load(f)

    train_texts = [r["title"] + " " + r["abstract"] for r in train]
    train_labels = np.array([r["label"] for r in train])
    val_texts = [r["title"] + " " + r["abstract"] for r in val]
    val_labels = np.array([r["label"] for r in val])

    # Publisher lookup setup
    journal_to_publisher = {}
    with open(os.path.join(project_root, 'data', 'raw', 'rwdb', 'retraction_watch.csv'), 'r', encoding='utf-8', errors='replace') as f:
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

    with open(os.path.join(data_dir, 'journal_to_nlm.json'), 'r', encoding='utf-8') as f:
        j2nlm = json.load(f)
    nlm2raw = {v: k for k, v in j2nlm.items()}

    def get_publisher(rec):
        j = rec['journal']
        raw_j = nlm2raw.get(j, j)
        pub = journal_to_publisher.get(raw_j, 'Direct PubMed')
        if 'computational and mathematical methods in medicine' in raw_j.lower() or 'comput math methods med' in j.lower():
            pub = 'Hindawi'
        return pub

    val_pubs = [get_publisher(r) for r in val]

    # Recomputing baselines
    print("Recomputing models...")
    
    # Trivial
    preds_trivial = np.ones_like(val_labels)
    probs_trivial = np.ones_like(val_labels, dtype=float)
    
    # TF-IDF Baseline
    vec_base = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_base = vec_base.fit_transform(train_texts)
    X_val_base = vec_base.transform(val_texts)
    clf_base = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    clf_base.fit(X_train_base, train_labels)
    probs_base = clf_base.predict_proba(X_val_base)[:, 1]
    preds_base = (probs_base >= 0.37).astype(int)

    # TF-IDF Combiné
    vec_comb = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    X_train_comb = vec_comb.fit_transform(train_texts)
    X_val_comb = vec_comb.transform(val_texts)
    clf_comb = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced", random_state=42)
    clf_comb.fit(X_train_comb, train_labels)
    probs_comb = clf_comb.predict_proba(X_val_comb)[:, 1]
    preds_comb = (probs_comb >= 0.36).astype(int)

    # Char N-grams
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    X_train_char = vec_char.fit_transform(train_texts)
    X_val_char = vec_char.transform(val_texts)
    clf_char = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    clf_char.fit(X_train_char, train_labels)
    probs_char = clf_char.predict_proba(X_val_char)[:, 1]
    preds_char = (probs_char >= 0.58).astype(int)

    # MiniLM
    print("Encoding MiniLM...")
    model_minilm = SentenceTransformer("all-MiniLM-L6-v2")
    X_train_minilm = model_minilm.encode(train_texts, batch_size=32, show_progress_bar=False)
    X_val_minilm = model_minilm.encode(val_texts, batch_size=32, show_progress_bar=False)
    clf_minilm = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    clf_minilm.fit(X_train_minilm, train_labels)
    probs_minilm = clf_minilm.predict_proba(X_val_minilm)[:, 1]
    best_f1_ml = 0.0
    best_thresh_ml = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs_minilm >= t).astype(int)
        f = f1_score(val_labels, preds)
        if f > best_f1_ml:
            best_f1_ml = f
            best_thresh_ml = t
    preds_minilm = (probs_minilm >= best_thresh_ml).astype(int)

    # PubMedBERT (frozen)
    print("Encoding PubMedBERT...")
    model_pm = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    X_train_pm = model_pm.encode(train_texts, batch_size=32, show_progress_bar=False)
    X_val_pm = model_pm.encode(val_texts, batch_size=32, show_progress_bar=False)
    clf_pm = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    clf_pm.fit(X_train_pm, train_labels)
    probs_pm = clf_pm.predict_proba(X_val_pm)[:, 1]
    preds_pm = (probs_pm >= 0.58).astype(int)

    # Load SVM and XGBoost results
    with open(os.path.join(models_dir, 'classical_baselines_results.json'), "r", encoding="utf-8") as f:
        classical_results = json.load(f)
        
    svm_overall = classical_results["svm"]["overall"]
    svm_strat = classical_results["svm"]["stratified"]
    xgb_overall = classical_results["xgboost"]["overall"]
    xgb_strat = classical_results["xgboost"]["stratified"]

    # Gather overall metrics
    def get_metrics_dict(preds, probs):
        return {
            "f1": f1_score(val_labels, preds),
            "auc": roc_auc_score(val_labels, probs)
        }

    trivial_metrics = get_metrics_dict(preds_trivial, probs_trivial)
    base_metrics = get_metrics_dict(preds_base, probs_base)
    comb_metrics = get_metrics_dict(preds_comb, probs_comb)
    char_metrics = get_metrics_dict(preds_char, probs_char)
    minilm_metrics = get_metrics_dict(preds_minilm, probs_minilm)
    pm_metrics = get_metrics_dict(preds_pm, probs_pm)

    models_list = [
        "Trivial Baseline",
        "TF-IDF Baseline",
        "TF-IDF Combiné",
        "Character N-grams",
        "Embeddings MiniLM",
        "PubMedBERT gelé",
        "SVM Linéaire",
        "XGBoost"
    ]

    f1_scores = [
        trivial_metrics["f1"],
        base_metrics["f1"],
        comb_metrics["f1"],
        char_metrics["f1"],
        minilm_metrics["f1"],
        pm_metrics["f1"],
        svm_overall["f1"],
        xgb_overall["f1"]
    ]

    auc_scores = [
        trivial_metrics["auc"],
        base_metrics["auc"],
        comb_metrics["auc"],
        char_metrics["auc"],
        minilm_metrics["auc"],
        pm_metrics["auc"],
        svm_overall["auc"],
        xgb_overall["auc"]
    ]

    # Plot style helper
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    # =========================================================================
    # FIGURE 1: Comparaison globale F1/ROC-AUC
    # =========================================================================
    print("\n" + "="*80)
    print("FIGURE 1: SOURCE DATA (LITERAL VALUES USED)")
    print("="*80)
    for name, f1, auc in zip(models_list, f1_scores, auc_scores):
        print(f"Model: {name:<20} | F1 = {f1*100:6.2f}% | ROC-AUC = {auc*100:6.2f}%")

    x = np.arange(len(models_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, [val * 100 for val in f1_scores], width, label='F1-Score (%)', color='#E06666')
    rects2 = ax.bar(x + width/2, [val * 100 for val in auc_scores], width, label='ROC-AUC (%)', color='#6FA8DC')

    ax.set_ylabel('Performance (%)', fontsize=12, labelpad=10)
    ax.set_title('Comparaison Globale des Performances des Modèles (Validation)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'comparaison_globale.png'), dpi=300)
    plt.close()
    print("--> Saved comparaison_globale.png")

    # =========================================================================
    # FIGURE 2: Comparaison stratifiée par éditeur (F1)
    # =========================================================================
    def get_stratified_f1(preds, pub_list):
        strat_f1s = {}
        for pub in ['Hindawi', 'Spandidos', 'Others']:
            if pub == 'Others':
                idx = [i for i, p in enumerate(pub_list) if p not in ['Hindawi', 'Spandidos']]
            else:
                idx = [i for i, p in enumerate(pub_list) if p == pub]
            strat_f1s[pub] = f1_score(val_labels[idx], preds[idx])
        return strat_f1s

    trivial_strat = get_stratified_f1(preds_trivial, val_pubs)
    base_strat = get_stratified_f1(preds_base, val_pubs)
    comb_strat = get_stratified_f1(preds_comb, val_pubs)
    char_strat = get_stratified_f1(preds_char, val_pubs)
    minilm_strat = get_stratified_f1(preds_minilm, val_pubs)
    pm_strat = get_stratified_f1(preds_pm, val_pubs)

    hindawi_f1s = [
        trivial_strat["Hindawi"],
        base_strat["Hindawi"],
        comb_strat["Hindawi"],
        char_strat["Hindawi"],
        minilm_strat["Hindawi"],
        pm_strat["Hindawi"],
        svm_strat["hindawi"]["f1"],
        xgb_strat["hindawi"]["f1"]
    ]

    spandidos_f1s = [
        trivial_strat["Spandidos"],
        base_strat["Spandidos"],
        comb_strat["Spandidos"],
        char_strat["Spandidos"],
        minilm_strat["Spandidos"],
        pm_strat["Spandidos"],
        svm_strat["spandidos"]["f1"],
        xgb_strat["spandidos"]["f1"]
    ]

    others_f1s = [
        trivial_strat["Others"],
        base_strat["Others"],
        comb_strat["Others"],
        char_strat["Others"],
        minilm_strat["Others"],
        pm_strat["Others"],
        svm_strat["others"]["f1"],
        xgb_strat["others"]["f1"]
    ]

    print("\n" + "="*80)
    print("FIGURE 2: SOURCE DATA (STRATIFIED F1 BY PUBLISHER)")
    print("="*80)
    for name, h_f1, s_f1, o_f1 in zip(models_list, hindawi_f1s, spandidos_f1s, others_f1s):
        print(f"Model: {name:<20} | Hindawi = {h_f1*100:6.2f}% | Spandidos = {s_f1*100:6.2f}% | Pooled Others = {o_f1*100:6.2f}%")

    x = np.arange(len(models_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, [val * 100 for val in hindawi_f1s], width, label='Hindawi In-Pool', color='#E06666')
    ax.bar(x, [val * 100 for val in spandidos_f1s], width, label='Spandidos', color='#6FA8DC')
    ax.bar(x + width, [val * 100 for val in others_f1s], width, label='Pooled Others', color='#8FCE00')

    ax.set_ylabel('F1-Score (%)', fontsize=12, labelpad=10)
    ax.set_title('Comparaison de la Performance F1 par Éditeur (Validation)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'comparaison_stratified.png'), dpi=300)
    plt.close()
    print("--> Saved comparaison_stratified.png")

    # =========================================================================
    # FIGURE 3: Modèles fine-tunés avec mise en garde visuelle
    # =========================================================================
    # Load fine-tuned results
    with open(os.path.join(models_dir, 'pubmedbert_finetuned_results.json'), "r", encoding="utf-8") as f:
        ft_pm_results = json.load(f)
    with open(os.path.join(models_dir, 'scibert_finetuned_results.json'), "r", encoding="utf-8") as f:
        ft_sci_results = json.load(f)

    ft_models_list = models_list + ["PubMedBERT (fine-tuned)", "SciBERT (fine-tuned)"]
    ft_f1_scores = f1_scores + [ft_pm_results["overall"]["f1"], ft_sci_results["overall"]["f1"]]
    
    # We will build a stratified F1 comparison for Spandidos, Others, Hindawi
    ft_hindawi_f1s = hindawi_f1s + [ft_pm_results["stratified"]["hindawi"]["f1"], ft_sci_results["stratified"]["hindawi"]["f1"]]
    ft_spandidos_f1s = spandidos_f1s + [ft_pm_results["stratified"]["spandidos"]["f1"], ft_sci_results["stratified"]["spandidos"]["f1"]]
    ft_others_f1s = others_f1s + [ft_pm_results["stratified"]["others"]["f1"], ft_sci_results["stratified"]["others"]["f1"]]

    print("\n" + "="*80)
    print("FIGURE 3: SOURCE DATA (FINE-TUNED INCLUDED STRATIFIED F1)")
    print("="*80)
    for name, h_f1, s_f1, o_f1 in zip(ft_models_list, ft_hindawi_f1s, ft_spandidos_f1s, ft_others_f1s):
        print(f"Model: {name:<25} | Hindawi = {h_f1*100:6.2f}% | Spandidos = {s_f1*100:6.2f}% | Pooled Others = {o_f1*100:6.2f}%")

    x_ft = np.arange(len(ft_models_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Draw bars for standard models
    bars1 = ax.bar(x_ft[:-2] - width, [val * 100 for val in ft_hindawi_f1s[:-2]], width, label='Hindawi In-Pool', color='#E06666')
    bars2 = ax.bar(x_ft[:-2], [val * 100 for val in ft_spandidos_f1s[:-2]], width, label='Spandidos', color='#6FA8DC')
    bars3 = ax.bar(x_ft[:-2] + width, [val * 100 for val in ft_others_f1s[:-2]], width, label='Pooled Others', color='#8FCE00')
    
    # Draw bars for fine-tuned models (hatch pattern to distinguish them)
    bars1_ft = ax.bar(x_ft[-2:] - width, [val * 100 for val in ft_hindawi_f1s[-2:]], width, color='#EA9999', edgecolor='black', hatch='//')
    bars2_ft = ax.bar(x_ft[-2:], [val * 100 for val in ft_spandidos_f1s[-2:]], width, color='#9FC5E8', edgecolor='black', hatch='\\\\')
    bars3_ft = ax.bar(x_ft[-2:] + width, [val * 100 for val in ft_others_f1s[-2:]], width, color='#B6E59E', edgecolor='black', hatch='xx')

    # Re-apply labels to legend
    ax.legend([bars1, bars2, bars3], ['Hindawi In-Pool', 'Spandidos', 'Pooled Others'], fontsize=11, loc='upper left')

    ax.set_ylabel('F1-Score (%)', fontsize=12, labelpad=10)
    ax.set_title('Performance F1 des Modèles avec Fine-Tuning Neuronal Profond', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_ft)
    ax.set_xticklabels(ft_models_list, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 115)
    
    # Add textual annotation for Section 5.8 warning
    ax.text(x_ft[-1] - 1.5, 105, 
            "Mise en garde (Section 5.8) :\nPerformance parfaite (100% F1) sur Hindawi due à un\nsurapprentissage stylistique local (syntaxe de phrases\ntypes et d'IA), vulnérable aux dérives.", 
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF2CC", ec="#D5A6BD", lw=1.5),
            fontsize=10, color="black", fontweight="semibold")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'modeles_finetunes.png'), dpi=300)
    plt.close()
    print("--> Saved modeles_finetunes.png")

    # =========================================================================
    # FIGURE 4: Forest plot de significativité statistique
    # =========================================================================
    with open(os.path.join(models_dir, 'significance_results.json'), "r", encoding="utf-8") as f:
        sig_results = json.load(f)

    comp_models = ["TF-IDF Baseline", "Char N-grams", "PubMedBERT (frozen)"]
    means = [sig_results["bootstrap_ci"][m]["mean"] * 100 for m in comp_models]
    lowers = [sig_results["bootstrap_ci"][m]["lower"] * 100 for m in comp_models]
    uppers = [sig_results["bootstrap_ci"][m]["upper"] * 100 for m in comp_models]
    p_values = [sig_results["mcnemar"][m]["pvalue"] for m in comp_models]
    chi2s = [sig_results["mcnemar"][m]["chi2"] for m in comp_models]

    print("\n" + "="*80)
    print("FIGURE 4: SOURCE DATA (SIGNIFICANCE STATISTICS)")
    print("="*80)
    for name, mean, lo, hi, pval, chi2 in zip(comp_models, means, lowers, uppers, p_values, chi2s):
        print(f"Comparison: TF-IDF Combiné vs {name:<20} | Mean Diff = {mean:+.2f}% | 95% CI = [{lo:.2f}%, {hi:.2f}%] | McNemar p = {pval:.6f} (chi2={chi2:.2f})")

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Vertically plotted forest plot
    y_pos = np.arange(len(comp_models))
    errors_low = np.array(means) - np.array(lowers)
    errors_high = np.array(uppers) - np.array(means)
    
    ax.errorbar(means, y_pos, xerr=[errors_low, errors_high], fmt='o', color='#1f77b4', 
                markersize=8, elinewidth=2.5, capsize=8, label="Différence F1 moyenne (95% CI)")
    
    # Reference vertical line at 0 (meaning no difference)
    ax.axvline(0, color='grey', linestyle='--', lw=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp_models, fontsize=11, fontweight="bold")
    ax.set_xlabel("Différence de score F1 par rapport à TF-IDF Combiné (points %)", fontsize=12, labelpad=10)
    ax.set_title("Différences de Performance (dF1) et Significativité Statistique", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(-1, 15)
    
    # Add textual annotation for each errorbar detailing McNemar p-value
    for idx, (mean, pval) in enumerate(zip(means, p_values)):
        p_text = f"p < 0.0001" if pval < 0.0001 else f"p = {pval:.4f}"
        ax.text(mean + 0.3, idx + 0.12, f"dF1 = {mean:+.2f}%\nMcNemar : {p_text}", 
                fontsize=10, va='center', fontweight="semibold")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'significativite_forest_plot.png'), dpi=300)
    plt.close()
    print("--> Saved significativite_forest_plot.png")

if __name__ == "__main__":
    main()
