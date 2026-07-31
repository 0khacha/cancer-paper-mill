import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def main():
    # 1. Load pre-normalization artifact stats
    artifacts_path = "models/pre_normalization_artifacts.json"
    if not os.path.exists(artifacts_path):
        print(f"Error: {artifacts_path} not found. Run quantify_extraction_artifacts.py first.")
        return
        
    with open(artifacts_path, "r", encoding="utf-8") as f:
        artifact_stats = json.load(f)
        
    # 2. Load all model evaluation results
    eval_results_path = "models/all_evaluation_results.json"
    if not os.path.exists(eval_results_path):
        print(f"Error: {eval_results_path} not found. Run training and evaluate_all_models.py first.")
        return
        
    with open(eval_results_path, "r", encoding="utf-8") as f:
        eval_results = json.load(f)
        
    # 3. Print Deliverable 1: Table of Artifact Rates by Class x Source Pipeline (Pre-normalization)
    print("\n" + "="*80)
    print("DELIVERABLE 1: PRE-NORMALIZATION ARTIFACT RATES TABLE")
    print("="*80)
    print(f"| {'Class':<10} | {'Source':<10} | {'N':<6} | {'Glued Hdr':<10} | {'Stray Punc':<10} | {'Whitespace':<10} | {'Unicode/Ent':<11} |")
    print("|" + "-"*12 + "|" + "-"*12 + "|" + "-"*8 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*13 + "|")
    
    cs_data = artifact_stats["class_source"]
    for key, counts in sorted(cs_data.items()):
        lbl, src = key.split("_")
        lbl_str = "Positive" if lbl == "1" else "Negative"
        n = counts["N"]
        print(f"| {lbl_str:<10} | {src:<10} | {n:<6} | {counts['glued_header']/n*100:10.2f}% | {counts['stray_punctuation']/n*100:10.2f}% | {counts['whitespace_irregularity']/n*100:10.2f}% | {counts['unicode_entity']/n*100:11.2f}% |")

    # Also print publisher table for completeness
    print("\nPre-normalization Artifact Rates by Class x Publisher:")
    print(f"| {'Class':<10} | {'Publisher':<13} | {'N':<6} | {'Glued Hdr':<10} | {'Stray Punc':<10} | {'Whitespace':<10} | {'Unicode/Ent':<11} |")
    print("|" + "-"*12 + "|" + "-"*15 + "|" + "-"*8 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*13 + "|")
    
    pub_data = artifact_stats["class_publisher"]
    for key, counts in sorted(pub_data.items()):
        lbl, pub = key.split("_", 1)
        lbl_str = "Positive" if lbl == "1" else "Negative"
        n = counts["N"]
        print(f"| {lbl_str:<10} | {pub:<13} | {n:<6} | {counts['glued_header']/n*100:10.2f}% | {counts['stray_punctuation']/n*100:10.2f}% | {counts['whitespace_irregularity']/n*100:10.2f}% | {counts['unicode_entity']/n*100:11.2f}% |")

    # 4. Print Deliverable 2: Table of AUC/F1 Before vs After Normalization
    print("\n" + "="*80)
    print("DELIVERABLE 2: AUC/F1 BEFORE VS AFTER NORMALIZATION")
    print("="*80)
    print(f"| {'Evaluation Set / Control':<45} | {'Before (Pre-Norm) AUC / F1':<28} | {'After (Post-Norm) AUC / F1':<28} | {'Delta AUC / F1':<18} |")
    print("|" + "-"*47 + "|" + "-"*30 + "|" + "-"*30 + "|" + "-"*20 + "|")
    
    # Define sets to show
    sets_to_show = [
        # (Row Label, Pre-Norm Model, Pre-Norm Set Key, Post-Norm Model, Post-Norm Set Key)
        ("Validation Set (Pooled)", "Pooled (Pre-Norm)", "val", "Pooled (Post-Norm)", "val"),
        ("Test Set (Pooled)", "Pooled (Pre-Norm)", "test", "Pooled (Post-Norm)", "test"),
        ("Hindawi Holdout (OOD)", "Pooled (Pre-Norm)", "holdout", "Pooled (Post-Norm)", "holdout"),
        ("Hindawi-Only Control", "Hindawi-Only (Pre-Norm)", "holdout", "Hindawi-Only (Post-Norm)", "holdout"),
        ("Cross-Transfer: Non-Hindawi -> Hindawi Holdout", "Non-Hindawi Trained (Pre-Norm)", "holdout", "Non-Hindawi Trained (Post-Norm)", "holdout"),
        ("Cross-Transfer: Hindawi -> Non-Hindawi Test", "Hindawi Trained -> Non-Hindawi Test (Pre-Norm)", "test_non_hindawi", "Hindawi Trained -> Non-Hindawi Test (Post-Norm)", "test_non_hindawi"),
    ]
    
    for row_name, pre_model, pre_key, post_model, post_key in sets_to_show:
        pre_metrics = eval_results.get(pre_model, {}).get(pre_key, {"auc": 0.0, "f1": 0.0})
        post_metrics = eval_results.get(post_model, {}).get(post_key, {"auc": 0.0, "f1": 0.0})
        
        pre_auc, pre_f1 = pre_metrics["auc"], pre_metrics["f1"]
        post_auc, post_f1 = post_metrics["auc"], post_metrics["f1"]
        
        delta_auc = post_auc - pre_auc
        delta_f1 = post_f1 - pre_f1
        
        print(f"| {row_name:<45} | {pre_auc:11.2%} / {pre_f1:11.2%} | {post_auc:11.2%} / {post_f1:11.2%} | {delta_auc:+10.2%} / {delta_f1:+9.2%} |")

    # 5. Print Deliverable 3: Three or Four Representative Before/After Examples per Artifact Type
    print("\n" + "="*80)
    print("DELIVERABLE 3: REPRESENTATIVE BEFORE/AFTER EXAMPLES FOR SANITY CHECK")
    print("="*80)
    
    examples = artifact_stats.get("examples", {})
    
    for art_type, ex_list in examples.items():
        name_map = {
            "glued_header": "Glued Section Headers",
            "stray_punctuation": "Stray Punctuation Preceding/Following Headers",
            "whitespace_irregularity": "Whitespace Irregularities",
            "unicode_entity": "Unicode and Entity Anomalies"
        }
        print(f"\n--- Category: {name_map.get(art_type, art_type)} ---")
        if not ex_list:
            print("  No representative examples found.")
            continue
        for idx, (before, after) in enumerate(ex_list[:3], 1):
            print(f"Example {idx}:")
            # Truncate strings around differences to keep output readable if they are long
            print(f"  [BEFORE]: {repr(before[:250])}...")
            print(f"  [AFTER] : {repr(after[:250])}...")
            print()

    # 6. Print Deliverable 4: Conclusion Paragraph Stating Hypothesis Supported
    print("\n" + "="*80)
    print("DELIVERABLE 4: EVALUATION AND CAUSAL HYPOTHESIS CONCLUSION")
    print("="*80)
    
    # Retrieve main results for conclusion calculation
    pre_holdout = eval_results.get("Pooled (Pre-Norm)", {}).get("holdout", {"auc": 0.0})["auc"]
    post_holdout = eval_results.get("Pooled (Post-Norm)", {}).get("holdout", {"auc": 0.0})["auc"]
    
    # Determine which hypothesis is supported
    if post_holdout < 0.70:
        hypothesis = "PUBLISHER-SPECIFIC FORMATTING ANOMALY DEPENDENCY (COLLAPSE TO BASELINE)"
        evidence = (
            f"The results strongly support the hypothesis of publisher-specific XML formatting dependency. "
            f"After removing formatting artifacts via the normalization pipeline, the external Hindawi holdout AUC collapsed "
            f"from {pre_holdout:.2%} to {post_holdout:.2%}, approaching the classical TF-IDF baseline (~63-67%). "
            f"This confirms that the transformer model's near-perfect accuracy was primarily driven by publisher-specific "
            f"formatting artifacts correlated with label, rather than genuine paper-mill writing style semantics."
        )
    elif post_holdout > 0.95:
        hypothesis = "GENUINE STYLE-OVERFITTING (ROBUST TO NORMALIZATION)"
        evidence = (
            f"The results strongly support the hypothesis of genuine style-overfitting (robust to formatting). "
            f"After removing formatting artifacts via the normalization pipeline, the external Hindawi holdout AUC remained "
            f"extremely high at {post_holdout:.2%} (compared to {pre_holdout:.2%} pre-normalization). "
            f"This provides strong empirical evidence that the model is relying on genuine stylistic/linguistic signatures of "
            f"paper-mill writing style, rather than extraction or rendering formatting artifacts. Section 7.2 of the report remains defensible."
        )
    else:
        hypothesis = "MIXED SIGNAL (PARTIAL FORMATTING DEPENDENCY)"
        evidence = (
            f"The results support a mixed hypothesis of partial formatting dependency and partial style-overfitting. "
            f"After removing formatting artifacts via the normalization pipeline, the external Hindawi holdout AUC dropped "
            f"from {pre_holdout:.2%} to {post_holdout:.2%}. This indicates that while the model was partially leveraging "
            f"publisher-specific XML formatting anomalies, it also captured genuine stylistic features of the paper-mill text "
            f"that allowed it to maintain performance above the classical TF-IDF baseline."
        )
        
    print(f"Supported Hypothesis: {hypothesis}\n")
    print(evidence)
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
