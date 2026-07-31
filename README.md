# Cancer Paper Mill Detection Dataset (CPM-11K)

CPM-11K is a verified, curated, and split dataset constructed for training and evaluating machine learning models to detect fraudulent research outputs originating from paper mills in the oncology and cancer biology domains. 

The dataset is built by matching retracted paper-mill papers (positives) from the **Retraction Watch Database (RWDB)** with clean, publication-year-and-journal-matched control articles (negatives) from MEDLINE/PubMed. It features rigorous filters, abstract recovery, and a group-aware splitting partition to prevent data leakage.

---

## Repository Structure

```text
cancer-paper-mill/
├── data/
│   ├── final/              <-- Final gold dataset and split partitions (Git ignored)
│   │   ├── cancer_pm_dataset.json      # Full stable dataset (11,106 records)
│   │   ├── cancer_pm_train.json        # Train partition (70% main pool)
│   │   ├── cancer_pm_val.json          # Validation partition (15% main pool)
│   │   ├── cancer_pm_test.json         # Test partition (15% main pool)
│   │   ├── cancer_pm_holdout.json      # 50% Hindawi Holdout Test set (OOD Generalization)
│   │   ├── journal_to_nlm.json         # Journal name to NLM abbreviation mappings
│   │   └── recent_unstable_dataset.json# Excluded 2024-2025 unstable records (retraction lag)
│   └── raw/                <-- Raw input data sources (Git ignored)
│       └── rwdb/
│           └── retraction_watch.csv    # Retraction Watch Database CSV dump
├── docs/
│   ├── DATASET_CARD.md     # Detailed dataset card, methodology, and limitations
│   ├── THESIS_CHAPTER.md   # Full thesis chapter: methodology, baseline model, and causal limits
│   └── figures/            <-- Embedded figures for the thesis
├── scripts/
│   ├── analysis/           <-- Core evaluation and experimental analysis scripts
│   │   ├── compare_normalization_results.py
│   │   ├── quantify_extraction_artifacts.py
│   │   ├── run_cross_transfer_attribution.py
│   │   ├── run_cross_transfer_audit.py
│   │   └── run_hard_negative_stress_test.py
│   ├── diagnostics/        <-- Active structural diagnostics and stratification checks
│   │   ├── check_leakage_and_score_space.py
│   │   ├── diagnose_purges.py
│   │   ├── diagnose_strata.py
│   │   └── evaluate_sous_groupe_b.py
│   └── pipeline/           <-- Active production data engineering & modeling scripts
│       ├── evaluate_all_models.py
│       ├── fetch_abstracts.py
│       ├── finetune_pubmedbert.py
│       ├── finetune_pubmedbert_normalized.py
│       ├── finetune_scibert.py
│       ├── generate_final_eval_figures.py
│       ├── generate_french_figures.py
│       ├── generate_negatives.py
│       ├── normalize_abstracts.py
│       ├── run_controls_training.py
│       ├── significance_testing.py
│       ├── split_dataset.py
│       ├── train_classical_baselines.py
│       └── validate_negatives.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Reproduction Instructions

The pipeline must be run from the root of the repository. All scripts resolve paths dynamically relative to their location.

### 0. Environment Setup
Install the required packages:
```bash
pip install -r requirements.txt
```

### 1. Match Negative Control Class
Run the MEDLINE query match engine to build the negative matching candidate class. This searches MEDLINE/PubMed by year and NLM journal abbreviation, falls back to $\pm 1$ year for depleted cells, and applies official Publication Type (`[PT]`) exclusion filters:
```bash
python scripts/pipeline/generate_negatives.py
```

### 2. Recover Abstracts and Apply Oncology Filtering
Fetch scientific abstracts from PubMed Efetch and Crossref API in rate-limited batches, strip out retraction watermarks/boilerplates, and verify oncology keyword relevance in Title + Abstract:
```bash
python scripts/pipeline/fetch_abstracts.py
```

### 3. Generate Group-Aware Splits
Construct the 50% Hindawi generalization holdout set and partition the remaining main pool into 70/15/15 splits. To prevent data leakage, records are clustered together into split-inseparable components if they share $\ge 2$ authors or have title character 3-gram Jaccard similarity $\ge 0.7$:
```bash
python scripts/pipeline/split_dataset.py
```

### 4. Run Audits and Re-generate Figures
Verify dataset statistics and re-generate the French audit figures from the final partitions on disk:
```bash
python scripts/pipeline/generate_french_figures.py
```

---

## Large Data Files Exclusions
Due to file size and licensing constraints, all raw source files (Retraction Watch CSV, Excel sheets) and finalized dataset partitions (`data/final/*.json`) are **excluded** from this repository via `.gitignore`. 

To obtain the datasets:
1.  Request access to the **Retraction Watch Database (RWDB)** directly via [Retraction Watch](https://retractionwatch.com/retraction-watch-database-user-guide/).
2.  Save the CSV export to `data/raw/rwdb/retraction_watch.csv`.
3.  Execute the reproduction pipeline to regenerate the final splits locally.

---

## Dataset Card & Limitations
For the full methodology, publisher distribution metrics, and a deep-dive analysis on class biases (e.g. purge rate asymmetry, funding index artifacts, and local publisher correlation), please read the **[DATASET_CARD.md](docs/DATASET_CARD.md)**.

---

## Attribution & Citations
*   **Retraction Watch Database:** This project relies on positive labels extracted from the Retraction Watch Database (RWDB), courtesy of the Center for Scientific Integrity. 
*   **MEDLINE/PubMed:** Control class metadata is queried and retrieved from NLM's MEDLINE/PubMed database.
