# Dataset Card: Cancer Paper Mill Detection Dataset (CPM-11K)

## Dataset Description
The Cancer Paper Mill (CPM-11K) dataset is a verified labeled corpus constructed for training and evaluating machine learning models to detect fraudulent research outputs originating from paper mills in the oncology and cancer biology domains.

*   **Total Stable Records (Main Dataset):** 11,106
*   **Positives (Paper Mill Retractions):** 2,111
*   **Negatives (Clean Matching Literature):** 8,995
*   **Dataset Ratio:** 4.26:1 (Deduplicated unique DOIs)

## Data Representation
Each record in `data/cancer_pm_dataset.json` contains:
*   `doi`: Digital Object Identifier.
*   `pmid`: PubMed identifier.
*   `title`: Paper Title.
*   `abstract`: Original un-retracted scientific abstract (retraction boilerplates stripped).
*   `journal`: Target NLM journal name.
*   `year`: Publication Year.
*   `label`: `1` (positive paper-mill case), `0` (negative clean case).
*   `abstract_source`: Source of abstract (`PubMed` or `Crossref`).

---

## Methodological Decisions & Stated Limitations

### 1. Mandatory Publisher-Stratified Evaluation
To address publisher distribution differences between positive and negative classes (Hindawi: 40.5% pos vs 23.8% neg; Spandidos: 15.6% pos vs 17.0% neg):
*   **Decision on Undersampling:** A simulation of negative undersampling to match publisher proportions was executed. To align Spandidos and other publishers perfectly with the positive mix, the negative class would have to be downsampled from 8,995 to 5,299, resulting in the loss of **3,696 clean negatives (41.1% of the negative class)** and reducing the ratio to **2.51:1**. Due to this severe loss of negative class volume and diversity, we decided **against** undersampling.
*   **Commitment to Stratified Evaluation:** To prevent style-confound cheating, **publisher-stratified precision and recall reporting is a mandatory evaluation step** for any model trained on this dataset. Precision and recall must be reported separately for the following three strata:
    1.  **Hindawi-only:** (854 positives, 2,144 negatives)
    2.  **Spandidos-only:** (330 positives, 1,532 negatives)
    3.  **Non-Hindawi/Non-Spandidos (Pooled Others):** (927 positives, 5,319 negatives)
*   **Low-Volume Strata Granularity Resolution:** Within the "Non-Hindawi/Non-Spandidos" pooled stratum, only three individual publishers have sufficient positive counts to support stable standalone precision/recall reporting: **Wiley (196 positives)**, **Verduci Editore (147 positives)**, and **Dove Press (103 positives)**. All other publishers have too few positive records to support stable estimates (e.g. *Portland Press* has 53, *RSC* has 36, *Elsevier* has 35, and *Karger* has 33). Therefore, Stratum 3 will be evaluated **pooled**, but standalone precision/recall can optionally be reported for the three high-volume sub-strata (Wiley, Verduci, Dove Press) to provide additional granularity.

### 2. Resolution of 2024-2025 (Retraction-Reporting Lag)
*   **Mechanism:** Positive counts for very recent years (2024: 6 pos; 2025: 0 pos) are structurally undercounted due to **retraction-reporting lag**. It takes between 1 to 3+ years for journal investigations, whistleblower flags, and official publisher retractions to occur and be indexed in Retraction Watch (RWDB).
*   **Decision:** All 41 records from 2024 and 2025 (6 positives, 35 negatives) have been **excluded** from the main training/evaluation dataset to prevent severe year-matched class imbalance. They are moved to a separate file: `data/recent_unstable_dataset.json` for independent testing of recent literature.

### 3. Purge Rate Asymmetry & Selection Bias
*   We observed a significant asymmetry in purge rates: **17.0% for positives (434 / 2,551)** vs **2.6% for negatives (243 / 9,273)**.
*   **Purge Breakdown:**
    *   **Positives Purged (434):** 322 failed abstract recovery (empty/boilerplate), 112 failed oncology relevance.
    *   **Negatives Purged (243):** 148 failed abstract recovery (empty), 95 failed oncology relevance.
*   **Publisher Specificity:** The abstract recovery failure is heavily concentrated in publishers who systematically delete or replace retracted abstracts with standard retraction boilerplate (e.g., *Biomedicine & Pharmacotherapy* (Elsevier) had a **76.7% positive purge rate**; *Inorganic and Nano-Metal Chemistry* (T&F) had a **100% positive purge rate**). Spandidos and Hindawi frequently preserve original abstracts with retraction watermarks, resulting in low purge rates.
*   **Selection Bias Risk:** This differential purge rate introduces a publisher selection bias into the positive training class. Post-purge, Hindawi's share in positives rose from 31.9% to 40.5%, while Elsevier's fell from 6.3% to 1.7%. The model may learn style features correlated with publishers who preserve abstracts rather than general paper-mill indicators.

### 4. Source Label Noise (RWDB Data-Quality Finding)
*   Out of 2,223 positives where a valid abstract was recovered, **112 papers (5.0%) failed oncology relevance verification**.
*   These papers were tagged with a "cancer" subject in Retraction Watch (RWDB) but did not contain any oncology keywords in their title or abstract (e.g. general medicine, toxicity, or chemistry papers). This represents a minor labeling noise ceiling in the source database.

### 5. Funding Support Gap: Indexing Artifact vs Textual Confound
*   The positive class had a lower rate of the PubMed Publication Type `Research Support, Non-U.S. Gov't` (10.6%) compared to negatives (19.5%).
*   **Textual Audit:** A keyword audit on abstract text showed that the occurrence of funding terms (*grant, fund, support, award*) is virtually identical between classes (**4.1% in positives** vs **4.2% in negatives**).
*   **Conclusion:** The funding gap is an **indexing artifact** rather than a textual confound. Clean negatives are more likely to have successfully passed indexing validation or formal metadata funding declarations in MEDLINE, whereas retracted paper mills are not tagged with this publication type despite similar abstract-text funding claims.

---

## Train / Validation / Test Partitioning

To ensure rigorous evaluation and prevent data leakage, the dataset is partitioned using the following constraints:

### 1. 50% Hindawi Generalization Holdout
Prior to partitioning, Hindawi-published records (2,998 total) were split near-even ($\pm 1$) stratified by year.
*   **Hindawi-Holdout:** **429 positives, 1,075 negatives** (held out completely from the main pool to serve as a generalization test set).
*   **Hindawi-In-Pool:** **425 positives, 1,069 negatives** (retained in the train/val/test pool).

### 2. Main Pool Split (70/15/15)
The remaining non-holdout main pool (9,602 records) is split 70% Train, 15% Validation, and 15% Test.
*   **Train:** **1,178 positives, 5,544 negatives** (Total: 6,722)
*   **Val:** **252 positives, 1,188 negatives** (Total: 1,440)
*   **Test:** **252 positives, 1,188 negatives** (Total: 1,440)

*Justification:* A 70/15/15 split provides a large validation and test pool (1,440 records each) containing 252 positives per split. This volume is well above the statistical stability threshold (30–50 positive cases) required to compute stable precision/recall curves.

### 3. Leakage Protection (Group-Aware Splitting)
To prevent template and author leakage, records were grouped into disjoint clusters prior to splitting using:
1.  **Author matching:** Positive records sharing **at least 2 authors** were merged into the same cluster. Single author matching was explicitly avoided to prevent connected-component percolation (which previously chained 979 records into a single giant component due to common Chinese surnames).
2.  **Title Jaccard similarity:** Any two records in the same journal with title character 3-gram Jaccard similarity $\ge 0.7$ were merged.
*   *Validation:* **0 grouped/duplicate-signature records cross split boundaries.**

### 4. Low-Volume Warning (Grouping Tradeoff)
Because group-aware clustering prevents split-crossing, several low-volume publishers have their positive records concentrated in a single split to avoid leakage:
*   *Elsevier (35 positives):* Train gets 34, Val gets 0, Test gets 1.
*   *Portland Press (53 positives):* Train gets 52, Val gets 0, Test gets 1.
*   *Taylor & Francis (34 positives):* Train gets 28, Val gets 2, Test gets 4.
This concentration is a mathematically necessary consequence of preventing author/template leakage when total positive counts are low. Standalone precision/recall reporting for these low-volume publishers in Val/Test is statistically unstable. They **must** be evaluated as part of the pooled "Non-Hindawi/Non-Spandidos" stratum.

---

## Dataset Distributions (Post-Purge Stable)

### Publisher Distribution
The post-purge stable distributions across splits. The "Other" row accounts for 32 distinct publishers with low-volume counts:

| Publisher | Train (Pos/Neg) | Val (Pos/Neg) | Test (Pos/Neg) | Holdout (Pos/Neg) |
| :--- | :---: | :---: | :---: | :---: |
| **Hindawi** | 157 / 356 | 134 / 356 | 134 / 357 | 429 / 1,075 |
| **Spandidos** | 290 / 1,444 | 22 / 45 | 18 / 43 | 0 / 0 |
| **Wiley** | 132 / 598 | 33 / 275 | 31 / 271 | 0 / 0 |
| **Verduci Editore** | 103 / 612 | 21 / 48 | 23 / 49 | 0 / 0 |
| **Taylor & Francis (Dove)** | 84 / 544 | 9 / 8 | 10 / 8 | 0 / 0 |
| **Portland Press** | 52 / 171 | 0 / 61 | 1 / 63 | 0 / 0 |
| **RSC** | 16 / 71 | 11 / 69 | 9 / 70 | 0 / 0 |
| **Elsevier** | 34 / 402 | 0 / 74 | 1 / 70 | 0 / 0 |
| **Taylor & Francis** | 28 / 408 | 2 / 28 | 4 / 30 | 0 / 0 |
| **Other (32 low-volume)** | 282 / 938 | 20 / 224 | 21 / 227 | 0 / 0 |
| **Total** | **1,178 / 5,544** | **252 / 1,188** | **252 / 1,188** | **429 / 1,075** |

### Year Distribution

| Year | Train (Pos/Neg) | Val (Pos/Neg) | Test (Pos/Neg) | Holdout (Pos/Neg) |
| :--- | :---: | :---: | :---: | :---: |
| 2011 | 1 / 1 | 0 / 1 | 0 / 1 | 0 / 0 |
| 2012 | 2 / 2 | 0 / 1 | 0 / 2 | 0 / 0 |
| 2013 | 5 / 13 | 0 / 12 | 1 / 8 | 0 / 0 |
| 2014 | 24 / 40 | 2 / 25 | 2 / 27 | 1 / 3 |
| 2015 | 71 / 192 | 12 / 50 | 4 / 55 | 0 / 0 |
| 2016 | 62 / 333 | 2 / 39 | 9 / 35 | 0 / 0 |
| 2017 | 171 / 490 | 16 / 86 | 14 / 80 | 0 / 0 |
| 2018 | 184 / 1,173 | 24 / 191 | 24 / 195 | 0 / 1 |
| 2019 | 249 / 1,243 | 44 / 270 | 40 / 275 | 2 / 6 |
| 2020 | 181 / 1,200 | 21 / 148 | 27 / 153 | 13 / 53 |
| 2021 | 94 / 535 | 32 / 152 | 31 / 145 | 97 / 323 |
| 2022 | 121 / 271 | 94 / 191 | 94 / 188 | 298 / 552 |
| 2023 | 13 / 51 | 5 / 22 | 6 / 24 | 18 / 65 |
| **Total** | **1,178 / 5,544** | **252 / 1,188** | **252 / 1,188** | **429 / 1,075** |
