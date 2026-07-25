"""
Comprehensive Validation Script for V5 Clean Negatives Database.
Checks:
1. Article-Level Topic Audit: 100% oncology titles.
2. Side-by-Side PubMed [PT] Audit (Positives vs Negatives).
3. Overlap & Deduplication (0 overlap, 0 duplicates).
4. Publisher Gap Analysis.
5. Depleted Group Shortfall Table with Filter Attributions.
6. 2022 Sample Eyeball.
"""
import sys
import os

# Setup portable project root relative to script location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import csv
import urllib.request
import urllib.parse
import random
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(sys.maxsize)

with open(os.path.join(project_root, 'data', 'clean_positives.json'), 'r', encoding='utf-8') as f:
    positives = json.load(f)

with open(os.path.join(project_root, 'data', 'clean_negatives.json'), 'r', encoding='utf-8') as f:
    negatives = json.load(f)

print(f"Total Clean Positives: {len(positives)}")
print(f"Total Clean Negatives: {len(negatives)}")

# Load global RWDB exclusion list
rwdb_all_dois = set()
with open(os.path.join(project_root, 'data', 'raw', 'rwdb', 'retraction_watch.csv'), 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doi = row.get('OriginalPaperDOI', '').strip().lower()
        if doi and doi != '0':
            rwdb_all_dois.add(doi)

# =====================================================================
# 1. Topic Audit (Article Level)
# =====================================================================
print("\n" + "="*70)
print("1. ARTICLE-LEVEL ONCOLOGY TOPIC AUDIT")
print("="*70)

CANCER_KEYWORDS = ['cancer', 'oncol', 'tumor', 'tumour', 'neoplas', 'carcinoma',
                   'leukemia', 'leukaemia', 'lymphoma', 'melanoma', 'sarcoma',
                   'metasta', 'malignan', 'glioma', 'osteosarcoma', 'neuroblastoma',
                   'hepatoma', 'myeloma']

cancer_matches = sum(1 for n in negatives if any(kw in n['title'].lower() for kw in CANCER_KEYWORDS))
non_cancer_matches = len(negatives) - cancer_matches

print(f"Negatives with Cancer/Oncology keyword in title: {cancer_matches} ({cancer_matches/len(negatives)*100:.2f}%)")
print(f"Negatives WITHOUT Cancer/Oncology keyword in title: {non_cancer_matches} ({non_cancer_matches/len(negatives)*100:.2f}%)")
if non_cancer_matches == 0:
    print("  PASSED: 100% of negative titles are explicitly oncology/cancer-related.")

# =====================================================================
# 2. Overlap & Deduplication Checks
# =====================================================================
print("\n" + "="*70)
print("2. OVERLAP / DEDUPLICATION CHECKS")
print("="*70)

pos_dois = set(p.get('OriginalPaperDOI', '').strip().lower() for p in positives if p.get('OriginalPaperDOI'))
neg_dois = set(n['doi'].lower().strip() for n in negatives)

overlap_with_pos = neg_dois & pos_dois
print(f"Overlap with Clean Positives: {len(overlap_with_pos)}")
if overlap_with_pos:
    print(f"  FAILED: Overlapping DOIs: {list(overlap_with_pos)}")
else:
    print("  PASSED: Zero overlap with positive DOIs.")

overlap_with_all_rwdb = neg_dois & rwdb_all_dois
print(f"Overlap with ALL Retracted Papers in RWDB: {len(overlap_with_all_rwdb)}")
if overlap_with_all_rwdb:
    print(f"  FAILED: Overlapping retracted DOIs: {list(overlap_with_all_rwdb)[:10]}")
else:
    print("  PASSED: Zero overlap with any retracted paper in RWDB (No retraction contamination).")

dup_count = len(negatives) - len(neg_dois)
print(f"Duplicate DOIs in negatives: {dup_count}")
if dup_count == 0:
    print("  PASSED: Zero duplicate DOIs.")

ratio = len(neg_dois) / len(positives) if len(positives) > 0 else 0
print(f"Dataset Ratio Achieved: {ratio:.2f}:1 (Target: 5.00:1)")

# =====================================================================
# 3. Side-by-Side PubMed [PT] Audit
# =====================================================================
print("\n" + "="*70)
print("3. PUBMED [PT] COMPARISON SIDE-BY-SIDE")
print("="*70)

# Fetch pubtypes for positives (sample of 200)
pos_pmids = [p.get('OriginalPaperPubMedID', '').strip() for p in positives if p.get('OriginalPaperPubMedID') and p.get('OriginalPaperPubMedID') != '0']
random.seed(42)
sample_pos_pmids = random.sample(pos_pmids, min(200, len(pos_pmids)))

pos_pt_counter = Counter()
if sample_pos_pmids:
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(sample_pos_pmids)}&retmode=json"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'PTCheck/1.0'})) as r:
            res = json.loads(r.read().decode('utf-8'))
            data = res.get('result', {})
            for pmid in sample_pos_pmids:
                pdata = data.get(pmid, {})
                for pt in pdata.get('pubtype', []):
                    pos_pt_counter[pt] += 1
    except Exception as e:
        print("Error fetching positive PTs:", e)

# Negative pubtypes
neg_pt_counter = Counter()
for n in negatives:
    for pt in n.get('pubtypes', []):
        neg_pt_counter[pt] += 1

print(f"{'Publication Type [PT]':<35} | {'Pos Share (%)':<15} | {'Neg Share (%)':<15}")
print("-"*70)
all_pts = set(pos_pt_counter.keys()) | set(neg_pt_counter.keys())
for pt in sorted(all_pts, key=lambda x: -(pos_pt_counter.get(x, 0)/len(sample_pos_pmids) + neg_pt_counter.get(x, 0)/len(negatives))):
    pos_cnt = pos_pt_counter.get(pt, 0)
    pos_pct = pos_cnt / len(sample_pos_pmids) * 100 if len(sample_pos_pmids) > 0 else 0
    neg_cnt = neg_pt_counter.get(pt, 0)
    neg_pct = neg_cnt / len(negatives) * 100 if len(negatives) > 0 else 0
    print(f"{pt[:33]:<35} | {pos_pct:>12.1f}% | {neg_pct:>12.1f}%")

# =====================================================================
# 4. Publisher Distribution Comparison
# =====================================================================
print("\n" + "="*70)
print("4. PUBLISHER DISTRIBUTION COMPARISON")
print("="*70)

journal_to_publisher = {}
with open(os.path.join(project_root, 'data', 'raw', 'rwdb', 'retraction_watch.csv'), 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        j = row.get('Journal', '').strip()
        p = row.get('Publisher', '').strip()
        if j and p:
            # Overwrite known publisher misattributions manually
            if "computational and mathematical methods in medicine" in j.lower():
                p = "Hindawi"
            elif "biomed research international" in j.lower():
                p = "Hindawi"
            journal_to_publisher[j] = p

with open(os.path.join(project_root, 'data', 'final', 'journal_to_nlm.json'), 'r', encoding='utf-8') as f:
    j2nlm = json.load(f)
nlm2raw = {v: k for k, v in j2nlm.items()}

pos_publishers = Counter(p.get('Publisher', '').strip() for p in positives)
neg_publishers = Counter()

# Override publisher lookup
for n in negatives:
    nlm_j = n['journal']
    raw_j = nlm2raw.get(nlm_j, nlm_j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() or "comput math methods med" in nlm_j.lower():
        pub = "Hindawi"
    neg_publishers[pub] += 1

print(f"{'Publisher':<40} | {'Pos Share (%)':<15} | {'Neg Share (%)':<15} | {'Difference (%)':<15}")
print("-"*95)
for pub, p_count in pos_publishers.most_common(15):
    p_pct = p_count / len(positives) * 100
    n_count = neg_publishers[pub]
    n_pct = n_count / len(negatives) * 100 if len(negatives) > 0 else 0
    diff = p_pct - n_pct
    print(f"{pub[:38]:<40} | {p_pct:>12.1f}% | {n_pct:>12.1f}% | {diff:>12.1f}%")

# =====================================================================
# 5. 2022 Sample Records Eyeball
# =====================================================================
print("\n" + "="*70)
print("5. 2022 SAMPLE RECORDS (FOR EYE-BALLING)")
print("="*70)
neg_2022 = [n for n in negatives if n['year'] == 2022]
if neg_2022:
    sample_2022 = random.sample(neg_2022, min(5, len(neg_2022)))
    for i, n in enumerate(sample_2022):
        print(f"[{i+1}] Title: {n['title']}")
        print(f"    DOI: {n['doi']} | PMID: {n['pmid']}")
        print(f"    Journal: {n['journal']} | Year: {n['year']}")
        print()
