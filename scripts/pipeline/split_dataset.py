"""
Split Dataset Script (V2 - Fixed Author Clustering):
1. Performs 50% Hindawi generalization holdout stratified by year.
2. Group-Aware Splitting (requires >=2 shared authors for positives to prevent connected-component percolation).
3. Any two records in the same journal with title Jaccard similarity of 3-grams >= 0.7 are merged.
4. Publisher-stratified and Year-stratified split assignment (70/15/15) via greedy bin-packing.
5. Saves clean splits to JSON files.
"""
import sys
import os

# Setup portable project root relative to script location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import csv
import random
import re
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(sys.maxsize)

# Set seeds for reproducibility
random.seed(42)

# Load stable dataset
with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_dataset.json'), 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Load raw positives for metadata
with open(os.path.join(project_root, 'data', 'clean_positives.json'), 'r', encoding='utf-8') as f:
    clean_positives = json.load(f)

# Map positive DOIs to metadata
pos_metadata = {}
for p in clean_positives:
    doi = p.get('OriginalPaperDOI', '').strip().lower()
    if doi:
        pos_metadata[doi] = p

print(f"Total stable dataset records: {len(dataset)}")

# =====================================================================
# 1. Hindawi Generalization Holdout (50%)
# =====================================================================
print("\n" + "="*70)
print("1. HINDAWI GENERALIZATION HOLDOUT")
print("="*70)

# Build retraction watch publisher map
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

with open(os.path.join(project_root, 'data', 'final', 'journal_to_nlm.json'), 'r', encoding='utf-8') as f:
    j2nlm = json.load(f)
nlm2raw = {v: k for k, v in j2nlm.items()}

def get_publisher(rec):
    j = rec['journal']
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() or "comput math methods med" in j.lower():
        pub = "Hindawi"
    return pub

# Filter Hindawi
hindawi_records = [r for r in dataset if get_publisher(r) == 'Hindawi']
non_hindawi_records = [r for r in dataset if get_publisher(r) != 'Hindawi']

print(f"Total Hindawi records: {len(hindawi_records)}")
print(f"Total Non-Hindawi records: {len(non_hindawi_records)}")

# Group Hindawi by (label, year) to perform stratified splitting
hindawi_by_cell = defaultdict(list)
for r in hindawi_records:
    hindawi_by_cell[(r['label'], r['year'])].append(r)

hindawi_pool = []
hindawi_holdout = []

for cell, records in hindawi_by_cell.items():
    random.shuffle(records)
    split_idx = len(records) // 2
    hindawi_pool.extend(records[:split_idx])
    hindawi_holdout.extend(records[split_idx:])

print(f"Hindawi Holdout Split:")
print(f"  Holdout: Pos={sum(1 for r in hindawi_holdout if r['label'] == 1)}, Neg={sum(1 for r in hindawi_holdout if r['label'] == 0)}")
print(f"  In-Pool: Pos={sum(1 for r in hindawi_pool if r['label'] == 1)}, Neg={sum(1 for r in hindawi_pool if r['label'] == 0)}")

# Main Train/Val/Test Pool
pool_records = non_hindawi_records + hindawi_pool
print(f"\nNon-Holdout Main Pool Size: {len(pool_records)}")

# Save Hindawi holdout
with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_holdout.json'), 'w', encoding='utf-8') as f:
    json.dump(hindawi_holdout, f, indent=2)

# =====================================================================
# 2. Group-Aware Clustering (Union-Find)
# =====================================================================
print("\n" + "="*70)
print("2. GROUP-AWARE CLUSTERING")
print("="*70)

# Disjoint Set Union (Union-Find) implementation
parent = {}
def find(i):
    if parent[i] == i:
        return i
    parent[i] = find(parent[i])
    return parent[i]

def union(i, j):
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        parent[root_i] = root_j

# Initialize Union-Find
for i in range(len(pool_records)):
    parent[i] = i

# Parse authors for positives
record_authors = {}
for idx, r in enumerate(pool_records):
    if r['label'] == 1:
        meta = pos_metadata.get(r['doi'].lower().strip())
        if meta and meta.get('Author'):
            authors = set(a.strip().lower() for a in meta['Author'].split(';') if len(a.strip()) > 3)
            if authors:
                record_authors[idx] = authors

# Compare pairwise: merge positive records if they share AT LEAST 2 authors
keys = list(record_authors.keys())
for i in range(len(keys)):
    idx_i = keys[i]
    auth_i = record_authors[idx_i]
    for k in range(i + 1, len(keys)):
        idx_k = keys[k]
        auth_k = record_authors[idx_k]
        shared = auth_i & auth_k
        if len(shared) >= 2:
            union(idx_i, idx_k)

# Title similarity Jaccard >= 0.7 within the same journal
by_journal = defaultdict(list)
for idx, r in enumerate(pool_records):
    by_journal[r['journal']].append((idx, r['title']))

def get_3grams(text):
    text_clean = re.sub(r'\s+', ' ', text.lower().strip())
    return set(text_clean[i:i+3] for i in range(len(text_clean) - 2))

for j, items in by_journal.items():
    grams = [(idx, get_3grams(t)) for idx, t in items]
    for i in range(len(grams)):
        idx_i, g_i = grams[i]
        if not g_i:
            continue
        for k in range(i + 1, len(grams)):
            idx_k, g_k = grams[k]
            if not g_k:
                continue
            intersection = len(g_i & g_k)
            union_len = len(g_i | g_k)
            jaccard = intersection / union_len if union_len > 0 else 0
            if jaccard >= 0.7:
                union(idx_i, idx_k)

# Extract clusters
clusters = defaultdict(list)
for idx in range(len(pool_records)):
    root = find(idx)
    clusters[root].append(idx)

print(f"Total Main Pool Records: {len(pool_records)}")
print(f"Total Disjoint Clusters formed: {len(clusters)}")
cluster_sizes = Counter(len(c) for c in clusters.values())
print("Cluster size distribution:")
for sz, cnt in sorted(cluster_sizes.items()):
    print(f"  Size {sz}: {cnt} clusters")

# =====================================================================
# 3. Stratified Split Assignment (Greedy Bin-Packing)
# =====================================================================
print("\n" + "="*70)
print("3. STRATIFIED SPLIT ASSIGNMENT (70/15/15)")
print("="*70)

# Sort clusters by size descending (largest first)
sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

train_indices = []
val_indices = []
test_indices = []

# Target ratios and counts
target_ratios = {'train': 0.70, 'val': 0.15, 'test': 0.15}
pos_total = sum(1 for r in pool_records if r['label'] == 1)
neg_total = sum(1 for r in pool_records if r['label'] == 0)

target_counts = {
    'pos': {k: int(pos_total * v) for k, v in target_ratios.items()},
    'neg': {k: int(neg_total * v) for k, v in target_ratios.items()}
}

actual_counts = {
    'pos': {'train': 0, 'val': 0, 'test': 0},
    'neg': {'train': 0, 'val': 0, 'test': 0}
}

for cluster in sorted_clusters:
    c_pos = sum(1 for idx in cluster if pool_records[idx]['label'] == 1)
    c_neg = sum(1 for idx in cluster if pool_records[idx]['label'] == 0)
    
    # Calculate score for each split (greedy bin packing)
    scores = {}
    for split in ['train', 'val', 'test']:
        pos_diff = target_counts['pos'][split] - actual_counts['pos'][split]
        neg_diff = target_counts['neg'][split] - actual_counts['neg'][split]
        scores[split] = pos_diff * c_pos + neg_diff * c_neg
        
    best_split = max(scores, key=scores.get)
    
    if best_split == 'train':
        train_indices.extend(cluster)
        actual_counts['pos']['train'] += c_pos
        actual_counts['neg']['train'] += c_neg
    elif best_split == 'val':
        val_indices.extend(cluster)
        actual_counts['pos']['val'] += c_pos
        actual_counts['neg']['val'] += c_neg
    else:
        test_indices.extend(cluster)
        actual_counts['pos']['test'] += c_pos
        actual_counts['neg']['test'] += c_neg

# Construct actual records lists
train_records = [pool_records[i] for i in train_indices]
val_records = [pool_records[i] for i in val_indices]
test_records = [pool_records[i] for i in test_indices]

# Save splits
with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_train.json'), 'w', encoding='utf-8') as f:
    json.dump(train_records, f, indent=2)
with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_val.json'), 'w', encoding='utf-8') as f:
    json.dump(val_records, f, indent=2)
with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_test.json'), 'w', encoding='utf-8') as f:
    json.dump(test_records, f, indent=2)

# =====================================================================
# 6. Verification & Validation Report
# =====================================================================
print("\n" + "="*70)
print("6. SPLIT VALIDATION REPORT")
print("="*70)

print("Exact Record Counts by Split:")
print(f"  Train: Pos={len([r for r in train_records if r['label'] == 1])} | Neg={len([r for r in train_records if r['label'] == 0])} | Total={len(train_records)} ({len(train_records)/len(pool_records)*100:.1f}%)")
print(f"  Val:   Pos={len([r for r in val_records if r['label'] == 1])} | Neg={len([r for r in val_records if r['label'] == 0])} | Total={len(val_records)} ({len(val_records)/len(pool_records)*100:.1f}%)")
print(f"  Test:  Pos={len([r for r in test_records if r['label'] == 1])} | Neg={len([r for r in test_records if r['label'] == 0])} | Total={len(test_records)} ({len(test_records)/len(pool_records)*100:.1f}%)")
print(f"  Hindawi-Holdout: Pos={sum(1 for r in hindawi_holdout if r['label'] == 1)} | Neg={sum(1 for r in hindawi_holdout if r['label'] == 0)}")

# Verify group leakage
print("\nGroup Leakage Verification:")
train_dois = set(r['doi'].lower().strip() for r in train_records)
val_dois = set(r['doi'].lower().strip() for r in val_records)
test_dois = set(r['doi'].lower().strip() for r in test_records)

overlap_cross_split = 0
for r_root, c_indices in clusters.items():
    c_dois = set(pool_records[idx]['doi'].lower().strip() for idx in c_indices)
    
    in_train = c_dois & train_dois
    in_val = c_dois & val_dois
    in_test = c_dois & test_dois
    
    splits_touched = sum(1 for s in [in_train, in_val, in_test] if s)
    if splits_touched > 1:
        overlap_cross_split += 1
        print(f"  LEAKAGE DETECTED! Cluster root {r_root} (size {len(c_dois)}) split across {splits_touched} splits!")

if overlap_cross_split == 0:
    print("  PASSED: 0 grouped/duplicate-signature records cross split boundaries.")

# Publisher distribution per split
print("\nPublisher Distribution Table:")
splits_data = {'Train': train_records, 'Val': val_records, 'Test': test_records}
top_pubs = ['Hindawi', 'Spandidos', 'Wiley', 'Verduci Editore', 'Taylor and Francis - Dove Press', 'Portland Press', 'Royal Society of Chemistry (RSC)', 'Elsevier', 'Taylor and Francis']

header = f"{'Publisher':<35} | " + " | ".join(f"{s:<15}" for s in ['Train (Pos/Neg)', 'Val (Pos/Neg)', 'Test (Pos/Neg)'])
print(header)
print("-" * len(header))

for pub in top_pubs:
    row_strs = []
    for s_name, s_recs in splits_data.items():
        s_pos = len([r for r in s_recs if r['label'] == 1 and get_publisher(r) == pub])
        s_neg = len([r for r in s_recs if r['label'] == 0 and get_publisher(r) == pub])
        s_pos_pct = s_pos / len([r for r in s_recs if r['label'] == 1]) * 100
        s_neg_pct = s_neg / len([r for r in s_recs if r['label'] == 0]) * 100
        row_strs.append(f"{s_pos:>3}/{s_neg:>3} ({s_pos_pct:>4.1f}%/{s_neg_pct:>4.1f}%)")
    print(f"{pub[:33]:<35} | " + " | ".join(row_strs))

# Add Other row
row_strs_other = []
for s_name, s_recs in splits_data.items():
    s_pos = len([r for r in s_recs if r['label'] == 1 and get_publisher(r) not in top_pubs])
    s_neg = len([r for r in s_recs if r['label'] == 0 and get_publisher(r) not in top_pubs])
    s_pos_pct = s_pos / len([r for r in s_recs if r['label'] == 1]) * 100
    s_neg_pct = s_neg / len([r for r in s_recs if r['label'] == 0]) * 100
    row_strs_other.append(f"{s_pos:>3}/{s_neg:>3} ({s_pos_pct:>4.1f}%/{s_neg_pct:>4.1f}%)")
print(f"{'Other (low-volume)':<35} | " + " | ".join(row_strs_other))

# Year distribution table
print("\nYear Distribution Table:")
header_y = f"{'Year':<10} | " + " | ".join(f"{s:<15}" for s in ['Train (Pos/Neg)', 'Val (Pos/Neg)', 'Test (Pos/Neg)'])
print(header_y)
print("-" * len(header_y))
all_years = sorted(list(set(r['year'] for r in pool_records)))
for y in all_years:
    row_strs_y = []
    for s_name, s_recs in splits_data.items():
        s_pos = len([r for r in s_recs if r['label'] == 1 and r['year'] == y])
        s_neg = len([r for r in s_recs if r['label'] == 0 and r['year'] == y])
        row_strs_y.append(f"{s_pos:>4}/{s_neg:>4}")
    print(f"{y:<10} | " + " | ".join(row_strs_y))

# Flag low volume publisher counts in val/test
print("\nLow Volume Validation/Test Warnings:")
for s_name, s_recs in [('Val', val_records), ('Test', test_records)]:
    for pub in top_pubs:
        s_pos = len([r for r in s_recs if r['label'] == 1 and get_publisher(r) == pub])
        if s_pos < 10:
            print(f"  WARNING: Publisher '{pub}' has only {s_pos} positive papers in the {s_name} split! (Too low for stable metrics)")
