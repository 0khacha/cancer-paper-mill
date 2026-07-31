"""
Detailed analysis of Step 5 purges and final dataset validation:
1. Breakdown of the 434 positive and 243 negative purges into abstract-failure vs oncology-failure.
2. Purge rate asymmetry analysis by journal and year to check for selection bias.
3. Quantify the RWDB positives that had valid abstracts but failed oncology check.
4. Tracing the 186 positive papers without PMIDs: did they survive via Crossref?
5. Funding statement presence/absence audit on abstracts.
6. Full year distribution of the final clean dataset (2,117 pos / 9,030 neg).
7. Publisher and year distribution comparison on the final dataset.
"""
import sys
import json
import csv
import urllib.request
import re
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(sys.maxsize)

CANCER_KEYWORDS = ['cancer', 'oncol', 'tumor', 'tumour', 'neoplas', 'carcinoma',
                   'leukemia', 'leukaemia', 'lymphoma', 'melanoma', 'sarcoma',
                   'metasta', 'malignan', 'glioma', 'osteosarcoma', 'neuroblastoma',
                   'hepatoma', 'myeloma']

FUNDING_KEYWORDS = ['grant', 'fund', 'support', 'sponsor', 'award', 'foundation', 'scholarship', 'financial assistance']

# Load inputs and outputs
with open(r'c:\projects\cancer-paper-mill\data\clean_positives.json', 'r', encoding='utf-8') as f:
    orig_positives = json.load(f)

with open(r'c:\projects\cancer-paper-mill\data\clean_negatives.json', 'r', encoding='utf-8') as f:
    orig_negatives = json.load(f)

with open(r'c:\projects\cancer-paper-mill\data\cancer_pm_dataset.json', 'r', encoding='utf-8') as f:
    final_dataset = json.load(f)

final_pos = [r for r in final_dataset if r['label'] == 1]
final_neg = [r for r in final_dataset if r['label'] == 0]

print("="*70)
print("1. PURGE BREAKDOWN (ABSTRACT FAILURE VS ONCOLOGY FAILURE)")
print("="*70)

# Let's reconstruct the status of each original record during V6 run
# To do this, we re-run the V6 logic on the original datasets
# For positives:
pos_abstract_fail = 0
pos_oncology_fail = 0
pos_purged_records = []

# Fetch abstracts mapping from final_dataset (since we saved them there)
pmid_to_abstract = {r['pmid']: r['abstract'] for r in final_dataset if r['pmid']}
doi_to_abstract = {r['doi'].lower().strip(): r['abstract'] for r in final_dataset if r['doi']}

# Also load from raw fetched results if we saved them
with open(r'c:\projects\cancer-paper-mill\data\fetched_raw_dataset.json', 'r', encoding='utf-8') as f:
    raw_fetched = json.load(f)

raw_pos_fetched = [r for r in raw_fetched if r['label'] == 1]
raw_neg_fetched = [r for r in raw_fetched if r['label'] == 0]

# Positives breakdown
for r in raw_pos_fetched:
    ab = r['abstract'].strip()
    doi = r['doi'].lower().strip()
    title = r['title']
    
    # Check if this record is in final_pos
    in_final = any(p['doi'].lower().strip() == doi for p in final_pos)
    if not in_final:
        # Purged!
        if not ab:
            pos_abstract_fail += 1
        else:
            pos_oncology_fail += 1
        pos_purged_records.append(r)

# Negatives breakdown
neg_abstract_fail = 0
neg_oncology_fail = 0
neg_purged_records = []

for r in raw_neg_fetched:
    ab = r['abstract'].strip()
    doi = r['doi'].lower().strip()
    title = r['title']
    
    in_final = any(n['doi'].lower().strip() == doi for n in final_neg)
    if not in_final:
        if not ab:
            neg_abstract_fail += 1
        else:
            neg_oncology_fail += 1
        neg_purged_records.append(r)

print(f"Positives Purged (Total={len(pos_purged_records)}):")
print(f"  (a) Failed abstract recovery (empty/boilerplate): {pos_abstract_fail}")
print(f"  (b) Failed oncology relevance (abstract recovered but lacked keywords): {pos_oncology_fail}")

print(f"\nNegatives Purged (Total={len(neg_purged_records)}):")
print(f"  (a) Failed abstract recovery (empty/boilerplate): {neg_abstract_fail}")
print(f"  (b) Failed oncology relevance (abstract recovered but lacked keywords): {neg_oncology_fail}")

# =====================================================================
# 2. Purge Rate Concentration and Selection Bias Analysis
# =====================================================================
print("\n" + "="*70)
print("2. PURGE CONCENTRATION & SELECTION BIAS ANALYSIS")
print("="*70)

# Check concentration by Journal
pos_purged_by_j = Counter(r['journal'] for r in pos_purged_records)
pos_total_by_j = Counter(p.get('Journal', '').strip() for p in orig_positives)

print("Top Journals with Positive Purges (Purged / Original):")
for j, count in pos_purged_by_j.most_common(5):
    total = pos_total_by_j[j]
    rate = count / total * 100 if total > 0 else 0
    print(f"  {j}: {count}/{total} purged ({rate:.1f}%)")

# Check concentration by Year
pos_purged_by_y = Counter(r['year'] for r in pos_purged_records)
pos_total_by_y = Counter()
for p in orig_positives:
    date_str = p.get('OriginalPaperDate', '').strip()
    year = None
    if date_str:
        parts = date_str.split('/')
        if len(parts) >= 3:
            try:
                year = int(parts[2].split(' ')[0])
            except ValueError:
                pass
    if year:
        pos_total_by_y[year] += 1

print("\nPositive Purges by Year (Purged / Original):")
for y in sorted(pos_total_by_y.keys()):
    count = pos_purged_by_y[y]
    total = pos_total_by_y[y]
    rate = count / total * 100 if total > 0 else 0
    print(f"  {y}: {count}/{total} purged ({rate:.1f}%)")

# =====================================================================
# 3. Tracing the 186 positive papers without PMIDs
# =====================================================================
print("\n" + "="*70)
print("3. TRACING THE 186 POSITIVE PAPERS WITHOUT PMIDS")
print("="*70)

# Identify the 186 positive papers that originally lacked PMIDs in RWDB
pmidless_positives = []
with open(r'c:\projects\cancer-paper-mill\data\clean_positives.json', 'r', encoding='utf-8') as f:
    clean_pos_orig = json.load(f)

for p in clean_pos_orig:
    pmid = p.get('OriginalPaperPubMedID', '').strip()
    # Check if it was originally '0' or empty in RWDB
    # In raw retraction_watch.csv, PMIDs can be '0', '', or 'None'
    # We find if it has PMID == '0' or empty in clean_positives.json before our search step
    # Note: in clean_positives.json it is still '0' or empty because search_pmid_by_doi only resolved it in memory or in raw_fetched
    if not pmid or pmid == '0' or pmid == '':
        pmidless_positives.append(p)

print(f"Original positives lacking PMID: {len(pmidless_positives)}")

survived_pmidless = []
resolved_to_pmid_survived = 0
crossref_only_survived = 0

for p in pmidless_positives:
    doi = p.get('OriginalPaperDOI', '').strip().lower()
    # Find in final_pos
    match = [r for r in final_pos if r['doi'].lower().strip() == doi]
    if match:
        item = match[0]
        survived_pmidless.append(item)
        if item['abstract_source'] == 'PubMed':
            resolved_to_pmid_survived += 1
        elif item['abstract_source'] == 'Crossref':
            crossref_only_survived += 1

print(f"PMID-less positives surviving in final dataset: {len(survived_pmidless)}/{len(pmidless_positives)} ({len(survived_pmidless)/len(pmidless_positives)*100:.1f}%)")
print(f"  Surviving via resolved PMID + PubMed: {resolved_to_pmid_survived}")
print(f"  Surviving via Crossref-only verification: {crossref_only_survived}")

# =====================================================================
# 5. Funding Statement Audit
# =====================================================================
print("\n" + "="*70)
print("5. FUNDING STATEMENT AUDIT ON ABSTRACTS")
print("="*70)

pos_funding_count = 0
for r in final_pos:
    ab_lower = r['abstract'].lower()
    if any(kw in ab_lower for kw in FUNDING_KEYWORDS):
        pos_funding_count += 1

neg_funding_count = 0
for r in final_neg:
    ab_lower = r['abstract'].lower()
    if any(kw in ab_lower for kw in FUNDING_KEYWORDS):
        neg_funding_count += 1

print(f"Positives containing funding terms in abstract: {pos_funding_count}/{len(final_pos)} ({pos_funding_count/len(final_pos)*100:.1f}%)")
print(f"Negatives containing funding terms in abstract: {neg_funding_count}/{len(final_neg)} ({neg_funding_count/len(final_neg)*100:.1f}%)")

# =====================================================================
# 6. Full Year Distribution
# =====================================================================
print("\n" + "="*70)
print("6. FULL YEAR DISTRIBUTION OF FINAL DATASET")
print("="*70)

year_counts = Counter(str(r.get('year', '')) for r in final_dataset)
total_years_sum = 0
for y, c in sorted(year_counts.items()):
    print(f"  {y}: {c}")
    total_years_sum += c
print(f"Total year sum: {total_years_sum}")

# =====================================================================
# 7. Post-Purge Publisher & Year Match Verification
# =====================================================================
print("\n" + "="*70)
print("7. POST-PURGE PUBLISHER & YEAR MATCH VERIFICATION")
print("="*70)

# Publisher distribution check
journal_to_publisher = {}
with open(r'c:\projects\cancer-paper-mill\data\raw\rwdb\retraction_watch.csv', 'r', encoding='utf-8', errors='replace') as f:
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

with open(r'c:\projects\cancer-paper-mill\data\journal_to_nlm.json', 'r', encoding='utf-8') as f:
    j2nlm = json.load(f)
nlm2raw = {v: k for k, v in j2nlm.items()}

final_pos_pubs = Counter()
for r in final_pos:
    raw_j = nlm2raw.get(r['journal'], r['journal'])
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() or "comput math methods med" in r['journal'].lower():
        pub = "Hindawi"
    final_pos_pubs[pub] += 1

final_neg_pubs = Counter()
for r in final_neg:
    raw_j = nlm2raw.get(r['journal'], r['journal'])
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() or "comput math methods med" in r['journal'].lower():
        pub = "Hindawi"
    final_neg_pubs[pub] += 1

print(f"{'Publisher':<40} | {'Pos Share (%)':<15} | {'Neg Share (%)':<15} | {'Difference (%)':<15}")
print("-"*95)
for pub, p_count in final_pos_pubs.most_common(12):
    p_pct = p_count / len(final_pos) * 100
    n_count = final_neg_pubs[pub]
    n_pct = n_count / len(final_neg) * 100 if len(final_neg) > 0 else 0
    diff = p_pct - n_pct
    print(f"{pub[:38]:<40} | {p_pct:>12.1f}% | {n_pct:>12.1f}% | {diff:>12.1f}%")

# Year match check
print("\nYear Match Check:")
print(f"{'Year':<10} | {'Pos Count':<12} | {'Neg Count':<12} | {'Ratio':<10}")
print("-"*50)
pos_years = Counter(r['year'] for r in final_pos)
neg_years = Counter(r['year'] for r in final_neg)

for y in sorted(pos_years.keys() | neg_years.keys()):
    pc = pos_years[y]
    nc = neg_years[y]
    r_val = nc / pc if pc > 0 else 0
    print(f"{y:<10} | {pc:>12} | {nc:>12} | {r_val:>8.2f}:1")
