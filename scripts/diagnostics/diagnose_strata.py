"""
Diagnose the evaluation strata and publisher table tail:
1. Count records in Hindawi, Spandidos, and Non-Hindawi/Non-Spandidos strata.
2. Count positive records for each individual publisher inside the Non-Hindawi/Non-Spandidos stratum.
3. Identify low-volume publishers (<30 positives).
4. Calculate the sum and counts for the "Other" publisher row in the table.
"""
import sys
import json
import csv
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(sys.maxsize)

with open(r'c:\projects\cancer-paper-mill\data\cancer_pm_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

positives = [r for r in dataset if r['label'] == 1]
negatives = [r for r in dataset if r['label'] == 0]

# Load publisher mapping from RWDB
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

def get_publisher(rec):
    j = rec['journal']
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() or "comput math methods med" in j.lower():
        pub = "Hindawi"
    return pub

pos_pubs = Counter(get_publisher(r) for r in positives)
neg_pubs = Counter(get_publisher(r) for r in negatives)

# =====================================================================
# 1. Stratum counts
# =====================================================================
print("="*70)
print("1. EVALUATION STRATUM COUNTS")
print("="*70)

hindawi_pos = pos_pubs['Hindawi']
hindawi_neg = neg_pubs['Hindawi']

spandidos_pos = pos_pubs['Spandidos']
spandidos_neg = neg_pubs['Spandidos']

other_pos = sum(v for k,v in pos_pubs.items() if k not in ['Hindawi', 'Spandidos'])
other_neg = sum(v for k,v in neg_pubs.items() if k not in ['Hindawi', 'Spandidos'])

print(f"Stratum 1: Hindawi-only")
print(f"  Positives: {hindawi_pos} | Negatives: {hindawi_neg} | Ratio: {hindawi_neg/hindawi_pos:.2f}:1")
print(f"Stratum 2: Spandidos-only")
print(f"  Positives: {spandidos_pos} | Negatives: {spandidos_neg} | Ratio: {spandidos_neg/spandidos_pos:.2f}:1")
print(f"Stratum 3: Non-Hindawi/Non-Spandidos (Pooled Others)")
print(f"  Positives: {other_pos} | Negatives: {other_neg} | Ratio: {other_neg/other_pos:.2f}:1")

# =====================================================================
# 2. Strata breakdown inside Non-Hindawi/Non-Spandidos
# =====================================================================
print("\n" + "="*70)
print("2. NON-HINDAWI/NON-SPANDIDOS INDIVIDUAL PUBLISHER COUNTS")
print("="*70)

print("Positives counts for individual publishers:")
other_pos_pubs = {k: v for k, v in pos_pubs.items() if k not in ['Hindawi', 'Spandidos']}
for pub, count in sorted(other_pos_pubs.items(), key=lambda x: -x[1]):
    neg_count = neg_pubs.get(pub, 0)
    print(f"  {pub:<40} | Pos: {count:>4} | Neg: {neg_count:>4}")

# =====================================================================
# 3. Calculate "Other" Row for final table (Top 9 publishers listed in card)
# =====================================================================
print("\n" + "="*70)
print("3. FINAL PUBLISHER TABLE RECONCILIATION")
print("="*70)

top_pubs = [
    'Hindawi', 'Spandidos', 'Wiley', 'Verduci Editore',
    'Taylor and Francis - Dove Press', 'Portland Press',
    'Royal Society of Chemistry (RSC)', 'Elsevier',
    'Taylor and Francis'
]

top_pos_sum = sum(pos_pubs[p] for p in top_pubs)
top_neg_sum = sum(neg_pubs[p] for p in top_pubs)

other_row_pos = len(positives) - top_pos_sum
other_row_neg = len(negatives) - top_neg_sum

print(f"Top 9 Publishers Sum: Pos={top_pos_sum} ({top_pos_sum/len(positives)*100:.1f}%), Neg={top_neg_sum} ({top_neg_sum/len(negatives)*100:.1f}%)")
print(f"Remaining 'Other' Row: Pos={other_row_pos} ({other_row_pos/len(positives)*100:.1f}%), Neg={other_row_neg} ({other_row_neg/len(negatives)*100:.1f}%)")
print(f"Total dataset sum verify: Pos={top_pos_sum + other_row_pos}, Neg={top_neg_sum + other_row_neg}")
