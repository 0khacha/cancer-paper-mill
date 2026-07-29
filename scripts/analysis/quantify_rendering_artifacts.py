import json
import re
import csv
import sys

# Write directly to file
output_path = r"C:\projects\cancer-paper-mill\tr_results_final.txt"
sys.stdout = open(output_path, "w", encoding="utf-8")
sys.stderr = sys.stdout

# Load all datasets
files = [
    "data/final/cancer_pm_train.json",
    "data/final/cancer_pm_val.json",
    "data/final/cancer_pm_test.json",
    "data/final/cancer_pm_holdout.json"
]

all_recs = []
for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        all_recs.extend(json.load(f))

# Define clean inline header pattern:
# We match headers: BACKGROUND, OBJECTIVE, AIM, METHOD, METHODS, RESULT, RESULTS, CONCLUSION, CONCLUSIONS
# We specifically avoid matching in "MATERIALS AND METHODS" or "PATIENTS AND METHODS" by ensuring they are not preceded by "MATERIALS AND" or "PATIENTS AND"
inline_hdr_pat = re.compile(r'\b(BACKGROUND|OBJECTIVE|AIM|METHODS?|RESULTS?|CONCLUSIONS?):')

def get_publisher(rec, journal_to_publisher, nlm2raw):
    j = rec['journal']
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, 'Direct PubMed')
    if 'computational and mathematical methods in medicine' in raw_j.lower() or 'comput math methods med' in j.lower():
        pub = 'Hindawi'
    return pub

# Awkward phrasing templates (from typical mill boilerplate)
awkward_needles = [
    r"attempted to clarify",
    r"belongs to a type of the most deadly",
    r"on carcinogenic and development",
    r"will increase in the coming decades",
    r"was designed to explore Tanshinone",
    r"is dysregulated in multiple",
    r"is sponged by"
]
awkward_pat = re.compile("|".join(awkward_needles), re.IGNORECASE)

def analyze_abstract_headers(abstract, title):
    full_text = title + " " + abstract
    
    has_inline = False
    has_weird_preceding = False
    weird_matches = []
    
    for m in inline_hdr_pat.finditer(full_text):
        start_idx = m.start()
        if start_idx <= 10:
            continue  # At the very start of abstract is normal
            
        # Get preceding text, strip whitespace
        prec_text = full_text[:start_idx].rstrip()
        if not prec_text:
            continue
            
        # Check if this METHODS is part of "MATERIALS AND METHODS" or "PATIENTS AND METHODS"
        # We check if the last few words are "MATERIALS AND" or "PATIENTS AND"
        if m.group(1).startswith("METHOD"):
            if prec_text.upper().endswith("MATERIALS AND") or prec_text.upper().endswith("PATIENTS AND"):
                continue # Skip this composite header, it's not inline text concatenation
                
        has_inline = True
        
        # Check the last non-space character
        last_char = prec_text[-1]
        
        # A header is weirdly preceded if it is not preceded by sentence-ending punctuation (., !, ?)
        # We also tolerate ) and ] just in case, but let's see. 
        if last_char not in ['.', '!', '?']:
            has_weird_preceding = True
            weird_matches.append((last_char, full_text[max(0, start_idx-25):min(len(full_text), start_idx+35)]))
            
    has_awkward = bool(awkward_pat.search(full_text))
    
    return {
        "has_inline": has_inline,
        "has_weird_preceding": has_weird_preceding,
        "has_awkward": has_awkward,
        "weird_matches": weird_matches
    }

def process_group(name, records):
    n = len(records)
    inline_cnt = 0
    weird_cnt = 0
    awkward_cnt = 0
    both_cnt = 0
    neither_cnt = 0
    
    for r in records:
        abstract = r.get("abstract", "") or ""
        title = r.get("title", "") or ""
        
        res = analyze_abstract_headers(abstract, title)
        
        if res["has_weird_preceding"]:
            weird_cnt += 1
        if res["has_awkward"]:
            awkward_cnt += 1
            
        if res["has_weird_preceding"] and res["has_awkward"]:
            both_cnt += 1
        elif not res["has_weird_preceding"] and not res["has_awkward"]:
            neither_cnt += 1
            
    print(f"\nGroup: {name} (N = {n})")
    print(f"  Inline Concatenation without Proper Sentence-Ending Punctuation (weird punctuation or none): {weird_cnt} / {n} ({weird_cnt/n*100:.2f}%)")
    print(f"  Awkward Phrasing Pattern: {awkward_cnt} / {n} ({awkward_cnt/n*100:.2f}%)")
    print(f"  Overlap:")
    print(f"    Both Concatenation AND Awkward: {both_cnt} / {n} ({both_cnt/n*100:.2f}%)")
    print(f"    Only Concatenation: {weird_cnt - both_cnt} / {n} ({(weird_cnt - both_cnt)/n*100:.2f}%)")
    print(f"    Only Awkward Phrasing: {awkward_cnt - both_cnt} / {n} ({(awkward_cnt - both_cnt)/n*100:.2f}%)")
    print(f"    Neither: {neither_cnt} / {n} ({neither_cnt/n*100:.2f}%)")
    
    return {
        "name": name,
        "total": n,
        "weird": weird_cnt,
        "awkward": awkward_cnt,
        "both": both_cnt,
        "neither": neither_cnt
    }

# Gather publisher mappings
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

# Set up the groups
with open("data/final/cancer_pm_val.json", "r", encoding="utf-8") as f:
    val_recs = json.load(f)
with open("data/final/cancer_pm_holdout.json", "r", encoding="utf-8") as f:
    holdout_recs = json.load(f)
with open("data/final/cancer_pm_train.json", "r", encoding="utf-8") as f:
    train_recs = json.load(f)
with open("data/final/cancer_pm_test.json", "r", encoding="utf-8") as f:
    test_recs = json.load(f)

# Sort out validation subgroup B
doi_to_reasons = {}
with open('data/raw/rwdb/retraction_watch.csv', 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doi = row.get('OriginalPaperDOI', '').strip().lower()
        reasons = row.get('Reason', '').strip()
        if doi:
            doi_to_reasons[doi] = reasons

val_hindawi = [r for r in val_recs if get_publisher(r, journal_to_publisher, nlm2raw) == 'Hindawi']
positives_val = [r for r in val_hindawi if r["label"] == 1]
negatives_val = [r for r in val_hindawi if r["label"] == 0]

sg_b = []
for r in positives_val:
    doi = r.get("doi", "").strip().lower()
    reasons = doi_to_reasons.get(doi, "")
    is_computer_generated = "computer-aided" in reasons.lower() or "computer-generated" in reasons.lower()
    if not is_computer_generated:
        sg_b.append(r)

holdout_hindawi_pos = [r for r in holdout_recs if r["label"] == 1]
holdout_hindawi_neg = [r for r in holdout_recs if r["label"] == 0]

train_val_test = train_recs + val_recs + test_recs
in_pool_hindawi_neg = [r for r in train_val_test if r["label"] == 0 and get_publisher(r, journal_to_publisher, nlm2raw) == 'Hindawi']

print("="*60)
print("FINAL PREVALENCE OF CLEAN SCHOLASTIC CORRUPTIONS")
print("="*60)

process_group("Sous-groupe B Positives", sg_b)
process_group("Hindawi Holdout Positives", holdout_hindawi_pos)
process_group("Validation Negatives (Hindawi)", negatives_val)
process_group("Holdout Negatives (Hindawi)", holdout_hindawi_neg)
process_group("In-Pool Negatives (Hindawi)", in_pool_hindawi_neg)

print("\nFinished successfully.")
