"""
Clean Negative Class Generator (v5) - Self-Healing NLM mapping + Year Widening Fallback + PubMed [PT] Filter
Fixes:
1. Re-establishes article-level cancer topic filter in PubMed query AND explicit Title validation.
2. Official PubMed [PT] (Publication Type) filtering: excludes Reviews, Letters, Editorials, Errata, Retractions.
3. Target list deduplicated by NLM Abbreviation to prevent target collisions.
4. Excludes delisted/corrupted paper-mill journals (e.g. Cellular Physiology and Biochemistry, JBUON) from negative pool.
5. Dynamic year-widening fallback (±1 year) for depleted target cells to recover the 5:1 ratio.
6. Parses actual publication year from PubMed pubdate.
7. Saves clean negatives.
"""
import sys
import os

# Setup portable project root relative to script location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import csv
import json
import time
import urllib.request
import urllib.parse
import random
import re
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(sys.maxsize)

CANCER_KEYWORDS = ['cancer', 'oncol', 'tumor', 'tumour', 'neoplas', 'carcinoma',
                   'leukemia', 'leukaemia', 'lymphoma', 'melanoma', 'sarcoma',
                   'metasta', 'malignan', 'glioma', 'osteosarcoma', 'neuroblastoma',
                   'hepatoma', 'myeloma']

CANCER_QUERY_PART = "(cancer OR oncology OR tumor OR tumour OR neoplasm OR carcinoma OR leukemia OR lymphoma OR melanoma OR sarcoma OR metastasis OR malignant OR glioma)"

EXCLUDED_PUBTYPES = [
    'Review', 'Letter', 'Editorial', 'Comment', 'Published Erratum', 
    'Retracted Publication', 'Retraction of Publication', 'Congresses', 
    'Practice Guideline', 'Biography', 'Directory', 'Interview', 'News',
    'Retraction Notice'
]

COMPROMISED_JOURNALS = [
    'cellular physiology and biochemistry', 'cell physiol biochem',
    'jbuon', 'journal of balkan union of oncology'
]

# =====================================================================
# 1. Load Positive Class
# =====================================================================
print("="*70)
print("PHASE 1: Loading Clean Positive Class")
print("="*70)

with open(os.path.join(project_root, 'data', 'clean_positives.json'), 'r', encoding='utf-8') as f:
    clean_positives = json.load(f)

pos_dois = set(p.get('OriginalPaperDOI', '').strip().lower() for p in clean_positives if p.get('OriginalPaperDOI'))
print(f"Loaded {len(clean_positives)} clean positives.")

# Load global RWDB exclusion list (DOIs and PMIDs)
global_excluded_dois = set()
global_excluded_pmids = set()
with open(os.path.join(project_root, 'data', 'raw', 'rwdb', 'retraction_watch.csv'), 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doi = row.get('OriginalPaperDOI', '').strip().lower()
        pmid = row.get('OriginalPaperPubMedID', '').strip()
        if doi and doi != '0':
            global_excluded_dois.add(doi)
        if pmid and pmid != '0':
            global_excluded_pmids.add(pmid)

print(f"Loaded global exclusions: {len(global_excluded_dois)} DOIs, {len(global_excluded_pmids)} PMIDs.")

# =====================================================================
# 2. Build Deduplicated NLM Target Groups
# =====================================================================
print("\n" + "="*70)
print("PHASE 2: Target Group Mapping & NLM Resolution")
print("="*70)

# Load cached NLM mappings
cache_path = os.path.join(project_root, 'data', 'final', 'journal_to_nlm.json')
with open(cache_path, 'r', encoding='utf-8') as f:
    journal_to_nlm = json.load(f)

# Group positives by (NLM Journal Abbreviation, Year)
nlm_targets = defaultdict(int)
for rec in clean_positives:
    raw_journal = rec.get('Journal', '').strip()
    nlm_journal = journal_to_nlm.get(raw_journal, raw_journal)
    
    # Exclude compromised delisted journals
    if any(cj in nlm_journal.lower() or cj in raw_journal.lower() for cj in COMPROMISED_JOURNALS):
        continue
        
    date_str = rec.get('OriginalPaperDate', '').strip()
    year = None
    if date_str:
        parts = date_str.split('/')
        if len(parts) >= 3:
            try:
                year = int(parts[2].split(' ')[0])
            except ValueError:
                pass
    if nlm_journal and year:
        nlm_targets[(nlm_journal, year)] += 1

target_list = [
    {'nlm_journal': k[0], 'year': k[1], 'count': v, 'target_negatives': v * 5} 
    for k, v in nlm_targets.items()
]
print(f"Total deduplicated target matching groups: {len(target_list)}")

# =====================================================================
# 3. Query PubMed with Fallback Year Widening
# =====================================================================
print("\n" + "="*70)
print("PHASE 3: Generating Negatives (with ±1 year fallback)")
print("="*70)

def search_pubmed(nlm_journal, year, retmax):
    term = f'"{nlm_journal}"[Journal] AND {year}[DP] AND {CANCER_QUERY_PART} NOT "Retracted Publication"[PT] NOT "Retraction of Publication"[PT]'
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote(term)}&retmode=json&retmax={retmax}"
    headers = {'User-Agent': 'CancerNegativesV5/1.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            res = json.loads(r.read().decode('utf-8'))
            return res.get('esearchresult', {}).get('idlist', [])
    except Exception:
        return []

def fetch_summaries(pmids):
    if not pmids:
        return {}
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(pmids)}&retmode=json"
    headers = {'User-Agent': 'CancerNegativesV5/1.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            res = json.loads(r.read().decode('utf-8'))
            return res.get('result', {})
    except Exception:
        return {}

negatives_db = []
failed_groups = []
seen_global_dois = set()
random.seed(42)

for i, tg in enumerate(target_list):
    nlm_journal = tg['nlm_journal']
    year = tg['year']
    req_count = tg['target_negatives']
    
    print(f"[{i+1}/{len(target_list)}] Matching '{nlm_journal}' ({year}) - Needs {req_count} negatives...")
    
    # 1. Primary search (target year)
    candidate_pmids = search_pubmed(nlm_journal, year, max(250, req_count * 4))
    time.sleep(0.12)
    
    # 2. Fallback: Widening year window to year-1 and year+1 if primary search is depleted
    if len(candidate_pmids) < req_count * 2:
        print(f"  Primary search thin ({len(candidate_pmids)} candidates). Widening window to {year-1} and {year+1}...")
        adj_pmids_1 = search_pubmed(nlm_journal, year - 1, max(150, req_count * 2))
        time.sleep(0.12)
        adj_pmids_2 = search_pubmed(nlm_journal, year + 1, max(150, req_count * 2))
        time.sleep(0.12)
        candidate_pmids = list(set(candidate_pmids + adj_pmids_1 + adj_pmids_2))
        
    if not candidate_pmids:
        print(f"  WARNING: No candidates found for '{nlm_journal}'")
        failed_groups.append((nlm_journal, year, req_count, "No candidates found"))
        continue
        
    # Fetch summaries
    summaries = {}
    for batch_start in range(0, len(candidate_pmids), 150):
        batch = candidate_pmids[batch_start:batch_start+150]
        batch_summaries = fetch_summaries(batch)
        summaries.update(batch_summaries)
        time.sleep(0.12)
        
    valid_candidates = []
    for pmid in candidate_pmids:
        sum_data = summaries.get(pmid, {})
        if not sum_data or pmid == 'uid':
            continue
            
        title = sum_data.get('title', '')
        title_lower = title.lower()
        
        # 1. Oncology title keyword verification
        if not any(kw in title_lower for kw in CANCER_KEYWORDS):
            continue
            
        # 2. Official PubMed PT filtering
        pubtypes = sum_data.get('pubtype', [])
        if any(pt in EXCLUDED_PUBTYPES for pt in pubtypes):
            continue
            
        # 3. Title Heuristic exclusions
        if any(kw in title_lower for kw in ['corrigendum', 'erratum', 'retract', 'withdraw', 'expression of concern', 'review']):
            continue
            
        # Extract DOI
        doi = ""
        for aid in sum_data.get('articleids', []):
            if aid.get('idtype') == 'doi':
                doi = aid.get('value', '').strip().lower()
                break
                
        if not doi or doi in seen_global_dois:
            continue
            
        pmid_str = str(pmid).strip()
        
        # Exclude retracted papers globally
        if doi in global_excluded_dois or pmid_str in global_excluded_pmids or doi in pos_dois:
            continue
            
        # Parse actual publication year from summary pubdate
        pubdate = sum_data.get('pubdate', '')
        match = re.search(r'\b(19|20)\d{2}\b', pubdate)
        actual_year = int(match.group(0)) if match else year
        
        valid_candidates.append({
            'pmid': pmid_str,
            'doi': doi,
            'title': title,
            'journal': nlm_journal,
            'year': actual_year,
            'pubtypes': pubtypes
        })
        
    print(f"  Valid oncology candidates: {len(valid_candidates)}")
    
    if len(valid_candidates) < req_count:
        print(f"  WARNING: Insufficient candidates ({len(valid_candidates)}/{req_count}) even with ±1 year window")
        failed_groups.append((nlm_journal, year, req_count, f"Found {len(valid_candidates)} valid"))
        sampled = valid_candidates
    else:
        sampled = random.sample(valid_candidates, req_count)
        
    for s in sampled:
        seen_global_dois.add(s['doi'])
        negatives_db.append(s)
        
    print(f"  Successfully matched {len(sampled)} negatives.")

# Save final clean dataset
with open(os.path.join(project_root, 'data', 'clean_negatives.json'), 'w', encoding='utf-8') as f:
    json.dump(negatives_db, f, indent=2)

print("\n" + "="*70)
print("GENERATION V5 COMPLETE")
print("="*70)
print(f"Total clean positives: {len(clean_positives)}")
print(f"Total negatives matched: {len(negatives_db)}")
print(f"Failed/incomplete groups: {len(failed_groups)}")
