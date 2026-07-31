"""
Provenance-Matched Negative Control Construction
=================================================
Phase 1 of the pipeline-provenance artifact diagnosis.

Objective: Fetch legitimate (never-retracted) Hindawi articles through the 
POSITIVE acquisition pathway (direct PMID -> efetch) to test whether the 
PubMedBERT model's performance depends on acquisition-pipeline differences.

Protocol:
1. Use PubMed esearch to DISCOVER candidate PMIDs (same journals/years as holdout positives)
2. Exclude ALL existing dataset records (train/val/test/holdout) + full RWDB
3. Fetch abstracts using DIRECT PMID -> efetch (positive pathway), skipping esummary
4. Extract title, abstract, DOI, pub types from efetch XML directly (same parsing as positives)
5. Apply same oncology keyword + pub type filters as original pipeline
6. Save with full audit log for reproducibility

This script also isolates the 13 Crossref-fallback holdout positives for separate analysis.
"""
import sys
import os
import json
import csv
import time
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import random
from collections import defaultdict, Counter
from datetime import datetime, timezone

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout.reconfigure(encoding='utf-8')

CANCER_KEYWORDS = ['cancer', 'oncol', 'tumor', 'tumour', 'neoplas', 'carcinoma',
                   'leukemia', 'leukaemia', 'lymphoma', 'melanoma', 'sarcoma',
                   'metasta', 'malignan', 'glioma', 'osteosarcoma', 'neuroblastoma',
                   'hepatoma', 'myeloma']

CANCER_QUERY_PART = ("(cancer OR oncology OR tumor OR tumour OR neoplasm OR "
                     "carcinoma OR leukemia OR lymphoma OR melanoma OR sarcoma OR "
                     "metastasis OR malignant OR glioma)")

EXCLUDED_PUBTYPES = [
    'Review', 'Letter', 'Editorial', 'Comment', 'Published Erratum',
    'Retracted Publication', 'Retraction of Publication', 'Congresses',
    'Practice Guideline', 'Biography', 'Directory', 'Interview', 'News',
    'Retraction Notice'
]

BOILERPLATE_MARKERS = [
    'retract', 'withdraw', 'this article has been', 'has been retracted',
    'the above article', 'publisher regrets', 'editorial board',
    'removed from publication', 'no longer available',
    'expression of concern', 'paper mill'
]

def is_boilerplate(text):
    if not text or len(text.strip()) < 50:
        return True
    text_lower = text.lower().strip()
    first_200 = text_lower[:200]
    return sum(1 for m in BOILERPLATE_MARKERS if m in first_200) >= 2

# Audit log
audit_log = {
    'experiment': 'Provenance-Matched Negative Control',
    'timestamp_start': datetime.now(timezone.utc).isoformat(),
    'queries': [],
    'exclusions': {},
    'results': {}
}

def log_query(endpoint, params, response_summary):
    audit_log['queries'].append({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'endpoint': endpoint,
        'params': params,
        'response_summary': response_summary
    })

# =====================================================================
# 1. Load exclusion sets (ALL existing dataset records + full RWDB)
# =====================================================================
print("=" * 70)
print("PHASE 1: Loading Exclusion Sets")
print("=" * 70)

# Load all existing dataset DOIs and PMIDs
existing_dois = set()
existing_pmids = set()
for split_name in ['train', 'val', 'test', 'holdout']:
    fpath = os.path.join(project_root, 'data', 'final', f'cancer_pm_{split_name}.json')
    with open(fpath, 'r', encoding='utf-8') as f:
        recs = json.load(f)
        for r in recs:
            doi = r.get('doi', '').strip().lower()
            pmid = r.get('pmid', '').strip()
            if doi:
                existing_dois.add(doi)
            if pmid:
                existing_pmids.add(pmid)

print(f"Existing dataset exclusions: {len(existing_dois)} DOIs, {len(existing_pmids)} PMIDs")

# Load full RWDB exclusion list
rwdb_dois = set()
rwdb_pmids = set()
rwdb_path = os.path.join(project_root, 'data', 'raw', 'rwdb', 'retraction_watch.csv')
with open(rwdb_path, 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doi = row.get('OriginalPaperDOI', '').strip().lower()
        pmid = row.get('OriginalPaperPubMedID', '').strip()
        if doi and doi != '0':
            rwdb_dois.add(doi)
        if pmid and pmid != '0':
            rwdb_pmids.add(pmid)

print(f"RWDB exclusions: {len(rwdb_dois)} DOIs, {len(rwdb_pmids)} PMIDs")

all_excluded_dois = existing_dois | rwdb_dois
all_excluded_pmids = existing_pmids | rwdb_pmids
print(f"Total exclusions: {len(all_excluded_dois)} DOIs, {len(all_excluded_pmids)} PMIDs")

audit_log['exclusions'] = {
    'existing_dataset_dois': len(existing_dois),
    'existing_dataset_pmids': len(existing_pmids),
    'rwdb_dois': len(rwdb_dois),
    'rwdb_pmids': len(rwdb_pmids),
    'total_excluded_dois': len(all_excluded_dois),
    'total_excluded_pmids': len(all_excluded_pmids),
}

# =====================================================================
# 2. Load Hindawi holdout positives to extract target cells
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 2: Extracting Target Cells from Holdout Positives")
print("=" * 70)

with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_holdout.json'), 'r', encoding='utf-8') as f:
    holdout = json.load(f)

holdout_pos = [r for r in holdout if r['label'] == 1]
print(f"Holdout positives: {len(holdout_pos)}")

# NLM mapping
with open(os.path.join(project_root, 'data', 'final', 'journal_to_nlm.json'), 'r', encoding='utf-8') as f:
    journal_to_nlm = json.load(f)
nlm_to_raw = {v: k for k, v in journal_to_nlm.items()}

# Build target cells: (journal, year) -> count
target_cells = Counter()
for r in holdout_pos:
    journal = r.get('journal', '').strip()
    year = r.get('year', 0)
    if journal and year:
        # Map to NLM if possible (for PubMed query), or use raw name
        nlm = journal_to_nlm.get(journal, journal)
        target_cells[(nlm, year)] += 1

print(f"Target cells (journal x year): {len(target_cells)}")
for (j, y), c in sorted(target_cells.items(), key=lambda x: -x[1])[:10]:
    print(f"  {j} ({y}): {c} positives")

# =====================================================================
# 3. PubMed esearch for candidate discovery (NOT the negative pathway)
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 3: Candidate Discovery via PubMed esearch")
print("=" * 70)

def search_pubmed(nlm_journal, year, retmax):
    """Use esearch to find candidate PMIDs. Same query format as negative pipeline."""
    term = (f'"{nlm_journal}"[Journal] AND {year}[DP] AND {CANCER_QUERY_PART} '
            f'NOT "Retracted Publication"[PT] NOT "Retraction of Publication"[PT]')
    url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
           f"db=pubmed&term={urllib.parse.quote(term)}&retmode=json&retmax={retmax}")
    headers = {'User-Agent': 'CancerPMProvControl/1.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            res = json.loads(r.read().decode('utf-8'))
            pmids = res.get('esearchresult', {}).get('idlist', [])
            log_query('esearch', {'journal': nlm_journal, 'year': year, 'retmax': retmax}, 
                     {'count': len(pmids)})
            return pmids
    except Exception as e:
        log_query('esearch', {'journal': nlm_journal, 'year': year}, {'error': str(e)})
        return []

# Gather candidate PMIDs for each target cell
random.seed(42)
all_candidate_pmids = {}  # pmid -> (journal, year)

for (nlm_journal, year), pos_count in sorted(target_cells.items(), key=lambda x: -x[1]):
    # Target: match the positive count (not 5x like negative pipeline)
    # We want ~1 provenance-matched negative per positive to maximize N
    target_n = pos_count
    
    print(f"\nSearching '{nlm_journal}' ({year}): target {target_n} negatives...")
    
    # Primary search
    pmids = search_pubmed(nlm_journal, year, max(500, target_n * 10))
    time.sleep(0.35)
    
    # Fallback: year ±1
    if len(pmids) < target_n * 3:
        print(f"  Primary search thin ({len(pmids)}). Widening to {year-1} and {year+1}...")
        pmids_m1 = search_pubmed(nlm_journal, year - 1, max(200, target_n * 5))
        time.sleep(0.35)
        pmids_p1 = search_pubmed(nlm_journal, year + 1, max(200, target_n * 5))
        time.sleep(0.35)
        pmids = list(set(pmids + pmids_m1 + pmids_p1))
    
    # Filter out excluded PMIDs
    clean_pmids = [p for p in pmids if p not in all_excluded_pmids]
    print(f"  Found {len(pmids)} candidates, {len(clean_pmids)} after PMID exclusion")
    
    # Sample
    if len(clean_pmids) > target_n * 3:
        sampled = random.sample(clean_pmids, target_n * 3)  # oversample for filtering
    else:
        sampled = clean_pmids
    
    for pmid in sampled:
        if pmid not in all_candidate_pmids:
            all_candidate_pmids[pmid] = (nlm_journal, year)

print(f"\nTotal unique candidate PMIDs after discovery: {len(all_candidate_pmids)}")

# =====================================================================
# 4. DIRECT PMID -> efetch (POSITIVE PATHWAY) - no esummary!
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 4: Direct PMID -> efetch (Positive Acquisition Pathway)")
print("=" * 70)
print("NOTE: Using ONLY efetch XML parsing, same code as positive pipeline.")
print("      Skipping esummary entirely to match positive acquisition pathway.")

def fetch_pubmed_full_batch(pmids):
    """
    Direct PMID -> efetch batch. Same parsing as fetch_abstracts.py lines 65-100.
    Returns dict: pmid -> {title, abstract, doi, pubtypes, journal, year}
    """
    if not pmids:
        return {}
    
    url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
           f"db=pubmed&id={','.join(pmids)}&retmode=xml")
    headers = {'User-Agent': 'CancerPMProvControl/1.0'}
    req = urllib.request.Request(url, headers=headers)
    results = {}
    
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            xml_text = r.read().decode('utf-8', errors='replace')
        
        root = ET.fromstring(xml_text)
        
        for article in root.findall('.//PubmedArticle'):
            pmid_el = article.find('.//PMID')
            pmid = pmid_el.text.strip() if pmid_el is not None else ""
            if not pmid:
                continue
            
            # === ABSTRACT EXTRACTION (same as fetch_abstracts.py lines 84-96) ===
            ab_parts = []
            for ab_el in article.findall('.//AbstractText'):
                if ab_el.text:
                    label = ab_el.attrib.get('Label')
                    if label:
                        ab_parts.append(f"{label}: {ab_el.text.strip()}")
                    else:
                        ab_parts.append(ab_el.text.strip())
            abstract = " ".join(ab_parts) if ab_parts else ""
            
            # === TITLE EXTRACTION (from XML, not esummary) ===
            title_el = article.find('.//ArticleTitle')
            title = ""
            if title_el is not None and title_el.text:
                title = title_el.text.strip()
            
            # === DOI EXTRACTION (from XML, not esummary) ===
            doi = ""
            for aid in article.findall('.//ArticleId'):
                if aid.attrib.get('IdType') == 'doi' and aid.text:
                    doi = aid.text.strip().lower()
                    break
            
            # === PUB TYPES (from XML, not esummary) ===
            pubtypes = []
            for pt_el in article.findall('.//PublicationType'):
                if pt_el.text:
                    pubtypes.append(pt_el.text.strip())
            
            # === JOURNAL (from XML) ===
            journal_el = article.find('.//ISOAbbreviation')
            journal = journal_el.text.strip() if journal_el is not None and journal_el.text else ""
            
            # === YEAR (from XML) ===
            year = 0
            year_el = article.find('.//PubDate/Year')
            if year_el is not None and year_el.text:
                try:
                    year = int(year_el.text.strip())
                except ValueError:
                    pass
            if not year:
                medline_date = article.find('.//PubDate/MedlineDate')
                if medline_date is not None and medline_date.text:
                    m = re.search(r'\b(19|20)\d{2}\b', medline_date.text)
                    if m:
                        year = int(m.group(0))
            
            results[pmid] = {
                'pmid': pmid,
                'doi': doi,
                'title': title,
                'abstract': abstract,
                'journal': journal,
                'year': year,
                'pubtypes': pubtypes,
                'abstract_source': 'PubMed' if abstract else 'None',
            }
        
        log_query('efetch', {'batch_size': len(pmids)}, 
                 {'fetched': len(results), 'with_abstract': sum(1 for r in results.values() if r['abstract'])})
    
    except Exception as e:
        log_query('efetch', {'batch_size': len(pmids)}, {'error': str(e)})
    
    return results

# Fetch in batches of 200 (same as positive pipeline)
pmid_list = list(all_candidate_pmids.keys())
all_fetched = {}
batch_size = 200

for i in range(0, len(pmid_list), batch_size):
    batch = pmid_list[i:i+batch_size]
    batch_num = i // batch_size + 1
    total_batches = (len(pmid_list) - 1) // batch_size + 1
    print(f"  Fetching batch {batch_num}/{total_batches} ({len(batch)} PMIDs)...")
    
    batch_results = fetch_pubmed_full_batch(batch)
    all_fetched.update(batch_results)
    time.sleep(0.4)  # rate limit

print(f"\nTotal records fetched via efetch: {len(all_fetched)}")
print(f"  With abstract: {sum(1 for r in all_fetched.values() if r['abstract'])}")
print(f"  Without abstract: {sum(1 for r in all_fetched.values() if not r['abstract'])}")

# =====================================================================
# 5. Filter and validate (same criteria as original pipeline)
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 5: Filtering and Validation")
print("=" * 70)

provenance_matched = []
filter_stats = Counter()

for pmid, rec in all_fetched.items():
    doi = rec['doi']
    title = rec['title']
    abstract = rec['abstract']
    pubtypes = rec['pubtypes']
    
    # 1. DOI exclusion check
    if doi and doi in all_excluded_dois:
        filter_stats['doi_excluded'] += 1
        continue
    
    # 2. Publication type filter (same as negative pipeline)
    if any(pt in EXCLUDED_PUBTYPES for pt in pubtypes):
        filter_stats['pubtype_excluded'] += 1
        continue
    
    # 3. Title heuristic exclusions
    title_lower = title.lower()
    if any(kw in title_lower for kw in ['corrigendum', 'erratum', 'retract', 'withdraw', 
                                          'expression of concern', 'review']):
        filter_stats['title_excluded'] += 1
        continue
    
    # 4. Must have abstract
    if not abstract or is_boilerplate(abstract):
        filter_stats['no_abstract'] += 1
        continue
    
    # 5. Oncology keyword verification (same as fetch_abstracts.py)
    combined_text = (title + " " + abstract).lower()
    if not any(kw in combined_text for kw in CANCER_KEYWORDS):
        filter_stats['not_oncology'] += 1
        continue
    
    # 6. Final RWDB DOI check (double-check)
    if doi and doi in rwdb_dois:
        filter_stats['rwdb_doi_excluded'] += 1
        continue
    
    # Passed all filters
    target_journal, target_year = all_candidate_pmids.get(pmid, ('', 0))
    provenance_matched.append({
        'doi': doi,
        'pmid': pmid,
        'title': title,
        'abstract': abstract,
        'journal': rec['journal'],
        'year': rec['year'],
        'label': 0,  # These are known-legitimate articles
        'abstract_source': rec['abstract_source'],
        'oncology_verified': True,
        'target_journal': target_journal,
        'target_year': target_year,
    })
    filter_stats['accepted'] += 1

print(f"Filter statistics:")
for reason, count in filter_stats.most_common():
    print(f"  {reason}: {count}")

# Sample to match holdout positive count if we have excess
if len(provenance_matched) > len(holdout_pos) * 2:
    random.seed(42)
    # Stratified sampling by journal×year to match positive distribution
    by_cell = defaultdict(list)
    for r in provenance_matched:
        key = (r['target_journal'], r['target_year'])
        by_cell[key].append(r)
    
    sampled = []
    for (j, y), pos_count in target_cells.items():
        cell_records = by_cell.get((j, y), [])
        target = pos_count  # 1:1 matching
        if len(cell_records) > target:
            sampled.extend(random.sample(cell_records, target))
        else:
            sampled.extend(cell_records)
    
    # If we need more, add from remaining
    remaining = [r for r in provenance_matched if r not in sampled]
    if len(sampled) < 300 and remaining:
        extra_needed = min(300 - len(sampled), len(remaining))
        sampled.extend(random.sample(remaining, extra_needed))
    
    provenance_matched = sampled

print(f"\nFinal provenance-matched negative set: N = {len(provenance_matched)}")

# Journal distribution
journal_dist = Counter(r['journal'] for r in provenance_matched)
print("\nJournal distribution:")
for j, c in journal_dist.most_common(15):
    print(f"  {j}: {c}")

# Year distribution
year_dist = Counter(r['year'] for r in provenance_matched)
print("\nYear distribution:")
for y, c in sorted(year_dist.items()):
    print(f"  {y}: {c}")

# =====================================================================
# 6. Save results and audit log
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 6: Saving Results and Audit Log")
print("=" * 70)

# Save provenance-matched negatives
out_path = os.path.join(project_root, 'data', 'final', 'provenance_matched_negatives.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(provenance_matched, f, indent=2, ensure_ascii=False)
print(f"Saved {len(provenance_matched)} provenance-matched negatives to {out_path}")

# Also separate the 13 Crossref-fallback holdout positives
holdout_crossref = [r for r in holdout_pos if r.get('abstract_source') == 'Crossref']
holdout_pubmed = [r for r in holdout_pos if r.get('abstract_source') == 'PubMed']
print(f"\nCrossref-fallback holdout positives: N = {len(holdout_crossref)}")
print(f"PubMed holdout positives: N = {len(holdout_pubmed)}")

crossref_out = os.path.join(project_root, 'data', 'final', 'holdout_crossref_positives.json')
with open(crossref_out, 'w', encoding='utf-8') as f:
    json.dump(holdout_crossref, f, indent=2, ensure_ascii=False)
print(f"Saved Crossref-fallback positives to {crossref_out}")

pubmed_out = os.path.join(project_root, 'data', 'final', 'holdout_pubmed_positives.json')
with open(pubmed_out, 'w', encoding='utf-8') as f:
    json.dump(holdout_pubmed, f, indent=2, ensure_ascii=False)
print(f"Saved PubMed holdout positives to {pubmed_out}")

# Save audit log
audit_log['timestamp_end'] = datetime.now(timezone.utc).isoformat()
audit_log['results'] = {
    'total_candidates_discovered': len(all_candidate_pmids),
    'total_fetched': len(all_fetched),
    'total_accepted': len(provenance_matched),
    'filter_stats': dict(filter_stats),
    'journal_distribution': dict(journal_dist),
    'year_distribution': {str(k): v for k, v in year_dist.items()},
    'crossref_holdout_positives': len(holdout_crossref),
    'pubmed_holdout_positives': len(holdout_pubmed),
    'total_api_calls': len(audit_log['queries']),
}

audit_path = os.path.join(project_root, 'data', 'final', 'provenance_control_audit.json')
with open(audit_path, 'w', encoding='utf-8') as f:
    json.dump(audit_log, f, indent=2, ensure_ascii=False, default=str)
print(f"Saved audit log ({len(audit_log['queries'])} API calls logged) to {audit_path}")

print("\n" + "=" * 70)
print("PHASE 1 CONSTRUCTION COMPLETE")
print("=" * 70)
print(f"Provenance-matched negatives: N = {len(provenance_matched)}")
print(f"Crossref-fallback positives: N = {len(holdout_crossref)}")
print(f"PubMed positives: N = {len(holdout_pubmed)}")
