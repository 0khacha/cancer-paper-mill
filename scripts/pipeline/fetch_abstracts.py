"""
Step 5: Full Text & Abstract Recovery Cascade (V6 - Batch Efetch)
1. Resolves PubMed rate limits by using batch E-fetch (up to 200 PMIDs per call).
2. Bypasses Crossref rate limiting by using it as a secondary, rate-limited fallback.
3. Automatically searches PubMed by DOI for positives that lack PMIDs in RWDB.
4. Parses XML output from Efetch to extract clean abstract text.
5. Re-verifies oncology relevance using combined Title + Abstract for both classes.
"""
import sys
import os

# Setup portable project root relative to script location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import csv
import time
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(sys.maxsize)

CANCER_KEYWORDS = ['cancer', 'oncol', 'tumor', 'tumour', 'neoplas', 'carcinoma',
                   'leukemia', 'leukaemia', 'lymphoma', 'melanoma', 'sarcoma',
                   'metasta', 'malignan', 'glioma', 'osteosarcoma', 'neuroblastoma',
                   'hepatoma', 'myeloma']

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

# Search PubMed by DOI to find missing PMIDs
def search_pmid_by_doi(doi):
    if not doi or doi == '0':
        return ""
    term = f'"{doi}"[Location ID]'
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote(term)}&retmode=json"
    headers = {'User-Agent': 'CancerPaperMill/1.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode('utf-8'))
            ids = data.get('esearchresult', {}).get('idlist', [])
            if ids:
                return ids[0]
    except Exception:
        pass
    return ""

# Fetch summaries/abstracts from PubMed in batches of 200
def fetch_pubmed_abstracts_batch(pmids):
    if not pmids:
        return {}
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(pmids)}&retmode=xml"
    headers = {'User-Agent': 'CancerPaperMill/1.0'}
    req = urllib.request.Request(url, headers=headers)
    results = {}
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            xml_text = r.read().decode('utf-8', errors='replace')
            
            # Use ElementTree to parse XML safely
            root = ET.fromstring(xml_text)
            for article in root.findall('.//PubmedArticle'):
                pmid_el = article.find('.//PMID')
                pmid = pmid_el.text.strip() if pmid_el is not None else ""
                if not pmid:
                    continue
                    
                # Extract abstract parts
                ab_parts = []
                for ab_el in article.findall('.//AbstractText'):
                    if ab_el.text:
                        # Handle label (e.g. BACKGROUND, METHODS)
                        label = ab_el.attrib.get('Label')
                        if label:
                            ab_parts.append(f"{label}: {ab_el.text.strip()}")
                        else:
                            ab_parts.append(ab_el.text.strip())
                            
                abstract = " ".join(ab_parts) if ab_parts else ""
                results[pmid] = abstract
    except Exception as e:
        # print(f"Batch fetch error: {e}")
        pass
    return results

# Rate-limited single Crossref fallback
def fetch_crossref_abstract(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    headers = {'User-Agent': 'CancerPaperMillDataset/1.0 (mailto:research@example.com)'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode('utf-8'))
            ab = data.get('message', {}).get('abstract', '')
            if ab:
                ab = re.sub(r'<[^>]+>', '', ab).strip()
                return ab
    except Exception:
        pass
    return ""

def main():
    print("Loading clean positives and negatives...")
    with open(os.path.join(project_root, 'data', 'clean_positives.json'), 'r', encoding='utf-8') as f:
        positives = json.load(f)
    with open(os.path.join(project_root, 'data', 'clean_negatives.json'), 'r', encoding='utf-8') as f:
        negatives = json.load(f)
        
    print(f"Clean positives: {len(positives)}")
    print(f"Clean negatives: {len(negatives)}")
    
    # 1. Resolve missing PMIDs for positives via search
    print("Checking for positives with missing PubMed IDs...")
    resolved_count = 0
    for i, p in enumerate(positives):
        pmid = p.get('OriginalPaperPubMedID', '').strip()
        if not pmid or pmid == '0' or pmid == '':
            doi = p.get('OriginalPaperDOI', '').strip()
            # Search PubMed
            resolved_pmid = search_pmid_by_doi(doi)
            if resolved_pmid:
                p['OriginalPaperPubMedID'] = resolved_pmid
                resolved_count += 1
            time.sleep(0.12)
    print(f"Resolved {resolved_count} missing positive PMIDs.")
    
    # 2. Gather PMIDs and mapping
    pmid_to_record = {}
    pmids_to_fetch = []
    
    # Map positives by PMID
    for p in positives:
        pmid = p.get('OriginalPaperPubMedID', '').strip()
        if pmid and pmid != '0':
            pmid_to_record[pmid] = ('pos', p)
            pmids_to_fetch.append(pmid)
            
    # Map negatives by PMID
    for n in negatives:
        pmid = n.get('pmid', '').strip()
        if pmid and pmid != '0':
            pmid_to_record[pmid] = ('neg', n)
            pmids_to_fetch.append(pmid)
            
    print(f"Total PMIDs to fetch in batches: {len(pmids_to_fetch)}")
    
    # 3. Batch fetch PubMed abstracts (batches of 200)
    fetched_abstracts = {}
    batch_size = 200
    start_time = time.time()
    
    for i in range(0, len(pmids_to_fetch), batch_size):
        batch = pmids_to_fetch[i:i+batch_size]
        print(f"Fetching batch {i//batch_size + 1}/{(len(pmids_to_fetch)-1)//batch_size + 1}...")
        batch_results = fetch_pubmed_abstracts_batch(batch)
        fetched_abstracts.update(batch_results)
        time.sleep(0.35) # respect rate limit (approx 3 calls/sec)
        
    print(f"Batch fetch complete. Fetched {len(fetched_abstracts)} abstracts in {time.time() - start_time:.1f}s.")
    
    # 4. Fallback: Identify records that still lack abstracts and query Crossref
    print("\nRunning Crossref fallback for records with missing abstracts...")
    fallback_tasks = []
    
    # Check positives
    for p in positives:
        doi = p.get('OriginalPaperDOI', '').strip().lower()
        pmid = p.get('OriginalPaperPubMedID', '').strip()
        ab = fetched_abstracts.get(pmid, "")
        if not ab or is_boilerplate(ab):
            fallback_tasks.append(('pos', p))
            
    # Check negatives
    for n in negatives:
        doi = n.get('doi', '').strip().lower()
        pmid = n.get('pmid', '').strip()
        ab = fetched_abstracts.get(pmid, "")
        if not ab or is_boilerplate(ab):
            fallback_tasks.append(('neg', n))
            
    print(f"Found {len(fallback_tasks)} records requiring Crossref fallback.")
    
    # Single rate-limited Crossref fetch worker
    def run_fallback(task):
        label, rec = task
        doi = rec.get('OriginalPaperDOI', rec.get('doi', '')).strip()
        if doi:
            ab = fetch_crossref_abstract(doi)
            return doi, ab
        return "", ""
        
    fallback_results = {}
    if fallback_tasks:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_fallback, task): task for task in fallback_tasks}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                doi, ab = future.result()
                if doi and ab:
                    fallback_results[doi] = ab
                if completed % 100 == 0 or completed == len(fallback_tasks):
                    print(f"  Crossref fallback progress: {completed}/{len(fallback_tasks)}")
                time.sleep(0.08) # rate limiting
                
    # 5. Consolidate and Validate Relevance (Title + Abstract)
    print("\nConsolidating and validating dataset...")
    final_dataset = []
    pos_failed = []
    neg_failed = []
    
    # Helper to process and check final oncology relevance
    def finalize_and_verify(rec, label, is_pos=True):
        doi = rec.get('OriginalPaperDOI', rec.get('doi', '')).strip()
        pmid = rec.get('OriginalPaperPubMedID', rec.get('pmid', '')).strip()
        title = rec.get('Title', rec.get('title', '')).strip()
        journal = rec.get('Journal', rec.get('journal', '')).strip()
        
        # Parse year
        year = rec.get('year', '')
        if not year:
            date_str = rec.get('OriginalPaperDate', '').strip()
            if date_str:
                parts = date_str.split('/')
                if len(parts) >= 3:
                    try:
                        year = int(parts[2].split(' ')[0])
                    except ValueError:
                        pass
                        
        # Get abstract from efetch or fallback
        abstract = fetched_abstracts.get(pmid, "")
        source = "PubMed" if abstract else "None"
        
        if (not abstract or is_boilerplate(abstract)) and doi:
            abstract = fallback_results.get(doi.lower(), "")
            if abstract:
                source = "Crossref"
                
        # Final cleanup of boilerplate
        if is_boilerplate(abstract):
            abstract = ""
            source = "None"
            
        # Re-verify Oncology Relevance on Combined Title + Abstract
        combined_text = (title + " " + abstract).lower()
        is_oncology = any(kw in combined_text for kw in CANCER_KEYWORDS)
        
        result_item = {
            'doi': doi,
            'pmid': pmid,
            'title': title,
            'abstract': abstract,
            'journal': journal,
            'year': year,
            'label': label,
            'abstract_source': source,
            'oncology_verified': is_oncology
        }
        
        if is_oncology and abstract:
            final_dataset.append(result_item)
        else:
            if is_pos:
                pos_failed.append(result_item)
            else:
                neg_failed.append(result_item)

    # Process positives
    for p in positives:
        finalize_and_verify(p, 1, is_pos=True)
    # Process negatives
    for n in negatives:
        finalize_and_verify(n, 0, is_pos=False)
        
    # Final Split
    final_pos = [r for r in final_dataset if r['label'] == 1]
    final_neg = [r for r in final_dataset if r['label'] == 0]
    
    # Save datasets
    with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_dataset.json'), 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2)
        
    print("\n" + "="*70)
    print("STEP 5 ANALYSIS SUMMARY (V6 - BATCH EFETCH)")
    print("="*70)
    print(f"Total Clean Positives in Pool: {len(positives)}")
    print(f"Total Clean Negatives in Pool: {len(negatives)}")
    print(f"Total Failed Oncology/Abstract Validation in Positives: {len(pos_failed)}")
    print(f"Total Failed Oncology/Abstract Validation in Negatives: {len(neg_failed)}")
    
    print("\nFirst 3 failures in positives:")
    for r in pos_failed[:3]:
        print(f"  DOI: {r['doi']} | Source: {r['abstract_source']} | Title: {r['title']}")
        
    print("\n" + "="*70)
    print("FINAL CONSOLIDATED DATASET SHARES")
    print("="*70)
    print(f"Final Clean Positives: {len(final_pos)}")
    print(f"Final Clean Negatives: {len(final_neg)}")
    print(f"Final Clean Ratio: {len(final_neg)/len(final_pos):.2f}:1")
    print("Saved cancer_pm_dataset.json")

if __name__ == "__main__":
    main()
