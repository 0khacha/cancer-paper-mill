import json
import csv
import sys
import os
import re

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(sys.maxsize)

def get_3grams(text):
    text_clean = re.sub(r'\s+', ' ', text.lower().strip())
    if len(text_clean) < 3:
        return set()
    return set(text_clean[i:i+3] for i in range(len(text_clean) - 2))

def load_journal_to_publisher():
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
    return journal_to_publisher

def get_publisher(rec, journal_to_publisher, nlm2raw):
    j = rec['journal']
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, 'Direct PubMed')
    if 'computational and mathematical methods in medicine' in raw_j.lower() or 'comput math methods med' in j.lower():
        pub = 'Hindawi'
    elif 'biomed research international' in raw_j.lower() or 'biomed res int' in j.lower():
        pub = 'Hindawi'
    return pub

def get_authors(doi, pmid, pos_metadata):
    if doi in pos_metadata:
        auth_str = pos_metadata[doi].get('Author', '')
    elif pmid in pos_metadata:
        auth_str = pos_metadata[pmid].get('Author', '')
    else:
        return set()
    return set(a.strip().lower() for a in auth_str.split(';') if len(a.strip()) > 3)

def main():
    print("Loading data...")
    with open("data/final/cancer_pm_train.json", "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open("data/final/cancer_pm_val.json", "r", encoding="utf-8") as f:
        val_records = json.load(f)
    with open("data/final/cancer_pm_holdout.json", "r", encoding="utf-8") as f:
        holdout_records = json.load(f)
        
    with open('data/final/journal_to_nlm.json', 'r', encoding='utf-8') as f:
        j2nlm = json.load(f)
    nlm2raw = {v: k for k, v in j2nlm.items()}
    
    journal_to_publisher = load_journal_to_publisher()
    
    pos_metadata = {}
    with open('data/raw/rwdb/retraction_watch.csv', 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doi = row.get('OriginalPaperDOI', '').strip().lower()
            pmid = row.get('OriginalPaperPubMedID', '').strip()
            if doi:
                pos_metadata[doi] = row
            if pmid:
                pos_metadata[pmid] = row
            
    # Non-Hindawi training pool (Train + Val)
    pool_recs = train_records + val_records
    non_hindawi_pos = [r for r in pool_recs if r["label"] == 1 and get_publisher(r, journal_to_publisher, nlm2raw) != 'Hindawi']
    
    # Hindawi Holdout Positives
    holdout_hindawi_pos = [r for r in holdout_records if r["label"] == 1]
    
    print(f"Non-Hindawi Training Positives: {len(non_hindawi_pos)}")
    print(f"Hindawi Holdout Positives: {len(holdout_hindawi_pos)}")
    
    # Pre-compute authors and 3-grams for the training pool
    pool_data = []
    for r in non_hindawi_pos:
        doi = str(r.get('doi', '')).strip().lower()
        pmid = str(r.get('pmid', '')).strip()
        authors = get_authors(doi, pmid, pos_metadata)
        grams = get_3grams(r['title'])
        pool_data.append({'authors': authors, 'grams': grams, 'rec': r})
        
    author_overlap_count = 0
    title_overlap_count = 0
    
    for h_rec in holdout_hindawi_pos:
        doi = str(h_rec.get('doi', '')).strip().lower()
        pmid = str(h_rec.get('pmid', '')).strip()
        h_authors = get_authors(doi, pmid, pos_metadata)
        h_grams = get_3grams(h_rec['title'])
        
        has_author_overlap = False
        has_title_overlap = False
        
        for p_data in pool_data:
            # Check author overlap (>= 2 shared)
            if len(h_authors & p_data['authors']) >= 2:
                has_author_overlap = True
                
            # Check title overlap (>= 0.70 jaccard)
            t_grams = p_data['grams']
            if h_grams and t_grams:
                inter = len(h_grams & t_grams)
                union = len(h_grams | t_grams)
                jac = inter / union if union > 0 else 0
                if jac >= 0.70:
                    has_title_overlap = True
                    
        if has_author_overlap:
            author_overlap_count += 1
        if has_title_overlap:
            title_overlap_count += 1
            
    total = len(holdout_hindawi_pos)
    print(f"\nRESULTS:")
    print(f"Author Overlap (>= 2 shared authors): {author_overlap_count} / {total} ({author_overlap_count/total*100:.2f}%)")
    print(f"Title Overlap (Jaccard >= 0.70): {title_overlap_count} / {total} ({title_overlap_count/total*100:.2f}%)")

if __name__ == '__main__':
    main()
