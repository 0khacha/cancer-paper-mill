import json
import urllib.request
import xml.etree.ElementTree as ET

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
            root = ET.fromstring(xml_text)
            for article in root.findall('.//PubmedArticle'):
                pmid_el = article.find('.//PMID')
                pmid = pmid_el.text.strip() if pmid_el is not None else ""
                if not pmid:
                    continue
                ab_parts = []
                for ab_el in article.findall('.//AbstractText'):
                    if ab_el.text:
                        label = ab_el.attrib.get('Label')
                        if label:
                            ab_parts.append(f"{label}: {ab_el.text.strip()}")
                        else:
                            ab_parts.append(ab_el.text.strip())
                abstract = " ".join(ab_parts) if ab_parts else ""
                results[pmid] = abstract
    except Exception as e:
        print(f"Batch fetch error: {e}")
    return results

def main():
    with open(r"C:\Users\moham\.gemini\antigravity\brain\c22a0783-bc26-4f1e-820b-3c6515e0eaa8\scratch\task0_sample.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    negatives = [s for s in samples if s['label'] == 0]
    pmids = [n['pmid'] for n in negatives]
    
    fetched = fetch_pubmed_abstracts_batch(pmids)
    
    identical_count = 0
    diff_count = 0
    
    for n in negatives:
        pmid = n['pmid']
        old_abstract = n.get('abstract', '')
        new_abstract = fetched.get(pmid, '')
        
        if old_abstract == new_abstract:
            identical_count += 1
        else:
            diff_count += 1
            print(f"--- DIFF for {pmid} ---")
            print(f"OLD: {repr(old_abstract[:100])}...")
            print(f"NEW: {repr(new_abstract[:100])}...")
            
    print(f"\nIdentical abstracts: {identical_count}")
    print(f"Different abstracts: {diff_count}")
    
    if diff_count == 0:
        print("CONCLUSION: Task 3 was a dud because the original negatives were ALREADY fetched using this exact pipeline or format.")

if __name__ == "__main__":
    main()
