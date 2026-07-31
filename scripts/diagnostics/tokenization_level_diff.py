import json
import re
import os
import unicodedata
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# We need the publisher mapping logic
import csv
journal_to_publisher = {}
rwdb_path = os.path.join(project_root, 'data', 'raw', 'rwdb', 'retraction_watch.csv')
if os.path.exists(rwdb_path):
    with open(rwdb_path, 'r', encoding='utf-8', errors='replace') as f:
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
    j = rec.get('journal', '')
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if 'computational and mathematical methods in medicine' in raw_j.lower() or 'comput math methods med' in j.lower():
        pub = 'Hindawi'
    elif 'biomed research international' in raw_j.lower() or 'biomed res int' in j.lower():
        pub = 'Hindawi'
    return pub

# Metrics calculation
def calculate_metrics(abstract):
    if not abstract:
        return None
    
    # 1. NFC diff
    is_nfc = (unicodedata.normalize('NFC', abstract) == abstract)
    non_nfc = int(not is_nfc)
    
    # 2. Invisible/non-breaking characters
    invisible_chars = ['\u00A0', '\u200B', '\u200C', '\u200D', '\uFEFF', '\u2009', '\u2028', '\u2007', '\u202F', '\u2003', '\u2002']
    has_invisible = int(any(c in abstract for c in invisible_chars))
    
    # 3. Whitespace run analysis
    ws_runs = len(re.findall(r'[ \t\n\r]{2,}', abstract))
    char_len = len(abstract)
    ws_run_per_1000 = (ws_runs / char_len * 1000) if char_len > 0 else 0
    
    # 4. Byte/char mismatch
    byte_len = len(abstract.encode('utf-8'))
    byte_char_ratio = byte_len / char_len if char_len > 0 else 1.0
    
    # 5. Reference/citation residue
    # look for stray [1], [1-3], [1,2], DOI strings, et al. truncated, etc.
    ref_residue_patterns = [
        r'\[\s*\d+\s*(?:,\s*\d+\s*)*\]',  # [1], [1, 2]
        r'\[\s*\d+\s*-\s*\d+\s*\]',       # [1-3]
        r'\bdoi\b\s*[:\s]\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+', # DOI string
        r'\bet\s+al\.',
        r'(?:\d+\.)+\d+\s+\[', # orphaned punctuation before bracket
    ]
    has_residue = int(any(re.search(p, abstract, re.IGNORECASE) for p in ref_residue_patterns))
    
    return {
        'non_nfc': non_nfc,
        'has_invisible': has_invisible,
        'ws_run_per_1000': ws_run_per_1000,
        'byte_char_ratio': byte_char_ratio,
        'has_residue': has_residue
    }

def bootstrap_ci_diff(arr1, arr2, func, n_resamples=2000):
    np.random.seed(42)
    diffs = []
    n1 = len(arr1)
    n2 = len(arr2)
    for _ in range(n_resamples):
        samp1 = np.random.choice(arr1, size=n1, replace=True)
        samp2 = np.random.choice(arr2, size=n2, replace=True)
        diffs.append(func(samp1) - func(samp2))
    return np.percentile(diffs, [2.5, 97.5])

def main():
    with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_holdout.json'), 'r', encoding='utf-8') as f:
        holdout = json.load(f)
    with open(os.path.join(project_root, 'data', 'final', 'provenance_matched_negatives.json'), 'r', encoding='utf-8') as f:
        prov = json.load(f)
        
    hindawi_pos = [r for r in holdout if get_publisher(r) == 'Hindawi' and r.get('label') == 1]
    hindawi_neg = [r for r in holdout if get_publisher(r) == 'Hindawi' and r.get('label') == 0]
    prov_neg = prov
    
    print(f"Positives: {len(hindawi_pos)}")
    print(f"Negatives: {len(hindawi_neg)}")
    print(f"Prov Negatives: {len(prov_neg)}")
    
    groups = {
        'Pos': hindawi_pos,
        'Neg': hindawi_neg,
        'ProvNeg': prov_neg
    }
    
    metrics_by_group = {k: [] for k in groups.keys()}
    
    # Pre-load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    
    for gname, records in groups.items():
        for r in records:
            abs_text = r.get('abstract', '')
            m = calculate_metrics(abs_text)
            if m:
                # 6. BPE tokens
                # We need word split count vs token count
                words = abs_text.split()
                word_count = len(words)
                tokens = tokenizer.tokenize(abs_text)
                token_count = len(tokens)
                bpe_ratio = token_count / word_count if word_count > 0 else 1.0
                m['bpe_ratio'] = bpe_ratio
                metrics_by_group[gname].append(m)
                
    df_pos = pd.DataFrame(metrics_by_group['Pos'])
    df_neg = pd.DataFrame(metrics_by_group['Neg'])
    df_prov = pd.DataFrame(metrics_by_group['ProvNeg'])
    
    results = []
    
    metrics_config = [
        ('non_nfc', 'rate'),
        ('has_invisible', 'rate'),
        ('ws_run_per_1000', 'continuous'),
        ('byte_char_ratio', 'continuous'),
        ('has_residue', 'rate'),
        ('bpe_ratio', 'continuous')
    ]
    
    for m, mtype in metrics_config:
        pos_vals = df_pos[m].values
        neg_vals = df_neg[m].values
        prov_vals = df_prov[m].values
        
        if mtype == 'rate':
            pos_est = np.mean(pos_vals)
            neg_est = np.mean(neg_vals)
            diff = pos_est - neg_est
            ci_pos_neg = bootstrap_ci_diff(pos_vals, neg_vals, np.mean)
            
            prov_est = np.mean(prov_vals)
            diff_prov = pos_est - prov_est
            ci_pos_prov = bootstrap_ci_diff(pos_vals, prov_vals, np.mean)
            
            results.append({
                'Metric': m,
                'Pos': f"{pos_est*100:.2f}%",
                'Neg': f"{neg_est*100:.2f}%",
                'ProvNeg': f"{prov_est*100:.2f}%",
                'Diff (Pos-Neg)': f"{diff*100:.2f}%",
                '95% CI (Pos-Neg)': f"[{ci_pos_neg[0]*100:.2f}%, {ci_pos_neg[1]*100:.2f}%]",
                'Diff (Pos-Prov)': f"{diff_prov*100:.2f}%",
                '95% CI (Pos-Prov)': f"[{ci_pos_prov[0]*100:.2f}%, {ci_pos_prov[1]*100:.2f}%]"
            })
        else:
            pos_med = np.median(pos_vals)
            pos_iqr = np.percentile(pos_vals, 75) - np.percentile(pos_vals, 25)
            
            neg_med = np.median(neg_vals)
            neg_iqr = np.percentile(neg_vals, 75) - np.percentile(neg_vals, 25)
            
            diff_med = pos_med - neg_med
            ci_pos_neg = bootstrap_ci_diff(pos_vals, neg_vals, np.median)
            
            prov_med = np.median(prov_vals)
            prov_iqr = np.percentile(prov_vals, 75) - np.percentile(prov_vals, 25)
            
            diff_med_prov = pos_med - prov_med
            ci_pos_prov = bootstrap_ci_diff(pos_vals, prov_vals, np.median)
            
            results.append({
                'Metric': m,
                'Pos': f"{pos_med:.4f} (IQR {pos_iqr:.4f})",
                'Neg': f"{neg_med:.4f} (IQR {neg_iqr:.4f})",
                'ProvNeg': f"{prov_med:.4f} (IQR {prov_iqr:.4f})",
                'Diff (Pos-Neg)': f"{diff_med:.4f}",
                '95% CI (Pos-Neg)': f"[{ci_pos_neg[0]:.4f}, {ci_pos_neg[1]:.4f}]",
                'Diff (Pos-Prov)': f"{diff_med_prov:.4f}",
                '95% CI (Pos-Prov)': f"[{ci_pos_prov[0]:.4f}, {ci_pos_prov[1]:.4f}]"
            })
            
    res_df = pd.DataFrame(results)
    print(res_df.to_string())

if __name__ == '__main__':
    main()
