"""
Provenance Control Analysis (Phase 2 + 3)
==========================================
Phase 2: Run frozen artifact-detection regexes on provenance-matched negatives
Phase 3: Score with frozen PubMedBERT checkpoint (no retraining)

Also analyzes:
- Crossref-fallback holdout positives separately (N=13 in holdout subset)
- Sub-group-B-style breakout (per-journal artifact rates)
- Full score distribution statistics

Pre-registered decision thresholds (set BEFORE seeing results):
- Header-concatenation rate threshold: >= 35% in provenance-matched negatives = "substantially elevated"
- Model FPR threshold: > 5% at threshold 0.10 = "spike"
- Conclusion (A): header rate >= 35% AND/OR FPR > 5%
- Conclusion (B): header rate < 35% AND FPR <= 5%
- Conclusion (C): N < 100 OR results mixed
"""
import json
import re
import os
import sys
import csv
import numpy as np
from datetime import datetime, timezone

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout.reconfigure(encoding='utf-8')

# =====================================================================
# Artifact detection regex (frozen from quantify_extraction_artifacts.py)
# =====================================================================
headers_list = [
    "MATERIALS AND METHODS", "PATIENTS AND METHODS", "BACKGROUND",
    "OBJECTIVE", "OBJECTIVES", "AIM", "AIMS", "INTRODUCTION",
    "PURPOSE", "METHOD", "METHODS", "RESULT", "RESULTS",
    "CONCLUSION", "CONCLUSIONS", "DISCUSSION", "SIGNIFICANCE", "DESIGN"
]
headers_pat = "|".join(headers_list)
header_regex = re.compile(
    rf'([.,!;?\-\s\(\[\]\)]*)\b({headers_pat})\b\s*[\]\)]*\s*:\s*([\(\[\]\)\-\s,;:]*)',
    re.IGNORECASE
)

def detect_glued_header(text):
    """
    Frozen glued-header detection from quantify_extraction_artifacts.py lines 82-109.
    Returns True if any glued header is found.
    """
    if not text:
        return False
    
    matches = list(header_regex.finditer(text))
    rebuilt_so_far = ""
    
    for m in matches:
        pre = m.group(1)
        start_idx = m.start()
        before_match = text[len(rebuilt_so_far):start_idx]
        rebuilt_so_far += before_match + m.group(0)
        
        is_start = (before_match.strip() == "" and rebuilt_so_far.replace(m.group(0), "").strip() == "")
        
        if not is_start:
            pre_stripped = pre.strip()
            if not pre_stripped or pre_stripped[-1] not in ['.', '!', '?']:
                return True
            elif len(pre) > 0 and pre[0] in ['.', '!', '?'] and not pre.endswith(' '):
                return True
    
    return False

def analyze_all_artifacts(text):
    """Full artifact analysis from quantify_extraction_artifacts.py."""
    import html
    import unicodedata
    
    if not text:
        return {'glued_header': False, 'stray_punctuation': False, 
                'whitespace_irregularity': False, 'unicode_entity': False}
    
    result = {'glued_header': False, 'stray_punctuation': False,
              'whitespace_irregularity': False, 'unicode_entity': False}
    
    # Unicode/entity checks
    if unicodedata.normalize('NFC', text) != text:
        result['unicode_entity'] = True
    if re.search(r'[\u00a0\u200b\u202f\u2007\xa0]', text):
        result['unicode_entity'] = True
    if html.unescape(text) != text:
        result['unicode_entity'] = True
    
    # Header artifacts
    matches = list(header_regex.finditer(text))
    rebuilt_so_far = ""
    
    for m in matches:
        pre = m.group(1)
        post = m.group(3)
        start_idx = m.start()
        end_idx = m.end()
        before_match = text[len(rebuilt_so_far):start_idx]
        rebuilt_so_far += before_match + m.group(0)
        
        is_start = (before_match.strip() == "" and rebuilt_so_far.replace(m.group(0), "").strip() == "")
        
        if not is_start:
            pre_stripped = pre.strip()
            if not pre_stripped or pre_stripped[-1] not in ['.', '!', '?']:
                result['glued_header'] = True
            elif len(pre) > 0 and pre[0] in ['.', '!', '?'] and not pre.endswith(' '):
                result['glued_header'] = True
            
            if any(c in pre for c in '([)]-,-;:'):
                result['stray_punctuation'] = True
        else:
            if pre.strip():
                result['stray_punctuation'] = True
        
        if any(c in post for c in '([)]-,-;:'):
            result['stray_punctuation'] = True
        
        if len(re.findall(r'\s{2,}', pre)) > 0 or len(re.findall(r'\s{2,}', post)) > 0:
            result['whitespace_irregularity'] = True
        if post == "" and end_idx < len(text):
            result['whitespace_irregularity'] = True
    
    return result

# =====================================================================
# Publisher mapping (same as evaluate_all_models.py)
# =====================================================================
journal_to_publisher = {}
rwdb_path = os.path.join(project_root, 'data', 'raw', 'rwdb', 'retraction_watch.csv')
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
    elif 'spandidos' in pub.lower():
        pub = 'Spandidos'
    elif pub not in ['Hindawi', 'Spandidos']:
        pub = 'Pooled Others'
    return pub


def main():
    print("=" * 80)
    print("PROVENANCE CONTROL ANALYSIS (Phase 2 + 3)")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)
    
    # =====================================================================
    # Load data
    # =====================================================================
    print("\nLoading data...")
    
    # Provenance-matched negatives
    prov_path = os.path.join(project_root, 'data', 'final', 'provenance_matched_negatives.json')
    if not os.path.exists(prov_path):
        print(f"ERROR: Provenance-matched negatives not found at {prov_path}")
        print("Run construct_provenance_matched_negatives.py first.")
        sys.exit(1)
    
    with open(prov_path, 'r', encoding='utf-8') as f:
        prov_neg = json.load(f)
    print(f"Provenance-matched negatives: N = {len(prov_neg)}")
    
    # Holdout data (for comparison baselines)
    with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_holdout.json'), 'r', encoding='utf-8') as f:
        holdout = json.load(f)
    
    holdout_pos = [r for r in holdout if r['label'] == 1]
    holdout_neg = [r for r in holdout if r['label'] == 0]
    
    # Separate Crossref vs PubMed positives
    holdout_pos_crossref = [r for r in holdout_pos if r.get('abstract_source') == 'Crossref']
    holdout_pos_pubmed = [r for r in holdout_pos if r.get('abstract_source') == 'PubMed']
    
    # Also load val for the Sub-group B comparison
    with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_val.json'), 'r', encoding='utf-8') as f:
        val = json.load(f)
    val_hindawi_pos = [r for r in val if r['label'] == 1 and get_publisher(r) == 'Hindawi']
    val_hindawi_neg = [r for r in val if r['label'] == 0 and get_publisher(r) == 'Hindawi']
    
    print(f"Holdout positives: {len(holdout_pos)} (PubMed: {len(holdout_pos_pubmed)}, Crossref: {len(holdout_pos_crossref)})")
    print(f"Holdout negatives: {len(holdout_neg)}")
    print(f"Val Hindawi positives: {len(val_hindawi_pos)}")
    print(f"Val Hindawi negatives: {len(val_hindawi_neg)}")
    
    # =====================================================================
    # PHASE 2: Artifact Regex Analysis
    # =====================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: ARTIFACT REGEX ANALYSIS")
    print("=" * 80)
    
    populations = {
        'Provenance-Matched Neg': prov_neg,
        'Original Holdout Neg (Hindawi)': holdout_neg,
        'Holdout Pos (PubMed)': holdout_pos_pubmed,
        'Holdout Pos (Crossref)': holdout_pos_crossref,
        'Val Hindawi Pos (all)': val_hindawi_pos,
        'Val Hindawi Neg': val_hindawi_neg,
    }
    
    artifact_results = {}
    
    for pop_name, records in populations.items():
        if not records:
            artifact_results[pop_name] = {'N': 0}
            continue
        
        N = len(records)
        counts = {'glued_header': 0, 'stray_punctuation': 0, 
                  'whitespace_irregularity': 0, 'unicode_entity': 0, 'any_artifact': 0}
        
        for r in records:
            abstract = r.get('abstract', '') or ''
            title = r.get('title', '') or ''
            text = title + " " + abstract
            
            arts = analyze_all_artifacts(text)
            if arts['glued_header']:
                counts['glued_header'] += 1
            if arts['stray_punctuation']:
                counts['stray_punctuation'] += 1
            if arts['whitespace_irregularity']:
                counts['whitespace_irregularity'] += 1
            if arts['unicode_entity']:
                counts['unicode_entity'] += 1
            if any(arts.values()):
                counts['any_artifact'] += 1
        
        artifact_results[pop_name] = {
            'N': N,
            'glued_header': counts['glued_header'],
            'glued_header_rate': counts['glued_header'] / N,
            'stray_punctuation': counts['stray_punctuation'],
            'stray_punc_rate': counts['stray_punctuation'] / N,
            'whitespace': counts['whitespace_irregularity'],
            'whitespace_rate': counts['whitespace_irregularity'] / N,
            'unicode': counts['unicode_entity'],
            'unicode_rate': counts['unicode_entity'] / N,
            'any_artifact': counts['any_artifact'],
            'any_rate': counts['any_artifact'] / N,
        }
    
    # Print Phase 2 results table
    print(f"\n{'Population':<35} | {'N':>5} | {'Glued Hdr':>12} | {'Stray Punc':>12} | {'Whitespace':>12} | {'Unicode':>12} | {'Any':>12}")
    print("-" * 120)
    for pop_name, res in artifact_results.items():
        if res['N'] == 0:
            print(f"{pop_name:<35} | {'N/A':>5} |")
            continue
        n = res['N']
        print(f"{pop_name:<35} | {n:>5} | "
              f"{res['glued_header']:>4}/{n} ({res['glued_header_rate']*100:5.1f}%) | "
              f"{res['stray_punctuation']:>4}/{n} ({res['stray_punc_rate']*100:5.1f}%) | "
              f"{res['whitespace']:>4}/{n} ({res['whitespace_rate']*100:5.1f}%) | "
              f"{res['unicode']:>4}/{n} ({res['unicode_rate']*100:5.1f}%) | "
              f"{res['any_artifact']:>4}/{n} ({res['any_rate']*100:5.1f}%)")
    
    # Sub-group-B-style breakout: per-journal artifact rates in provenance-matched negatives
    print(f"\n{'='*80}")
    print("SUB-GROUP-B-STYLE BREAKOUT: Per-Journal Glued Header Rates (Provenance-Matched)")
    print(f"{'='*80}")
    
    by_journal = defaultdict(list)
    for r in prov_neg:
        j = r.get('journal', '') or r.get('target_journal', '')
        abstract = r.get('abstract', '') or ''
        title = r.get('title', '') or ''
        has_glued = detect_glued_header(title + " " + abstract)
        by_journal[j].append(has_glued)
    
    print(f"\n{'Journal':<40} | {'N':>5} | {'Glued Hdr':>15}")
    print("-" * 70)
    for j, vals in sorted(by_journal.items(), key=lambda x: -len(x[1])):
        n = len(vals)
        rate = sum(vals) / n if n > 0 else 0
        flag = " *** ELEVATED ***" if rate >= 0.35 else ""
        print(f"{j[:40]:<40} | {n:>5} | {sum(vals):>4}/{n} ({rate*100:5.1f}%){flag}")
    
    # =====================================================================
    # PHASE 3: Model Scoring
    # =====================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: MODEL SCORING (Frozen PubMedBERT epoch 1)")
    print("=" * 80)
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModel
    
    MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    FROZEN_THRESHOLD = 0.10  # From evaluation results
    
    class AbstractDataset(Dataset):
        def __init__(self, records, tokenizer, max_length=512):
            self.records = records
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.records)
        
        def __getitem__(self, idx):
            rec = self.records[idx]
            title = rec.get("title", "") or ""
            abstract = rec.get("abstract", "") or ""
            text = title + " " + abstract
            
            encoding = self.tokenizer(
                text, truncation=True, max_length=self.max_length,
                padding="max_length", return_tensors="pt"
            )
            
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
    
    class BERTClassifier(nn.Module):
        def __init__(self, model_name=MODEL_NAME):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.encoder.config.hidden_size, 1)
            )
        
        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(cls_embedding)
            return logits
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model_path = os.path.join(project_root, 'models', 'pubmedbert_finetuned_best.pt')
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BERTClassifier(MODEL_NAME).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")
    
    def score_records(records, desc=""):
        """Score a set of records and return probability array."""
        if not records:
            return np.array([])
        
        dataset = AbstractDataset(records, tokenizer, MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
        
        all_probs = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(input_ids, mask)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                all_probs.extend(probs)
                
                if (i + 1) % 10 == 0:
                    print(f"  {desc}: Batch {i+1}/{len(loader)}")
        
        return np.array(all_probs)
    
    # Score all populations
    scoring_populations = {
        'Provenance-Matched Neg': prov_neg,
        'Original Holdout Neg (Hindawi)': holdout_neg,
        'Holdout Pos (PubMed)': holdout_pos_pubmed,
        'Holdout Pos (Crossref)': holdout_pos_crossref,
    }
    
    # Cache per-record scores from first pass for per-journal breakdown
    all_cached_scores = {}
    score_results = {}
    
    for pop_name, records in scoring_populations.items():
        if not records:
            continue
        print(f"\nScoring {pop_name} (N={len(records)})...")
        probs = score_records(records, desc=pop_name)
        all_cached_scores[pop_name] = probs
        
        result = {
            'N': len(records),
            'min': float(np.min(probs)),
            'Q1': float(np.percentile(probs, 25)),
            'median': float(np.median(probs)),
            'Q3': float(np.percentile(probs, 75)),
            'max': float(np.max(probs)),
            'mean': float(np.mean(probs)),
            'std': float(np.std(probs)),
            'predicted_positive_at_010': int(np.sum(probs >= FROZEN_THRESHOLD)),
            'fpr_at_010': float(np.mean(probs >= FROZEN_THRESHOLD)),
        }
        score_results[pop_name] = result
        
        print(f"  Min={result['min']:.6f}, Q1={result['Q1']:.6f}, Median={result['median']:.6f}, "
              f"Q3={result['Q3']:.6f}, Max={result['max']:.6f}")
        print(f"  FPR at threshold 0.10: {result['predicted_positive_at_010']}/{result['N']} "
              f"({result['fpr_at_010']*100:.2f}%)")
    
    # Per-journal score breakdown using cached scores (no re-scoring needed)
    print(f"\n{'='*80}")
    print("PER-JOURNAL SCORE BREAKDOWN (Provenance-Matched Negatives)")
    print(f"{'='*80}")
    
    prov_scores = all_cached_scores.get('Provenance-Matched Neg', np.array([]))
    
    journal_scores = defaultdict(list)
    for r, s in zip(prov_neg, prov_scores):
        j = r.get('journal', '') or r.get('target_journal', '')
        journal_scores[j].append(s)
    
    print(f"\n{'Journal':<40} | {'N':>5} | {'Median':>8} | {'Max':>8} | {'FP@0.10':>10}")
    print("-" * 80)
    for j, scores in sorted(journal_scores.items(), key=lambda x: -len(x[1])):
        scores_arr = np.array(scores)
        n = len(scores)
        fp = int(np.sum(scores_arr >= FROZEN_THRESHOLD))
        print(f"{j[:40]:<40} | {n:>5} | {np.median(scores_arr):.6f} | {np.max(scores_arr):.6f} | {fp:>3}/{n} ({fp/n*100:.1f}%)")
    
    # =====================================================================
    # VERDICT: Apply pre-registered decision criteria
    # =====================================================================
    print("\n" + "=" * 80)
    print("VERDICT: APPLYING PRE-REGISTERED DECISION CRITERIA")
    print("=" * 80)
    
    prov_glued_rate = artifact_results['Provenance-Matched Neg']['glued_header_rate']
    orig_neg_glued_rate = artifact_results['Original Holdout Neg (Hindawi)']['glued_header_rate']
    prov_fpr = score_results.get('Provenance-Matched Neg', {}).get('fpr_at_010', 0)
    orig_neg_fpr = score_results.get('Original Holdout Neg (Hindawi)', {}).get('fpr_at_010', 0)
    prov_n = len(prov_neg)
    
    print(f"\n--- Pre-registered thresholds ---")
    print(f"  Glued header rate threshold: >= 35% = 'substantially elevated'")
    print(f"  Model FPR threshold: > 5% at threshold 0.10 = 'spike'")
    print(f"  Minimum N for powered conclusion: >= 100")
    
    print(f"\n--- Observed values ---")
    print(f"  Provenance-matched N: {prov_n}")
    print(f"  Provenance-matched glued header rate: {prov_glued_rate*100:.2f}%")
    print(f"  Original holdout neg glued header rate: {orig_neg_glued_rate*100:.2f}%")
    print(f"  Provenance-matched FPR at 0.10: {prov_fpr*100:.2f}%")
    print(f"  Original holdout neg FPR at 0.10: {orig_neg_fpr*100:.2f}%")
    
    # Check subgroup effects
    subgroup_elevated = False
    for j, vals in by_journal.items():
        if len(vals) >= 10 and sum(vals)/len(vals) >= 0.35:
            subgroup_elevated = True
            print(f"  WARNING: Journal '{j}' has elevated glued header rate: {sum(vals)}/{len(vals)} ({sum(vals)/len(vals)*100:.1f}%)")
    
    journal_fpr_elevated = False
    for j, scores in journal_scores.items():
        scores_arr = np.array(scores)
        if len(scores) >= 10 and np.mean(scores_arr >= FROZEN_THRESHOLD) > 0.05:
            journal_fpr_elevated = True
            fp = int(np.sum(scores_arr >= FROZEN_THRESHOLD))
            print(f"  WARNING: Journal '{j}' has elevated FPR: {fp}/{len(scores)} ({fp/len(scores)*100:.1f}%)")
    
    print(f"\n--- Decision ---")
    
    if prov_n < 100:
        conclusion = 'C'
        reason = f"N={prov_n} < 100: underpowered to draw conclusion"
    elif prov_glued_rate >= 0.35 or prov_fpr > 0.05:
        conclusion = 'A'
        reasons = []
        if prov_glued_rate >= 0.35:
            reasons.append(f"glued header rate {prov_glued_rate*100:.2f}% >= 35%")
        if prov_fpr > 0.05:
            reasons.append(f"FPR {prov_fpr*100:.2f}% > 5%")
        reason = " AND ".join(reasons)
    elif subgroup_elevated or journal_fpr_elevated:
        conclusion = 'A'  # subgroup effect counts
        reason = "Aggregate below threshold but subgroup effect detected (see warnings above)"
    else:
        conclusion = 'B'
        reason = (f"glued header rate {prov_glued_rate*100:.2f}% < 35% AND "
                  f"FPR {prov_fpr*100:.2f}% <= 5% AND no subgroup effects detected")
    
    print(f"  CONCLUSION: ({conclusion})")
    print(f"  REASON: {reason}")
    
    # =====================================================================
    # SUMMARY TABLE
    # =====================================================================
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS TABLE")
    print("=" * 80)
    
    print(f"\n{'Population':<35} | {'N':>5} | {'Glued Hdr%':>10} | {'Model Median':>12} | {'FPR@0.10':>10}")
    print("-" * 85)
    
    for pop_name in ['Provenance-Matched Neg', 'Original Holdout Neg (Hindawi)', 
                      'Holdout Pos (PubMed)', 'Holdout Pos (Crossref)']:
        art = artifact_results.get(pop_name, {})
        scr = score_results.get(pop_name, {})
        n = art.get('N', 0)
        gh = art.get('glued_header_rate', 0)
        med = scr.get('median', 0)
        fpr = scr.get('fpr_at_010', 0)
        
        if n == 0:
            continue
        
        # For positives, report TPR instead of FPR
        if 'Pos' in pop_name:
            rate_label = "TPR"
        else:
            rate_label = "FPR"
        
        print(f"{pop_name:<35} | {n:>5} | {gh*100:>9.2f}% | {med:>12.6f} | {fpr*100:>8.2f}% ({rate_label})")
    
    # =====================================================================
    # SAVE ALL RESULTS
    # =====================================================================
    all_results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'pre_registered_thresholds': {
            'glued_header_threshold': 0.35,
            'fpr_threshold': 0.05,
            'min_n': 100,
        },
        'artifact_results': {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))} 
                            for k, v in artifact_results.items()},
        'score_results': score_results,
        'per_journal_glued_rates': {j: {'N': len(vals), 'rate': sum(vals)/len(vals) if vals else 0} 
                                    for j, vals in by_journal.items()},
        'per_journal_scores': {j: {'N': len(s), 'median': float(np.median(s)), 'max': float(np.max(s)),
                                   'fpr': float(np.mean(np.array(s) >= FROZEN_THRESHOLD))}
                              for j, s in journal_scores.items()},
        'conclusion': conclusion,
        'conclusion_reason': reason,
    }
    
    out_path = os.path.join(project_root, 'models', 'provenance_control_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")


if __name__ == '__main__':
    main()
