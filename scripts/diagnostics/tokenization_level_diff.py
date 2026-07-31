"""
Tokenization-Level Diff: Secondary Check 1
===========================================
Compares raw byte/whitespace sequences between RWDB-sourced positive abstracts
and keyword-query negative abstracts at the Hindawi stratum level.

Looks for systematic differences in:
1. Whitespace runs (consecutive spaces, tabs, newlines)
2. Non-breaking spaces (U+00A0, U+200B, U+202F, U+2007)
3. HTML entity remnants
4. Unicode normalization (NFC vs NFD) 
5. BPE token patterns (class-discriminative tokens invisible to word-level TF-IDF)
6. Section header formatting patterns at byte level
"""
import json
import re
import os
import sys
import html
import unicodedata
from collections import Counter, defaultdict

PYTHON = sys.executable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout.reconfigure(encoding='utf-8')

# Load publisher mapping
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
    elif 'spandidos' in pub.lower():
        pub = 'Spandidos'
    elif pub not in ['Hindawi', 'Spandidos']:
        pub = 'Pooled Others'
    return pub

def analyze_byte_patterns(text):
    """Analyze raw byte-level patterns in text."""
    results = {}
    
    # 1. Whitespace analysis
    results['total_chars'] = len(text)
    results['total_bytes'] = len(text.encode('utf-8'))
    
    # Count different whitespace types
    results['regular_spaces'] = text.count(' ')
    results['tabs'] = text.count('\t')
    results['newlines'] = text.count('\n')
    results['carriage_returns'] = text.count('\r')
    results['nbsp_00a0'] = text.count('\u00a0')
    results['zwsp_200b'] = text.count('\u200b')
    results['narrow_nbsp_202f'] = text.count('\u202f')
    results['figure_space_2007'] = text.count('\u2007')
    results['em_space_2003'] = text.count('\u2003')
    results['en_space_2002'] = text.count('\u2002')
    results['thin_space_2009'] = text.count('\u2009')
    
    # Total exotic whitespace
    exotic_ws_chars = '\u00a0\u200b\u202f\u2007\u2003\u2002\u2009\u200c\u200d\ufeff'
    results['any_exotic_whitespace'] = sum(text.count(c) for c in exotic_ws_chars)
    results['has_exotic_whitespace'] = results['any_exotic_whitespace'] > 0
    
    # 2. Consecutive whitespace runs
    ws_runs = re.findall(r' {2,}', text)
    results['double_space_runs'] = len(ws_runs)
    results['max_consecutive_spaces'] = max((len(r) for r in ws_runs), default=1)
    
    # 3. HTML entity remnants
    html_entities = re.findall(r'&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);', text)
    results['html_entity_count'] = len(html_entities)
    results['html_entities'] = html_entities[:5]  # sample
    
    # 4. Unicode normalization
    results['nfc_differs'] = unicodedata.normalize('NFC', text) != text
    results['nfkc_differs'] = unicodedata.normalize('NFKC', text) != text
    results['html_unescape_differs'] = html.unescape(text) != text
    
    # 5. Control characters (beyond standard whitespace)
    control_chars = [c for c in text if unicodedata.category(c).startswith('C') and c not in '\n\r\t']
    results['control_char_count'] = len(control_chars)
    results['control_chars'] = list(set(f'U+{ord(c):04X}' for c in control_chars))[:10]
    
    # 6. Multi-byte UTF-8 characters
    multibyte = [c for c in text if len(c.encode('utf-8')) > 1]
    results['multibyte_char_count'] = len(multibyte)
    # Categories of multibyte chars
    mb_cats = Counter(unicodedata.category(c) for c in multibyte)
    results['multibyte_categories'] = dict(mb_cats.most_common(10))
    
    # 7. Section header formatting patterns
    headers_list = [
        "MATERIALS AND METHODS", "PATIENTS AND METHODS", "BACKGROUND",
        "OBJECTIVE", "OBJECTIVES", "AIM", "AIMS", "INTRODUCTION",
        "PURPOSE", "METHOD", "METHODS", "RESULT", "RESULTS",
        "CONCLUSION", "CONCLUSIONS", "DISCUSSION", "SIGNIFICANCE", "DESIGN"
    ]
    headers_pat = "|".join(headers_list)
    
    # Look for different header formats
    header_patterns = {
        'newline_before_header': len(re.findall(rf'\n\s*\b({headers_pat})\b\s*:', text, re.IGNORECASE)),
        'period_space_header': len(re.findall(rf'\.\s+\b({headers_pat})\b\s*:', text, re.IGNORECASE)),
        'period_no_space_header': len(re.findall(rf'\.\b({headers_pat})\b\s*:', text, re.IGNORECASE)),
        'word_space_header': len(re.findall(rf'[a-z]\s+\b({headers_pat})\b\s*:', text, re.IGNORECASE)),
        'word_no_space_header': len(re.findall(rf'[a-z]\b({headers_pat})\b\s*:', text, re.IGNORECASE)),
        'start_of_string_header': 1 if re.match(rf'\s*\b({headers_pat})\b\s*:', text, re.IGNORECASE) else 0,
    }
    results['header_patterns'] = header_patterns
    
    # 8. Trailing/leading whitespace
    results['has_leading_whitespace'] = text != text.lstrip()
    results['has_trailing_whitespace'] = text != text.rstrip()
    
    # 9. Special punctuation
    results['em_dashes'] = text.count('\u2014') + text.count('\u2013')
    results['smart_quotes'] = sum(text.count(c) for c in '\u201c\u201d\u2018\u2019')
    results['degree_symbols'] = text.count('\u00b0')
    results['micro_signs'] = text.count('\u00b5')
    results['superscripts'] = sum(text.count(c) for c in '\u00b2\u00b3\u2070\u2071\u2074\u2075\u2076\u2077\u2078\u2079')
    results['subscripts'] = sum(text.count(c) for c in '\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089')
    
    return results

def main():
    print("=" * 80)
    print("TOKENIZATION-LEVEL DIFF: SECONDARY CHECK 1")
    print("=" * 80)
    
    # Load all splits (pre-normalization, original text)
    all_records = []
    for split_name in ['train', 'val', 'test', 'holdout']:
        fpath = os.path.join(project_root, 'data', 'final', f'cancer_pm_{split_name}.json')
        with open(fpath, 'r', encoding='utf-8') as f:
            recs = json.load(f)
            for r in recs:
                r['_split'] = split_name
            all_records.extend(recs)
    
    print(f"Loaded {len(all_records)} total records across all splits")
    
    # Filter to Hindawi only
    hindawi_pos = [r for r in all_records if get_publisher(r) == 'Hindawi' and r['label'] == 1]
    hindawi_neg = [r for r in all_records if get_publisher(r) == 'Hindawi' and r['label'] == 0]
    
    # Also separate by abstract_source
    hindawi_pos_pubmed = [r for r in hindawi_pos if r.get('abstract_source') == 'PubMed']
    hindawi_pos_crossref = [r for r in hindawi_pos if r.get('abstract_source') == 'Crossref']
    hindawi_neg_pubmed = [r for r in hindawi_neg if r.get('abstract_source') == 'PubMed']
    
    print(f"\nHindawi records:")
    print(f"  Positives: {len(hindawi_pos)} (PubMed: {len(hindawi_pos_pubmed)}, Crossref: {len(hindawi_pos_crossref)})")
    print(f"  Negatives: {len(hindawi_neg)} (PubMed: {len(hindawi_neg_pubmed)})")
    
    # =========================================================================
    # PART 1: Raw byte-level analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: RAW BYTE-LEVEL ANALYSIS (Hindawi abstracts only)")
    print("=" * 80)
    
    groups = {
        'Hindawi Pos (PubMed)': hindawi_pos_pubmed,
        'Hindawi Pos (Crossref)': hindawi_pos_crossref,
        'Hindawi Neg (PubMed)': hindawi_neg_pubmed,
    }
    
    aggregated = {}
    for group_name, records in groups.items():
        if not records:
            continue
        group_results = defaultdict(list)
        for r in records:
            abstract = r.get('abstract', '') or ''
            bp = analyze_byte_patterns(abstract)
            for k, v in bp.items():
                if isinstance(v, (int, float, bool)):
                    group_results[k].append(v)
        
        N = len(records)
        agg = {'N': N}
        for k, vals in group_results.items():
            if all(isinstance(v, bool) for v in vals):
                agg[k] = f"{sum(vals)}/{N} ({sum(vals)/N*100:.1f}%)"
                agg[f"{k}_rate"] = sum(vals) / N
            elif all(isinstance(v, (int, float)) for v in vals):
                import statistics
                agg[f"{k}_mean"] = statistics.mean(vals)
                agg[f"{k}_median"] = statistics.median(vals)
                agg[f"{k}_max"] = max(vals)
                agg[f"{k}_nonzero"] = f"{sum(1 for v in vals if v > 0)}/{N} ({sum(1 for v in vals if v > 0)/N*100:.1f}%)"
        aggregated[group_name] = agg
    
    # Print comparison table
    key_metrics = [
        ('has_exotic_whitespace', 'Has exotic whitespace', 'rate'),
        ('any_exotic_whitespace', 'Exotic WS char count', 'mean'),
        ('double_space_runs', 'Double-space runs', 'mean'),
        ('html_entity_count', 'HTML entity count', 'mean'),
        ('nfc_differs', 'NFC normalization differs', 'rate'),
        ('nfkc_differs', 'NFKC normalization differs', 'rate'),
        ('html_unescape_differs', 'HTML unescape differs', 'rate'),
        ('control_char_count', 'Control char count', 'mean'),
        ('multibyte_char_count', 'Multibyte char count', 'mean'),
        ('em_dashes', 'Em/en dashes', 'mean'),
        ('smart_quotes', 'Smart quotes', 'mean'),
        ('superscripts', 'Superscripts', 'mean'),
        ('subscripts', 'Subscripts', 'mean'),
    ]
    
    print(f"\n{'Metric':<30} | ", end="")
    for gn in groups.keys():
        if gn in aggregated:
            print(f"{gn:<25} | ", end="")
    print()
    print("-" * 120)
    
    for metric_key, metric_name, agg_type in key_metrics:
        print(f"{metric_name:<30} | ", end="")
        for gn in groups.keys():
            if gn not in aggregated:
                continue
            agg = aggregated[gn]
            if agg_type == 'rate':
                val = agg.get(f"{metric_key}", "N/A")
                print(f"{val:<25} | ", end="")
            elif agg_type == 'mean':
                mean_val = agg.get(f"{metric_key}_mean", 0)
                nz = agg.get(f"{metric_key}_nonzero", "0/0")
                print(f"{mean_val:.3f} (nonzero: {nz})"[:25].ljust(25) + " | ", end="")
        print()
    
    # =========================================================================
    # PART 2: BPE Token Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: BPE TOKEN ANALYSIS (PubMedBERT tokenizer)")
    print("=" * 80)
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        
        # Tokenize all Hindawi abstracts
        pos_token_counter = Counter()
        neg_token_counter = Counter()
        pos_token_docs = 0
        neg_token_docs = 0
        
        # Track token presence per document (for document frequency)
        pos_doc_freq = Counter()
        neg_doc_freq = Counter()
        
        for r in hindawi_pos_pubmed:
            abstract = r.get('abstract', '') or ''
            title = r.get('title', '') or ''
            text = title + " " + abstract
            tokens = tokenizer.tokenize(text)
            pos_token_counter.update(tokens)
            pos_doc_freq.update(set(tokens))
            pos_token_docs += 1
        
        for r in hindawi_neg_pubmed:
            abstract = r.get('abstract', '') or ''
            title = r.get('title', '') or ''
            text = title + " " + abstract
            tokens = tokenizer.tokenize(text)
            neg_token_counter.update(tokens)
            neg_doc_freq.update(set(tokens))
            neg_token_docs += 1
        
        print(f"\nTokenized {pos_token_docs} positive and {neg_token_docs} negative Hindawi abstracts (PubMed-sourced)")
        print(f"Positive total tokens: {sum(pos_token_counter.values())}")
        print(f"Negative total tokens: {sum(neg_token_counter.values())}")
        
        # Find tokens with highest class-discriminative power
        # Use normalized frequency ratio
        all_tokens = set(pos_token_counter.keys()) | set(neg_token_counter.keys())
        pos_total = sum(pos_token_counter.values())
        neg_total = sum(neg_token_counter.values())
        
        discriminative_tokens = []
        for token in all_tokens:
            pos_freq = pos_token_counter[token] / pos_total if pos_total else 0
            neg_freq = neg_token_counter[token] / neg_total if neg_total else 0
            
            # Require minimum document frequency
            pos_df = pos_doc_freq[token] / pos_token_docs if pos_token_docs else 0
            neg_df = neg_doc_freq[token] / neg_token_docs if neg_token_docs else 0
            
            if pos_doc_freq[token] + neg_doc_freq[token] < 10:
                continue
            
            # Log ratio (with smoothing)
            import math
            smooth = 1e-8
            ratio = math.log2((pos_freq + smooth) / (neg_freq + smooth))
            
            discriminative_tokens.append({
                'token': token,
                'pos_freq': pos_freq,
                'neg_freq': neg_freq,
                'pos_doc_freq': pos_df,
                'neg_doc_freq': neg_df,
                'log2_ratio': ratio,
                'pos_count': pos_token_counter[token],
                'neg_count': neg_token_counter[token],
            })
        
        # Sort by absolute log ratio
        discriminative_tokens.sort(key=lambda x: abs(x['log2_ratio']), reverse=True)
        
        print(f"\nTop 30 most class-discriminative BPE tokens (by |log2 frequency ratio|):")
        print(f"{'Token':<20} | {'Pos freq':>10} | {'Neg freq':>10} | {'Pos DF%':>8} | {'Neg DF%':>8} | {'log2 ratio':>10} | {'Direction':>10}")
        print("-" * 100)
        for t in discriminative_tokens[:30]:
            direction = "POS-biased" if t['log2_ratio'] > 0 else "NEG-biased"
            tok_display = repr(t['token'])[:18]
            print(f"{tok_display:<20} | {t['pos_freq']:>10.6f} | {t['neg_freq']:>10.6f} | {t['pos_doc_freq']*100:>7.1f}% | {t['neg_doc_freq']*100:>7.1f}% | {t['log2_ratio']:>+10.3f} | {direction:>10}")
        
        # Special focus: whitespace and formatting tokens
        print(f"\n{'='*80}")
        print("WHITESPACE & FORMATTING TOKENS:")
        print(f"{'='*80}")
        formatting_tokens = [t for t in discriminative_tokens if any(c in t['token'] for c in [' ', '\t', '\n', '##', '[', ']', ':', '.']) or t['token'].startswith('Ġ')]
        for t in formatting_tokens[:20]:
            direction = "POS-biased" if t['log2_ratio'] > 0 else "NEG-biased"
            print(f"  {repr(t['token']):<25} | pos_df={t['pos_doc_freq']*100:.1f}% | neg_df={t['neg_doc_freq']*100:.1f}% | log2={t['log2_ratio']:+.3f} | {direction}")
        
        # Also: compare Crossref vs PubMed positives at token level
        if hindawi_pos_crossref:
            print(f"\n{'='*80}")
            print(f"CROSSREF vs PUBMED POSITIVE TOKEN COMPARISON (Hindawi)")
            print(f"{'='*80}")
            crossref_counter = Counter()
            for r in hindawi_pos_crossref:
                abstract = r.get('abstract', '') or ''
                title = r.get('title', '') or ''
                text = title + " " + abstract
                tokens = tokenizer.tokenize(text)
                crossref_counter.update(tokens)
            
            cr_total = sum(crossref_counter.values())
            pm_total = pos_total
            
            # Average tokens per abstract
            avg_cr = cr_total / len(hindawi_pos_crossref) if hindawi_pos_crossref else 0
            avg_pm = pm_total / len(hindawi_pos_pubmed) if hindawi_pos_pubmed else 0
            print(f"Crossref avg tokens/abstract: {avg_cr:.1f}")
            print(f"PubMed avg tokens/abstract: {avg_pm:.1f}")
            
            # Check for section header tokens in Crossref vs PubMed
            header_tokens = ['background', 'methods', 'results', 'conclusion', 'objectives', 'aim', 'purpose', 'introduction']
            colon_tokens = [t for t in set(crossref_counter.keys()) | set(pos_token_counter.keys()) if ':' in t]
            
            print(f"\nSection header-related tokens:")
            for ht in header_tokens:
                cr_count = sum(v for t, v in crossref_counter.items() if ht in t.lower())
                pm_count = sum(v for t, v in pos_token_counter.items() if ht in t.lower())
                cr_per_doc = cr_count / len(hindawi_pos_crossref) if hindawi_pos_crossref else 0
                pm_per_doc = pm_count / len(hindawi_pos_pubmed) if hindawi_pos_pubmed else 0
                print(f"  '{ht}': Crossref={cr_per_doc:.2f}/doc, PubMed={pm_per_doc:.2f}/doc")
    
    except ImportError as e:
        print(f"WARNING: Could not load transformers for BPE analysis: {e}")
    except Exception as e:
        print(f"WARNING: BPE analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # PART 3: Header formatting deep-dive
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: SECTION HEADER FORMATTING DEEP-DIVE")
    print("=" * 80)
    
    headers_list = [
        "MATERIALS AND METHODS", "PATIENTS AND METHODS", "BACKGROUND",
        "OBJECTIVE", "OBJECTIVES", "AIM", "AIMS", "INTRODUCTION",
        "PURPOSE", "METHOD", "METHODS", "RESULT", "RESULTS",
        "CONCLUSION", "CONCLUSIONS", "DISCUSSION", "SIGNIFICANCE", "DESIGN"
    ]
    headers_pat = "|".join(headers_list)
    
    # Categorize header formatting more finely
    def categorize_header_contexts(text):
        """Find all section headers and classify the byte context around them."""
        contexts = []
        for m in re.finditer(rf'\b({headers_pat})\b\s*:', text, re.IGNORECASE):
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 5)
            
            before = text[start:m.start()]
            after = text[m.end():end]
            header = m.group(0)
            
            # Classify the transition
            if m.start() == 0 or text[:m.start()].strip() == '':
                ctx = 'START_OF_TEXT'
            elif before.rstrip()[-1:] in '.!?':
                # Check spacing
                stripped_before = before.rstrip()
                gap = before[len(stripped_before):]
                if '\n' in gap:
                    ctx = 'PERIOD_NEWLINE_HEADER'
                elif len(gap) >= 1:
                    ctx = 'PERIOD_SPACE_HEADER'
                else:
                    ctx = 'PERIOD_NO_SPACE_HEADER'
            elif before.rstrip()[-1:] in ',;':
                ctx = 'COMMA_HEADER'
            elif before.rstrip()[-1:] in ')]}':
                ctx = 'BRACKET_HEADER'
            elif before.rstrip()[-1:].isalpha():
                gap = len(before) - len(before.rstrip())
                if gap >= 1:
                    ctx = 'WORD_SPACE_HEADER'
                else:
                    ctx = 'WORD_GLUED_HEADER'
            elif before.rstrip()[-1:].isdigit():
                ctx = 'NUMBER_HEADER'
            else:
                ctx = 'OTHER'
            
            contexts.append(ctx)
        return contexts
    
    for group_name, records in groups.items():
        if not records:
            continue
        all_contexts = Counter()
        docs_with_headers = 0
        for r in records:
            abstract = r.get('abstract', '') or ''
            ctxs = categorize_header_contexts(abstract)
            if ctxs:
                docs_with_headers += 1
            all_contexts.update(ctxs)
        
        N = len(records)
        print(f"\n{group_name} (N={N}, docs with headers: {docs_with_headers}/{N} = {docs_with_headers/N*100:.1f}%):")
        for ctx, count in all_contexts.most_common():
            print(f"  {ctx:<30}: {count:>5} occurrences")
    
    # =========================================================================
    # PART 4: Exact byte comparison between a sample
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 4: SAMPLE BYTE-LEVEL COMPARISON (first 5 Hindawi pos vs neg with headers)")
    print("=" * 80)
    
    header_regex = re.compile(rf'\b({headers_pat})\b\s*:', re.IGNORECASE)
    
    for group_name, records in [('POSITIVE', hindawi_pos_pubmed), ('NEGATIVE', hindawi_neg_pubmed)]:
        print(f"\n--- {group_name} SAMPLES ---")
        shown = 0
        for r in records:
            abstract = r.get('abstract', '') or ''
            matches = list(header_regex.finditer(abstract))
            if matches and shown < 3:
                print(f"\nDOI: {r.get('doi', 'N/A')}")
                print(f"Source: {r.get('abstract_source', 'N/A')}")
                # Show 30-char context around each header
                for m in matches[:3]:
                    start = max(0, m.start() - 30)
                    end = min(len(abstract), m.end() + 10)
                    snippet = abstract[start:end]
                    # Show raw bytes
                    hex_bytes = ' '.join(f'{b:02x}' for b in snippet.encode('utf-8'))
                    print(f"  Context: {repr(snippet)}")
                    print(f"  Hex:     {hex_bytes[:100]}...")
                shown += 1
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: TOKENIZATION-LEVEL DIFF FINDINGS")
    print("=" * 80)
    
    # Compare key rates between pos and neg
    if 'Hindawi Pos (PubMed)' in aggregated and 'Hindawi Neg (PubMed)' in aggregated:
        pos_agg = aggregated['Hindawi Pos (PubMed)']
        neg_agg = aggregated['Hindawi Neg (PubMed)']
        
        summary_metrics = [
            ('has_exotic_whitespace_rate', 'Exotic whitespace rate'),
            ('nfc_differs_rate', 'NFC normalization differs rate'),
            ('html_unescape_differs_rate', 'HTML unescape differs rate'),
        ]
        
        print("\nKey byte-level differences (PubMed-sourced Hindawi only):")
        for key, name in summary_metrics:
            pos_val = pos_agg.get(key, 0)
            neg_val = neg_agg.get(key, 0)
            diff = pos_val - neg_val
            print(f"  {name}: Pos={pos_val:.3f}, Neg={neg_val:.3f}, Diff={diff:+.3f}")
    
    # Save results
    output = {
        'aggregated': {},
        'timestamp': __import__('datetime').datetime.now().isoformat(),
    }
    for gn, agg in aggregated.items():
        clean_agg = {}
        for k, v in agg.items():
            if isinstance(v, (int, float, str)):
                clean_agg[k] = v
        output['aggregated'][gn] = clean_agg
    
    out_path = os.path.join(project_root, 'models', 'tokenization_diff_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved detailed results to {out_path}")

if __name__ == '__main__':
    main()
