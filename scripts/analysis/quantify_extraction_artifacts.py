import json
import re
import os
import html
import unicodedata
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Set up paths
original_splits = [
    ("train", "data/final/cancer_pm_train.json"),
    ("val", "data/final/cancer_pm_val.json"),
    ("test", "data/final/cancer_pm_test.json"),
    ("holdout", "data/final/cancer_pm_holdout.json")
]

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.utils.artifacts import header_regex

# Load publisher mapping logic
journal_to_publisher = {}
if os.path.exists('data/raw/rwdb/retraction_watch.csv'):
    import csv
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

if os.path.exists('data/final/journal_to_nlm.json'):
    with open('data/final/journal_to_nlm.json', 'r', encoding='utf-8') as f:
        j2nlm = json.load(f)
    nlm2raw = {v: k for k, v in j2nlm.items()}
else:
    nlm2raw = {}

def get_publisher(rec):
    j = rec.get('journal', '')
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if 'computational and mathematical methods in medicine' in raw_j.lower() or 'comput math methods med' in j.lower():
        pub = 'Hindawi'
    elif 'spandidos' in pub.lower():
        pub = 'Spandidos'
    elif pub not in ['Hindawi', 'Spandidos']:
        pub = 'Pooled Others'
    return pub

# Function to detect artifact types and return descriptions
def analyze_artifacts(text):
    if not text:
        return []
        
    artifacts = []
    
    # 1. Unicode/Entity artifacts
    if unicodedata.normalize('NFC', text) != text:
        artifacts.append("unicode_nfc")
    if re.search(r'[\u00a0\u200b\u202f\u2007\xa0]', text):
        artifacts.append("nbsp")
    if html.unescape(text) != text:
        artifacts.append("html_entity")
        
    # 2. Header-based artifacts
    matches = list(header_regex.finditer(text))
    
    # Track rebuilt string state to determine if it is start of string
    rebuilt_so_far = ""
    
    for m in matches:
        pre = m.group(1)
        hdr = m.group(2).upper()
        post = m.group(3)
        
        start_idx = m.start()
        end_idx = m.end()
        before_match = text[len(rebuilt_so_far):start_idx]
        rebuilt_so_far += before_match + m.group(0)
        
        is_start = (before_match.strip() == "" and rebuilt_so_far.replace(m.group(0), "").strip() == "")
        
        if not is_start:
            # Check for glued headers
            # Glued means not preceded by sentence-ending punctuation (., !, ?) and proper spacing.
            # E.g. "pathway METHODS:" or "pathway.METHODS:"
            pre_stripped = pre.strip()
            # If the preceding punctuation doesn't end with a sentence-ending char
            if not pre_stripped or pre_stripped[-1] not in ['.', '!', '?']:
                artifacts.append("glued_header")
            # If there's no space after the sentence-ending char (e.g. "pathway.METHODS:")
            elif len(pre) > 0 and pre[0] in ['.', '!', '?'] and not pre.endswith(' '):
                artifacts.append("glued_header")
                
            # Check for stray punctuation before header
            # E.g. "( CONCLUSION:" or "[CONCLUSION]:" or ", CONCLUSION:"
            # If Group 1 has parens, brackets, commas, semicolons, hyphens
            if any(c in pre for c in '([)]-,-;:'):
                artifacts.append("stray_punctuation")
        else:
            # At start, if there's any preceding punctuation, it's stray
            if pre.strip():
                artifacts.append("stray_punctuation")
                
        # Check for stray punctuation after header
        # E.g. "CONCLUSION: ( We found" or "CONCLUSION: - We found"
        if any(c in post for c in '([)]-,-;:'):
            artifacts.append("stray_punctuation")
            
        # Check for whitespace irregularities
        # E.g. multiple spaces, tabs, or newlines inside Group 1 or Group 3, or missing space after colon
        if len(re.findall(r'\s{2,}', pre)) > 0 or len(re.findall(r'\s{2,}', post)) > 0:
            artifacts.append("whitespace_irregularity")
        # Missing space after colon (unless at the end of text)
        if post == "" and end_idx < len(text):
            artifacts.append("whitespace_irregularity")
            
    return list(set(artifacts))

def main():
    # Counters for statistics
    # Format: stats[group_key][artifact_type] = count
    # group_key: (label, source_pipeline, publisher)
    stats = {}
    totals = {}
    
    # Examples collector
    examples = {
        "glued_header": [],
        "stray_punctuation": [],
        "whitespace_irregularity": [],
        "unicode_entity": []
    }
    
    for split_name, fpath in original_splits:
        with open(fpath, "r", encoding="utf-8") as f:
            records = json.load(f)
            
        for r in records:
            title = r.get("title", "") or ""
            abstract = r.get("abstract", "") or ""
            text = title + " " + abstract
            
            label = r.get("label", 0)
            source = r.get("abstract_source", "PubMed")
            pub = get_publisher(r)
            
            group_key = (label, source, pub)
            if group_key not in stats:
                stats[group_key] = {
                    "glued_header": 0,
                    "stray_punctuation": 0,
                    "whitespace_irregularity": 0,
                    "unicode_entity": 0,
                    "any_artifact": 0
                }
                totals[group_key] = 0
                
            totals[group_key] += 1
            
            # Analyze title and abstract
            t_arts = analyze_artifacts(title)
            a_arts = analyze_artifacts(abstract)
            all_arts = list(set(t_arts + a_arts))
            
            # Map specific categories to general deliverables
            has_glued = "glued_header" in all_arts
            has_stray = "stray_punctuation" in all_arts
            has_space = "whitespace_irregularity" in all_arts
            has_unicode = any(x in all_arts for x in ["unicode_nfc", "nbsp", "html_entity"])
            
            if has_glued:
                stats[group_key]["glued_header"] += 1
            if has_stray:
                stats[group_key]["stray_punctuation"] += 1
            if has_space:
                stats[group_key]["whitespace_irregularity"] += 1
            if has_unicode:
                stats[group_key]["unicode_entity"] += 1
            if has_glued or has_stray or has_space or has_unicode:
                stats[group_key]["any_artifact"] += 1
                
            # Collect examples
            from scripts.pipeline.normalize_abstracts import normalize_text_fields
            
            if has_glued and len(examples["glued_header"]) < 4:
                # Find before/after
                norm_ab, _ = normalize_text_fields(abstract)
                if norm_ab != abstract:
                    examples["glued_header"].append((abstract, norm_ab))
                    
            if has_stray and len(examples["stray_punctuation"]) < 4:
                norm_ab, _ = normalize_text_fields(abstract)
                if norm_ab != abstract:
                    examples["stray_punctuation"].append((abstract, norm_ab))
                    
            if has_space and len(examples["whitespace_irregularity"]) < 4:
                norm_ab, _ = normalize_text_fields(abstract)
                if norm_ab != abstract:
                    examples["whitespace_irregularity"].append((abstract, norm_ab))
                    
            if has_unicode and len(examples["unicode_entity"]) < 4:
                norm_ab, _ = normalize_text_fields(abstract)
                if norm_ab != abstract:
                    examples["unicode_entity"].append((abstract, norm_ab))
                    
    # Generate tables and save JSON
    print("\n" + "="*80)
    print("PRE-NORMALIZATION ARTIFACT RATES BY CLASS × SOURCE PIPELINE")
    print("="*80)
    print(f"{'Class':<10} | {'Source':<10} | {'N':<6} | {'Glued Hdr':<10} | {'Stray Punc':<10} | {'Whitespace':<10} | {'Unicode/Ent':<11} | {'Any Art':<8}")
    print("-" * 88)
    
    # Aggregate by Class x Source
    agg_class_source = {}
    for (lbl, src, pub), counts in stats.items():
        key = (lbl, src)
        if key not in agg_class_source:
            agg_class_source[key] = {k: 0 for k in counts.keys()}
            agg_class_source[key]["N"] = 0
        agg_class_source[key]["N"] += totals[(lbl, src, pub)]
        for k, v in counts.items():
            agg_class_source[key][k] += v
            
    for (lbl, src), counts in sorted(agg_class_source.items()):
        lbl_str = "Positive" if lbl == 1 else "Negative"
        n = counts["N"]
        print(f"{lbl_str:<10} | {src:<10} | {n:<6} | {counts['glued_header']/n*100:8.2f}% | {counts['stray_punctuation']/n*100:8.2f}% | {counts['whitespace_irregularity']/n*100:8.2f}% | {counts['unicode_entity']/n*100:9.2f}% | {counts['any_artifact']/n*100:6.2f}%")

    print("\n" + "="*80)
    print("PRE-NORMALIZATION ARTIFACT RATES BY CLASS × PUBLISHER")
    print("="*80)
    print(f"{'Class':<10} | {'Publisher':<13} | {'N':<6} | {'Glued Hdr':<10} | {'Stray Punc':<10} | {'Whitespace':<10} | {'Unicode/Ent':<11} | {'Any Art':<8}")
    print("-" * 91)
    
    # Aggregate by Class x Publisher
    agg_class_pub = {}
    for (lbl, src, pub), counts in stats.items():
        key = (lbl, pub)
        if key not in agg_class_pub:
            agg_class_pub[key] = {k: 0 for k in counts.keys()}
            agg_class_pub[key]["N"] = 0
        agg_class_pub[key]["N"] += totals[(lbl, src, pub)]
        for k, v in counts.items():
            agg_class_pub[key][k] += v
            
    for (lbl, pub), counts in sorted(agg_class_pub.items()):
        lbl_str = "Positive" if lbl == 1 else "Negative"
        n = counts["N"]
        print(f"{lbl_str:<10} | {pub:<13} | {n:<6} | {counts['glued_header']/n*100:8.2f}% | {counts['stray_punctuation']/n*100:8.2f}% | {counts['whitespace_irregularity']/n*100:8.2f}% | {counts['unicode_entity']/n*100:9.2f}% | {counts['any_artifact']/n*100:6.2f}%")

    # Save to disk
    results = {
        "class_source": {f"{lbl}_{src}": {k: v for k, v in counts.items()} for (lbl, src), counts in agg_class_source.items()},
        "class_publisher": {f"{lbl}_{pub}": {k: v for k, v in counts.items()} for (lbl, pub), counts in agg_class_pub.items()},
        "raw_stats": {f"{lbl}_{src}_{pub}": {"N": totals[(lbl,src,pub)], "counts": counts} for (lbl,src,pub), counts in stats.items()},
        "examples": examples
    }
    
    os.makedirs("models", exist_ok=True)
    with open("models/pre_normalization_artifacts.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved artifact rates to models/pre_normalization_artifacts.json")

if __name__ == "__main__":
    main()
