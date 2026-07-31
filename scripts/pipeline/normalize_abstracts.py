import json
import os
import re
import html
import unicodedata

# Set up paths
train_path = "data/final/cancer_pm_train.json"
val_path = "data/final/cancer_pm_val.json"
test_path = "data/final/cancer_pm_test.json"
holdout_path = "data/final/cancer_pm_holdout.json"

output_dir = "data/final"
audit_path = os.path.join(output_dir, "normalization_audit.json")

# Define header patterns
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

def normalize_text_fields(text):
    if not text:
        return "", {}
    
    logs = {}
    original = text
    
    # 1. Unicode NFC normalization
    norm_nfc = unicodedata.normalize('NFC', text)
    if norm_nfc != text:
        logs["unicode_nfc_changed"] = True
        text = norm_nfc
        
    # 2. Replace non-breaking spaces and other weird spaces with standard spaces
    # \u00a0 is NBSP, \u200b is zero-width space, \u202f is narrow NBSP, \u2007 is figure space
    nbsp_pattern = re.compile(r'[\u00a0\u200b\u202f\u2007\xa0]')
    nbsp_matches = len(nbsp_pattern.findall(text))
    if nbsp_matches > 0:
        logs["nbsp_replaced_count"] = nbsp_matches
        text = nbsp_pattern.sub(' ', text)
        
    # 3. HTML/XML entity decoding
    decoded = html.unescape(text)
    if decoded != text:
        logs["html_entities_decoded"] = True
        text = decoded
        
    # 4. Canonicalize section headers and strip stray punctuation around them
    last_idx = 0
    new_text = ""
    headers_found = []
    
    for m in header_regex.finditer(text):
        start_idx = m.start()
        end_idx = m.end()
        
        pre_punctuation = m.group(1)
        header_word = m.group(2).upper()
        post_punctuation = m.group(3)
        
        headers_found.append(header_word)
        
        # Rebuild preceding text
        before_match = text[last_idx:start_idx]
        new_text += before_match
        
        is_start = (new_text.strip() == "")
        
        if is_start:
            replacement = f"{header_word}: "
            new_text = ""
        else:
            rebuilt_clean = new_text.rstrip()
            # Strip trailing brackets, parens, commas, semicolons, hyphens
            while rebuilt_clean and rebuilt_clean[-1] in '([)]-,-;:':
                rebuilt_clean = rebuilt_clean[:-1].rstrip()
                
            if not rebuilt_clean:
                replacement = f"{header_word}: "
            else:
                last_char = rebuilt_clean[-1]
                if last_char not in ['.', '!', '?']:
                    rebuilt_clean += "."
                replacement = f"\n\n{header_word}: "
                new_text = rebuilt_clean
                
        new_text += replacement
        last_idx = end_idx
        
    new_text += text[last_idx:]
    
    if headers_found:
        logs["headers_canonicalized"] = headers_found
        
    # 5. Collapse multiple consecutive spaces (except \n\n before headers)
    temp_text = new_text.replace("\n\n", " [NEWLINE_MARKER] ")
    collapsed = re.sub(r'[ \t\r\f\v]+', ' ', temp_text) # Collapse spaces/tabs only
    # Also collapse multiple newlines if any are left
    collapsed = re.sub(r'\n+', '\n', collapsed)
    final_text = collapsed.replace(" [NEWLINE_MARKER] ", "\n\n").strip()
    
    # If final text differs from original, note it
    if final_text != original:
        logs["text_changed"] = True
        
    return final_text, logs

def process_file(fpath, out_fpath):
    print(f"Normalizing {fpath}...")
    with open(fpath, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    normalized_records = []
    file_audits = []
    
    for r in records:
        orig_title = r.get("title", "") or ""
        orig_abstract = r.get("abstract", "") or ""
        
        norm_title, title_logs = normalize_text_fields(orig_title)
        norm_abstract, abstract_logs = normalize_text_fields(orig_abstract)
        
        # Merge logs
        doc_audit = {
            "doi": r.get("doi", ""),
            "pmid": r.get("pmid", ""),
            "label": r.get("label", 0),
            "abstract_source": r.get("abstract_source", "PubMed"),
            "journal": r.get("journal", ""),
            "title_changed": title_logs.get("text_changed", False),
            "abstract_changed": abstract_logs.get("text_changed", False),
            "title_logs": title_logs,
            "abstract_logs": abstract_logs
        }
        
        # Check if anything changed
        has_changed = doc_audit["title_changed"] or doc_audit["abstract_changed"]
        doc_audit["changed"] = has_changed
        
        # Update record
        r_new = r.copy()
        r_new["title"] = norm_title
        r_new["abstract"] = norm_abstract
        
        normalized_records.append(r_new)
        file_audits.append(doc_audit)
        
    with open(out_fpath, "w", encoding="utf-8") as f:
        json.dump(normalized_records, f, indent=2, ensure_ascii=False)
        
    print(f"Saved normalized split to {out_fpath}")
    return file_audits

def main():
    splits = [
        ("train", train_path, os.path.join(output_dir, "cancer_pm_train_normalized.json")),
        ("val", val_path, os.path.join(output_dir, "cancer_pm_val_normalized.json")),
        ("test", test_path, os.path.join(output_dir, "cancer_pm_test_normalized.json")),
        ("holdout", holdout_path, os.path.join(output_dir, "cancer_pm_holdout_normalized.json"))
    ]
    
    all_audits = {}
    total_changed = 0
    total_docs = 0
    
    for name, in_p, out_p in splits:
        file_audits = process_file(in_p, out_p)
        all_audits[name] = file_audits
        
        changed_count = sum(1 for a in file_audits if a["changed"])
        total_changed += changed_count
        total_docs += len(file_audits)
        print(f"  Split '{name}': {changed_count} / {len(file_audits)} documents modified ({changed_count/len(file_audits)*100:.2f}%)")
        
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(all_audits, f, indent=2, ensure_ascii=False)
        
    print(f"Normalization complete. Audit log saved to {audit_path}")
    print(f"Total documents modified across all splits: {total_changed} / {total_docs} ({total_changed/total_docs*100:.2f}%)")

if __name__ == "__main__":
    main()
