import json
import os

def main():
    with open(r"C:\Users\moham\.gemini\antigravity\brain\c22a0783-bc26-4f1e-820b-3c6515e0eaa8\scratch\task0_sample.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    positives = [s for s in samples if s['label'] == 1]
    negatives = [s for s in samples if s['label'] == 0]
    
    print("="*60)
    print("TASK 1: RAW REPR() COMPARISON (FIRST 500 CHARS)")
    print("="*60)
    
    print("\n--- POSITIVES ---")
    for i, p in enumerate(positives[:5]):
        text = (p.get('title', '') + " " + p.get('abstract', ''))
        print(f"P{i+1}: {repr(text[:500])}")
        
    print("\n--- NEGATIVES ---")
    for i, n in enumerate(negatives[:5]):
        text = (n.get('title', '') + " " + n.get('abstract', ''))
        print(f"N{i+1}: {repr(text[:500])}")
        
    print("\n" + "="*60)
    print("TASK 2: STRUCTURAL KEY DIFF")
    print("="*60)
    
    pos_keys = set()
    for p in positives:
        pos_keys.update(p.keys())
        
    neg_keys = set()
    for n in negatives:
        neg_keys.update(n.keys())
        
    print("Keys ONLY in Positives:", pos_keys - neg_keys)
    print("Keys ONLY in Negatives:", neg_keys - pos_keys)
    print("Keys in BOTH:", pos_keys.intersection(neg_keys))
    
    # Check for empty fields
    print("\nCheck missing/empty values in common fields:")
    common_fields = ['doi', 'pmid', 'title', 'abstract', 'journal', 'year', 'abstract_source', 'oncology_verified']
    
    def check_empty(group):
        empty_counts = {k: 0 for k in common_fields}
        for rec in group:
            for k in common_fields:
                if not rec.get(k):
                    empty_counts[k] += 1
        return empty_counts
        
    print("Positives empty counts:", check_empty(positives))
    print("Negatives empty counts:", check_empty(negatives))

if __name__ == "__main__":
    main()
