import json
import re
import unicodedata

def aggressive_normalize(text):
    if not text:
        return ""
    
    # 1. Complete Unicode unification (NFKC)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Total lowercasing
    text = text.lower()
    
    # 3. Removal of structural headers
    headers = [
        "materials and methods", "patients and methods", "background", 
        "objective", "objectives", "aim", "aims", "introduction", 
        "purpose", "method", "methods", "result", "results", 
        "conclusion", "conclusions", "discussion", "significance", "design"
    ]
    headers_pat = "|".join(headers)
    
    # Remove headers that end with a colon or similar
    # e.g. "background:", "methods :", "[results]" (well, brackets removed later, but let's handle "background:")
    header_regex = re.compile(rf'\b({headers_pat})\b\s*:', re.IGNORECASE)
    text = header_regex.sub(' ', text)
    
    # Also handle some headers that might just be capitalized words before a colon, 
    # but since we already lowercased, we just catch the specific list.
    
    # 4. Replace punctuation with spaces to avoid gluing words together
    # Includes standard punctuation, hyphens, slashes, etc.
    text = re.sub(r'[.,;:!?\"\'\(\)\[\]\{\}\-_/\\|]', ' ', text)
    
    # 5. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    with open(r"C:\Users\moham\.gemini\antigravity\brain\c22a0783-bc26-4f1e-820b-3c6515e0eaa8\scratch\task0_sample.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    positives = [s for s in samples if s['label'] == 1]
    negatives = [s for s in samples if s['label'] == 0]
    
    print("="*60)
    print("TASK 5: AGGRESSIVE NORMALIZATION SANITY CHECK")
    print("="*60)
    
    def show_examples(group, label):
        print(f"\n--- {label} ---")
        for i, rec in enumerate(group[:5]):
            original = (rec.get('title', '') + " " + rec.get('abstract', ''))
            normalized = aggressive_normalize(original)
            
            print(f"{label[0]}{i+1} ORIGINAL (Snippet): {repr(original[:250])}...")
            print(f"{label[0]}{i+1} NORMALIZED (Snippet): {repr(normalized[:250])}...\n")
            
    show_examples(positives, "POSITIVES")
    show_examples(negatives, "NEGATIVES")

if __name__ == "__main__":
    main()
