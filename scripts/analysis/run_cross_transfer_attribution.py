import json
import sys
import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

sys.stdout.reconfigure(encoding='utf-8')

import os
# Redirect output to file
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
output_path = os.path.join(project_root, "cross_transfer_attribution.txt")
sys.stdout = open(output_path, "w", encoding="utf-8")
sys.stderr = sys.stdout

class BERTClassifier(nn.Module):
    def __init__(self, model_name):
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

def main():
    print("="*80)
    print("CROSS-TRANSFER MODEL ATTRIBUTION ON HINDAWI HOLDOUT (POST-NORMALIZATION)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {device}")

    # Load normalized holdout records
    with open("data/final/cancer_pm_holdout_normalized.json", "r", encoding="utf-8") as f:
        holdout_records = json.load(f)
        
    hindawi_pos = [r for r in holdout_records if r["label"] == 1]
    print(f"Loaded {len(hindawi_pos)} Hindawi holdout positives.")

    # Load Model (Cross-Transfer: trained on non-Hindawi, evaluated on Hindawi)
    # We use the normalized version to completely isolate genuine style from formatting.
    pm_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(pm_model_name)
    model_path = "models/pubmedbert_norm_non_hindawi_best.pt"
    
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = BERTClassifier(pm_model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    def predict_prob(title_text, abstract_text):
        full_text = title_text + " " + abstract_text
        encoding = tokenizer(
            full_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            logit = model(input_ids, mask)
            prob = torch.sigmoid(logit.squeeze(-1)).item()
        return prob

    # Find highest confidence predictions
    print("Scoring Hindawi Holdout Positives...")
    predictions = []
    for r in hindawi_pos:
        prob = predict_prob(r["title"], r["abstract"])
        predictions.append((prob, r))
    
    # Sort by confidence descending
    predictions.sort(key=lambda x: x[0], reverse=True)
    
    # Output the top 5
    top_k = 5
    print(f"\nTop {top_k} most confident true positives (probabilities from {predictions[0][0]:.8f} to {predictions[top_k-1][0]:.8f}):")

    for k in range(top_k):
        orig_prob, r = predictions[k]
        title = r["title"]
        abstract = r["abstract"]
        
        print("\n" + "-"*80)
        print(f"PAPER {k+1}: DOI: {r.get('doi', '')} | PMID: {r.get('pmid', '')} | CONFIDENCE: {orig_prob:.8f}")
        print(f"TITLE: {title}")
        print("-"*80)
        
        # 1. Sentence-level perturbation
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', abstract) if s.strip()]
        
        sentence_attributions = []
        for i, s in enumerate(sentences):
            omitted_abstract = " ".join([sentences[j] for j in range(len(sentences)) if j != i])
            prob_without = predict_prob(title, omitted_abstract)
            drop = orig_prob - prob_without
            sentence_attributions.append((drop, s, prob_without))
        
        sentence_attributions.sort(key=lambda x: x[0], reverse=True)
        
        print("\nSentence Attributions:")
        for idx, (drop, s, p_w) in enumerate(sentence_attributions[:3]):
            print(f"  Rank {idx+1}: [Delta Prob: {drop:>+.8f} | Prob without: {p_w:.8f}]")
            print(f"    Text: \"{s}\"")

        # 2. Phrase-level perturbation (N-gram sliding window of length 5 words)
        words = abstract.split()
        phrase_attributions = []
        
        W = 5
        if len(words) >= W:
            for i in range(len(words) - W + 1):
                phrase = " ".join(words[i:i+W])
                omitted_abstract = " ".join(words[:i] + words[i+W:])
                prob_without = predict_prob(title, omitted_abstract)
                drop = orig_prob - prob_without
                phrase_attributions.append((drop, phrase, prob_without))
            
            phrase_attributions.sort(key=lambda x: x[0], reverse=True)
            
            print("\nPhrase Attributions (5-word windows):")
            # Only print top 5 distinct phrases to avoid overlapping windows cluttering
            printed_phrases = []
            printed_count = 0
            for drop, phrase, p_w in phrase_attributions:
                if printed_count >= 5:
                    break
                # Check overlap
                overlap = False
                for p in printed_phrases:
                    if len(set(phrase.split()) & set(p.split())) >= 3:
                        overlap = True
                        break
                if not overlap:
                    print(f"  Rank {printed_count+1}: [Delta: {drop:>+.8f}] \"{phrase}\"")
                    printed_phrases.append(phrase)
                    printed_count += 1

if __name__ == '__main__':
    main()
