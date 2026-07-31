import json
import pandas as pd

with open('data/final/cancer_pm_holdout.json', 'r', encoding='utf-8') as f:
    holdout = json.load(f)

df = pd.DataFrame(holdout)
print("Holdout Columns:", df.columns.tolist())
print("Publisher distribution:")
if 'publisher' in df.columns:
    print(df['publisher'].value_counts())
elif 'source' in df.columns:
    print(df['source'].value_counts())

print("\nHindawi holdout by label:")
if 'publisher' in df.columns:
    hindawi = df[df['publisher'] == 'Hindawi']
    print(hindawi['label'].value_counts())

with open('data/final/provenance_matched_negatives.json', 'r', encoding='utf-8') as f:
    prov = json.load(f)
df_prov = pd.DataFrame(prov)
print("\nProvenance matched negatives count:", len(df_prov))
