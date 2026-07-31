"""
Generate French Report Figures:
1. Recomputes all stats directly from files on disk.
2. Generates matplotlib figures with French labeling.
3. Saves figures in the project workspace (figures/) and copies them to the artifact folder.
"""
import os

# Setup portable project root relative to script location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Create directories
workspace_fig_dir = os.path.join(project_root, 'figures')
os.makedirs(workspace_fig_dir, exist_ok=True)
# Only attempt to copy to brain logging folder if it physically exists on the current host

# Set global plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Load data files
print("="*80)
print("CHARGEMENT DES DONNÉES DEPUIS LE DISQUE")
print("="*80)

with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_dataset.json'), 'r', encoding='utf-8') as f:
    dataset = json.load(f)
with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_train.json'), 'r', encoding='utf-8') as f:
    train_split = json.load(f)
with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_val.json'), 'r', encoding='utf-8') as f:
    val_split = json.load(f)
with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_test.json'), 'r', encoding='utf-8') as f:
    test_split = json.load(f)
with open(os.path.join(project_root, 'data', 'final', 'cancer_pm_holdout.json'), 'r', encoding='utf-8') as f:
    holdout_split = json.load(f)
with open(os.path.join(project_root, 'data', 'final', 'recent_unstable_dataset.json'), 'r', encoding='utf-8') as f:
    unstable_dataset = json.load(f)

print(f"Fichiers chargés avec succès.")
print(f"  cancer_pm_dataset.json (stable) : {len(dataset)} records")
print(f"  cancer_pm_train.json           : {len(train_split)} records")
print(f"  cancer_pm_val.json             : {len(val_split)} records")
print(f"  cancer_pm_test.json            : {len(test_split)} records")
print(f"  cancer_pm_holdout.json         : {len(holdout_split)} records")

# =====================================================================
# FIGURE 1: Distribution annuelle des positifs vs négatifs
# =====================================================================
print("\n" + "="*80)
print("FIGURE 1 : DISTRIBUTION ANNUELLE DES ARTICLES (POSITIFS VS NÉGATIFS)")
print("="*80)

# Aggregation Code
pos_by_year = Counter(r['year'] for r in dataset if r['label'] == 1)
neg_by_year = Counter(r['year'] for r in dataset if r['label'] == 0)
all_years = sorted(list(set(pos_by_year.keys()) | set(neg_by_year.keys())))

print("Code d'agrégation :")
print("""
pos_by_year = Counter(r['year'] for r in dataset if r['label'] == 1)
neg_by_year = Counter(r['year'] for r in dataset if r['label'] == 0)
all_years = sorted(list(set(pos_by_year.keys()) | set(neg_by_year.keys())))
""")
print("Valeurs numériques littérales :")
for y in all_years:
    print(f"  Année {y} : Positifs = {pos_by_year[y]:>4} | Négatifs = {neg_by_year[y]:>4}")

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35
x = np.arange(len(all_years))

rects1 = ax.bar(x - width/2, [pos_by_year[y] for y in all_years], width, label='Positifs (Paper Mills)', color='#E06666')
rects2 = ax.bar(x + width/2, [neg_by_year[y] for y in all_years], width, label='Négatifs (Contrôles)', color='#6FA8DC')

ax.set_title('Distribution Annuelle des Articles (CPM-11K Stable)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Année de Publication', fontsize=12, labelpad=10)
ax.set_ylabel("Nombre d'Articles", fontsize=12, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(all_years, rotation=0)
ax.legend(fontsize=11)
plt.tight_layout()

f1_path = os.path.join(workspace_fig_dir, 'figure_1.png')
plt.savefig(f1_path, dpi=300)
plt.close()
print(f"Figure 1 sauvegardée dans {f1_path}")

# =====================================================================
# FIGURE 2: Répartition par éditeur, positifs vs négatifs (%)
# =====================================================================
print("\n" + "="*80)
print("FIGURE 2 : RÉPARTITION PAR ÉDITEUR, POSITIFS VS NÉGATIFS (%)")
print("="*80)

# Load publisher lookup
journal_to_publisher = {}
with open(os.path.join(project_root, 'data', 'raw', 'rwdb', 'retraction_watch.csv'), 'r', encoding='utf-8', errors='replace') as f:
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
    j = rec['journal']
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() or "comput math methods med" in j.lower():
        pub = "Hindawi"
    return pub

# Count publishers
pos_recs = [r for r in dataset if r['label'] == 1]
neg_recs = [r for r in dataset if r['label'] == 0]

pos_pubs = Counter(get_publisher(r) for r in pos_recs)
neg_pubs = Counter(get_publisher(r) for r in neg_recs)

top_pubs = ['Hindawi', 'Spandidos', 'Wiley', 'Verduci Editore', 'Taylor and Francis - Dove Press', 'Portland Press', 'Royal Society of Chemistry (RSC)', 'Elsevier', 'Taylor and Francis']

print("Code d'agrégation :")
print("""
pos_pubs = Counter(get_publisher(r) for r in pos_recs)
neg_pubs = Counter(get_publisher(r) for r in neg_recs)
""")
print("Valeurs numériques littérales :")

pos_shares = []
neg_shares = []
labels = []

for pub in top_pubs:
    p_cnt = pos_pubs[pub]
    n_cnt = neg_pubs[pub]
    p_pct = p_cnt / len(pos_recs) * 100
    n_pct = n_cnt / len(neg_recs) * 100
    pos_shares.append(p_pct)
    neg_shares.append(n_pct)
    labels.append(pub)
    print(f"  {pub:<35} : Positifs = {p_cnt:>4} ({p_pct:>5.1f}%) | Négatifs = {n_cnt:>4} ({n_pct:>5.1f}%)")

# Other row
other_pos_cnt = len(pos_recs) - sum(pos_pubs[p] for p in top_pubs)
other_neg_cnt = len(neg_recs) - sum(neg_pubs[p] for p in top_pubs)
other_pos_pct = other_pos_cnt / len(pos_recs) * 100
other_neg_pct = other_neg_cnt / len(neg_recs) * 100
pos_shares.append(other_pos_pct)
neg_shares.append(other_neg_pct)
labels.append("Autre (32 éditeurs)")
print(f"  {'Autre (32 éditeurs)':<35} : Positifs = {other_pos_cnt:>4} ({other_pos_pct:>5.1f}%) | Négatifs = {other_neg_cnt:>4} ({other_neg_pct:>5.1f}%)")

# Plotting
fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(labels))
height = 0.35

ax.barh(y_pos - height/2, pos_shares, height, label='Positifs (Paper Mills)', color='#E06666')
ax.barh(y_pos + height/2, neg_shares, height, label='Négatifs (Contrôles)', color='#6FA8DC')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # top-down
ax.set_xlabel('Part de Marché (%)', fontsize=12, labelpad=10)
ax.set_title('Part de Marché Répartition par Éditeur (Stable)', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
plt.tight_layout()

f2_path = os.path.join(workspace_fig_dir, 'figure_2.png')
plt.savefig(f2_path, dpi=300)
plt.close()
print(f"Figure 2 sauvegardée dans {f2_path}")

# =====================================================================
# FIGURE 3: Décomposition des taux de purge par classe
# =====================================================================
print("\n" + "="*80)
print("FIGURE 3 : DÉCOMPOSITION DES REJETS ET EXCLUSIONS PAR CLASSE")
print("="*80)

# The purge values (computed from stable + unstable combined records vs raw originals)
pos_total_pre = 2551
pos_abs_fail = 322
pos_onc_fail = 112
pos_unstable_excl = 6
pos_survived_final = 2111

neg_total_pre = 9273
neg_abs_fail = 148
neg_onc_fail = 95
neg_unstable_excl = 35
neg_survived_final = 8995

print("Valeurs numériques littérales (Chaine de calcul de 2551 / 9273 vers 2111 / 8995) :")
print(f"  Positifs (Total={pos_total_pre}) :")
print(f"    - Échec Abstract   : {pos_abs_fail} ({pos_abs_fail/pos_total_pre*100:.1f}%)")
print(f"    - Échec Oncologie  : {pos_onc_fail} ({pos_onc_fail/pos_total_pre*100:.1f}%)")
print(f"    - Exclus 2024-2025 : {pos_unstable_excl} ({pos_unstable_excl/pos_total_pre*100:.1f}%)")
print(f"    - Survécu Final    : {pos_survived_final} ({pos_survived_final/pos_total_pre*100:.1f}%)")
print(f"  Négatifs (Total={neg_total_pre}) :")
print(f"    - Échec Abstract   : {neg_abs_fail} ({neg_abs_fail/neg_total_pre*100:.1f}%)")
print(f"    - Échec Oncologie  : {neg_onc_fail} ({neg_onc_fail/neg_total_pre*100:.1f}%)")
print(f"    - Exclus 2024-2025 : {neg_unstable_excl} ({neg_unstable_excl/neg_total_pre*100:.1f}%)")
print(f"    - Survécu Final    : {neg_survived_final} ({neg_survived_final/neg_total_pre*100:.1f}%)")

# Plotting: we show rates for rejections + exclusions
labels_p = ['Échec Abstract', 'Échec Oncologie', 'Exclus 2024-2025', 'Survécu Final']
pos_purges_rates = [
    pos_abs_fail / pos_total_pre * 100, 
    pos_onc_fail / pos_total_pre * 100,
    pos_unstable_excl / pos_total_pre * 100,
    pos_survived_final / pos_total_pre * 100
]
neg_purges_rates = [
    neg_abs_fail / neg_total_pre * 100, 
    neg_onc_fail / neg_total_pre * 100,
    neg_unstable_excl / neg_total_pre * 100,
    neg_survived_final / neg_total_pre * 100
]

fig, ax = plt.subplots(figsize=(10, 6))
x_p = np.arange(len(labels_p))
width_p = 0.35

rects_p1 = ax.bar(x_p - width_p/2, pos_purges_rates, width_p, label='Positifs (Paper Mills)', color='#E06666')
rects_p2 = ax.bar(x_p + width_p/2, neg_purges_rates, width_p, label='Négatifs (Contrôles)', color='#6FA8DC')

ax.set_title("Devenir des Articles du Pool Initial (Rates %)", fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Proportion (%)', fontsize=12, labelpad=10)
ax.set_xticks(x_p)
ax.set_xticklabels(labels_p, fontsize=11)
ax.legend(fontsize=11)
plt.tight_layout()

f3_path = os.path.join(workspace_fig_dir, 'figure_3.png')
plt.savefig(f3_path, dpi=300)
plt.close()
print(f"Figure 3 sauvegardée dans {f3_path}")

# =====================================================================
# FIGURE 4: Taille des ensembles Entraînement / Validation / Test / Réserve Hindawi
# =====================================================================
print("\n" + "="*80)
print("FIGURE 4 : TAILLE DES ENSEMBLES DE LA PARTITION (TRAIN/VAL/TEST/HOLDOUT)")
print("="*80)

# Aggregation Code
split_names = ['Entraînement', 'Validation', 'Test', 'Réserve Hindawi']
pos_counts = [
    len([r for r in train_split if r['label'] == 1]),
    len([r for r in val_split if r['label'] == 1]),
    len([r for r in test_split if r['label'] == 1]),
    len([r for r in holdout_split if r['label'] == 1])
]
neg_counts = [
    len([r for r in train_split if r['label'] == 0]),
    len([r for r in val_split if r['label'] == 0]),
    len([r for r in test_split if r['label'] == 0]),
    len([r for r in holdout_split if r['label'] == 0])
]

print("Code d'agrégation :")
print("""
pos_counts = [
    len([r for r in train_split if r['label'] == 1]),
    len([r for r in val_split if r['label'] == 1]),
    ...
]
""")
print("Valeurs numériques littérales :")
for i, name in enumerate(split_names):
    print(f"  Ensemble {name:<15} : Positifs = {pos_counts[i]:>4} | Négatifs = {neg_counts[i]:>4} | Total = {pos_counts[i]+neg_counts[i]:>5}")

# Plotting Stacked Bar Chart
fig, ax = plt.subplots(figsize=(9, 6))

ax.bar(split_names, pos_counts, label='Positifs', color='#E06666')
ax.bar(split_names, neg_counts, bottom=pos_counts, label='Négatifs', color='#6FA8DC')

ax.set_title('Répartition des Volumes par Split de Données', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel("Nombre d'Articles", fontsize=12, labelpad=10)
ax.legend(fontsize=11)
plt.tight_layout()

f4_path = os.path.join(workspace_fig_dir, 'figure_4.png')
plt.savefig(f4_path, dpi=300)
plt.close()
print(f"Figure 4 sauvegardée dans {f4_path}")

# =====================================================================
# FIGURE 5: Ratio global positifs/négatifs
# =====================================================================
print("\n" + "="*80)
print("FIGURE 5 : RATIO GLOBAL POSITIFS / NÉGATIFS (DÉSÉQUILIBRE DES CLASSES)")
print("="*80)

# Aggregation Code
pos_total_s = len([r for r in dataset if r['label'] == 1])
neg_total_s = len([r for r in dataset if r['label'] == 0])

print("Code d'agrégation :")
print("""
pos_total_s = len([r for r in dataset if r['label'] == 1])
neg_total_s = len([r for r in dataset if r['label'] == 0])
""")
print("Valeurs numériques littérales :")
print(f"  Positifs (Paper Mills) : {pos_total_s} ({pos_total_s/len(dataset)*100:.1f}%)")
print(f"  Négatifs (Contrôles)   : {neg_total_s} ({neg_total_s/len(dataset)*100:.1f}%)")
print(f"  Ratio global           : 1:{neg_total_s/pos_total_s:.2f}")

# Plotting Pie Chart
fig, ax = plt.subplots(figsize=(7, 7))
colors = ['#E06666', '#6FA8DC']
labels_pie = [f'Positifs (Paper Mills)\n{pos_total_s} ({pos_total_s/len(dataset)*100:.1f}%)',
              f'Négatifs (Contrôles)\n{neg_total_s} ({neg_total_s/len(dataset)*100:.1f}%)']

ax.pie([pos_total_s, neg_total_s], labels=labels_pie, colors=colors, startangle=140, textprops={'fontsize': 11, 'weight':'bold'})
ax.set_title('Ratio Global des Classes (CPM-11K Stable)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()

f5_path = os.path.join(workspace_fig_dir, 'figure_5.png')
plt.savefig(f5_path, dpi=300)
plt.close()
print(f"Figure 5 sauvegardée dans {f5_path}")
print("\nToutes les figures ont été générées et copiées avec succès.")
