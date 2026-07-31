# Chapitre : Détection Automatique des Usines à Articles (Paper Mills) dans la Littérature Oncologique

## Résumé

Ce chapitre présente une étude systématique sur la détection automatique des publications scientifiques frauduleuses issues d'usines à articles (*paper mills*) dans le domaine de l'oncologie. Pour ce faire, nous construisons le jeu de données CPM-11K, regroupant 11 106 articles cliniques et translationnels indexés dans la base de données Retraction Watch (RWDB) et MEDLINE/PubMed, et structurés de manière à éliminer les fuites de cibles (*label leakage*) et les biais temporels. Un protocole strict de division des données par regroupement d'auteurs et de titres disjoint est implémenté, résolvant un problème majeur de percolation des patronymes fréquents. Nous entraînons et comparons plusieurs configurations sémantiques et lexicales. 

Notre modèle de référence, un classifieur linéaire robuste basé sur des sacs de mots pondérés (TF-IDF Combiné, $min\_df=5$, $C=10,0$), atteint une performance globale de premier ordre avec un score F1 global de 53,66 % et un ROC-AUC de 80,89 % sur l'ensemble de validation. Cependant, une évaluation stratifiée par éditeur révèle une asymétrie de performance majeure : alors que les articles frauduleux de Spandidos (caractérisés par des fraudes d'images grossières habillées de textes stéréotypés) sont détectés avec un F1 de 87,80 %, le modèle échoue face à une prédiction positive naïve sur le sous-ensemble éditorial Hindawi. Une analyse causale qualitative des motifs de rétractation et un test dose-réponse par rééchantillonnage bootstrap suggèrent que les fraudes basées sur la manipulation des processus éditoriaux (évaluation par les pairs biaisée, cartels de citations) ne laissent pas de signal lexical facilement extractible par les représentations testées dans les résumés. Ces résultats sont cohérents avec une évaluation finale sur un ensemble de test indépendant (ROC-AUC de 65,71 % sur Hindawi) et un ensemble de généralisation étanche Hindawi-Holdout de 1 504 articles (F1 de 40,63 % vs 44,39 % pour la baseline triviale, ROC-AUC de 64,47 %). Ce travail suggère les limites de l'analyse textuelle sémantique (au niveau des résumés) pour la détection de la fraude de processus, et ouvre la voie à des approches multi-modales intégrant les métadonnées de publication.

---

## Introduction

La prolifération des usines à articles (*paper mills*) représente l'un des défis les plus critiques pour l'intégrité de la communication scientifique contemporaine. Ces organisations commerciales produisent à l'échelle des manuscrits frauduleux, souvent basés sur des modèles sémantiques et des données d'imagerie falsifiées, pour les vendre à des chercheurs sous pression de publication. Le domaine de l'oncologie clinique et translationnelle est particulièrement touché par ce fléau en raison de la forte demande de publications dans cette discipline.

L'objectif de ce travail est de concevoir et d'évaluer des modèles d'apprentissage automatique capables de détecter automatiquement ces publications frauduleuses à partir de leurs résumés (*abstracts*). Ce chapitre détaille la construction du jeu de données de référence CPM-11K, le protocole strict de division des données visant à empêcher les fuites d'information, l'analyse exploratoire des résumés, la modélisation lexicale et sémantique, et enfin, une analyse causale approfondie des limites rencontrées sur le sous-ensemble éditorial Hindawi.

---

## 1. Construction du Jeu de Données CPM-11K

Pour l'apprentissage supervisé, nous avons construit le jeu de données Cancer Paper Mill (CPM-11K), composé de **11 106 articles** (2 111 positifs, 8 995 négatifs, ratio 1:4,26).

### 1.1. Source positive et groupe contrôle négatif
*   **Classe positive (Paper Mills) :** Extraite de la base Retraction Watch Database (RWDB) [1] pour le sujet « *cancer* », puis récupérée via APIs PubMed/Crossref sous sa forme originale (avant rétraction).
*   **Classe négative (Contrôles sains) :** Appariée thématiquement pour chaque cas positif dans le même journal et la même année ($± 1$ an). La recherche PubMed [2] a été limitée au domaine oncologique par une requête de mots-clés, avec validation automatisée via un script de correspondance de mots-clés oncologiques dans le titre. Les publications de type éditoriaux ou revues ont été exclues.

![Distribution annuelle des classes positive et négative](figures/figure_1.png)
*Figure 1 : Distribution annuelle des articles positifs (paper mills) et négatifs (contrôles sains) dans le jeu de données CPM-11K.*

![Part des éditeurs post-purge](figures/figure_2.png)
*Figure 2 : Parts relatives des principaux éditeurs au sein de la classe positive (paper mills) après application des filtres de purge.*

![Ratio global de déséquilibre des classes](figures/figure_5.png)
*Figure 3 : Proportion globale des classes positive et négative au sein du jeu de données final CPM-11K (ratio de déséquilibre de 1:4.26).*

### 1.2. Nettoyage et purge des données
1.  **Label leakage :** Les mentions éditoriales de rétraction insérées a posteriori (ex. « *Retracted:* ») ont été éliminées par expressions régulières.
2.  **Cascade de purge :** Après exclusion des résumés irrécupérables ou non oncologiques, le taux de perte s'élève à 17,0 % pour la classe positive contre 2,6 % pour la classe négative. Ce déséquilibre s'explique par les politiques de retrait sélectif des éditeurs.
3.  **Filtrage temporel :** Les articles des années 2024-2025 ont été exclus en raison du délai incompressible de signalement des rétractions (*retraction lag*).

![Cascade de purge et d'exclusion](figures/figure_3.png)
*Figure 4 : Cascade méthodologique d'inclusion, d'exclusion et de nettoyage des données pour les classes positive et négative.*

---

## 2. Protocole Stratifié de Division des Données

Afin d'éviter le surapprentissage de signatures spécifiques, nous avons conçu un protocole de division rigoureux.

### 2.1. Regroupement par clusters disjoints
Pour empêcher que des articles issus d'une même campagne d'usines à articles se retrouvent à la fois dans le train et le test, les résumés ont été regroupés par clusters selon deux critères :
*   Similarité de titre (distance de Jaccard sur 3-grammes de caractères $≥ 0,7$ dans un même journal).
*   Co-autorat partagé.

### 2.2. Percolation des patronymes
L'analyse de l'implémentation initiale du co-autorat (un seul auteur commun) a révélé un problème de percolation : un cluster unique regroupait 979 articles en raison de la prévalence de noms de famille chinois très fréquents (ex. *Wang*, *Zhang*). Ce problème a été résolu en **durcissant le critère à au moins 2 auteurs partagés en commun** pour lier deux documents.

### 2.3. Réserve étanche Hindawi et splits
1.  **Holdout Hindawi :** Un pool indépendant contenant 50 % des articles Hindawi (**429 positifs, 1 075 négatifs**) a été mis de côté avant toute analyse ou entraînement. Il sert exclusivement de validation externe hors-domaine.
2.  **Splits Entraînement / Validation / Test :** Les 9 602 articles restants ont été divisés à l'échelle des clusters disjoints selon le ratio 70/15/15 :
    *   **Entraînement :** 6 722 articles (1 178 pos, 5 544 neg)
    *   **Validation :** 1 440 articles (252 pos, 1 188 neg)
    *   **Test :** 1 440 articles (252 pos, 1 188 neg)

![Partition split volumes](figures/figure_4.png)
*Figure 5 : Répartition des volumes d'articles positifs et négatifs au sein des partitions d'entraînement, de validation et de test (le pool Holdout Hindawi restant à part).*

---

## 3. Analyse Exploratoire des Données (EDA)

Une analyse descriptive a été menée sur la partition d'entraînement afin de vérifier l'absence de biais.

### 3.1. Longueurs des textes
Les distributions des longueurs de résumés sont homogènes entre les classes (médianes de 213 mots pour les positifs vs 208 pour les négatifs), neutralisant l'utilisation de la longueur comme descripteur.

| Classe | Statistique | Longueur en caractères | Longueur en mots |
| :--- | :--- | :---: | :---: |
| **Positifs (Usines à articles)** | Moyenne $±$ Écart-type | 1437,1 $±$ 499,1 | 203,8 $±$ 70,5 |
| *(N = 1 178)* | Médiane | 1492,0 | 213,0 |
| | Min / Max | 50 / 3703 | 7 / 540 |
| **Négatifs (Légitimes)** | Moyenne $±$ Écart-type | 1417,1 $±$ 510,4 | 202,2 $±$ 73,3 |
| *(N = 5 544)* | Médiane | 1463,0 | 208,0 |
| | Min / Max | 50 / 4983 | 6 / 789 |

![Distribution des longueurs des résumés](figures/figure_eda_lengths.png)
*Figure 6 : Histogrammes comparatifs des distributions de longueurs des résumés (mots et caractères) pour les classes positive (paper mills) et négative (littérature légitime).*

### 3.2. Biais lexicaux et d'éditeurs
*   **Fuite textuelle :** Les termes comme `retract` ou `withdraw` représentent moins de 0,35 % des occurrences. Les noms des éditeurs (`hindawi`, `spandidos`) sont absents des résumés, garantissant le nettoyage du texte standardisé (boilerplate).
*   **Analyse thématique locale :** L'extraction par ratio de log-cotes pondérées montre que les termes discriminants de la classe positive sont très spécifiques et diffèrent selon les éditeurs (ex. `efhd2` pour Hindawi, `mir-744` pour Spandidos, `hispidulin` pour les autres). La quasi-totalitÃ© de ces termes ont une fréquence de document extrêmement faible (DF < 10), suggérant la présence de gabarits locaux de campagnes de fraude plutôt qu'un signal sémantique généralisable.

---
## 4. Modélisation et Évaluations Lexicales

Nous présentons dans cette section l'entraînement du modèle de référence, les stratégies d'optimisation lexicale et la comparaison des performances globales sur l'ensemble de validation.

### 4.1. Modèle de référence et optimisation des hyperparamètres
Le classifieur principal est un modèle linéaire robuste (Régression Logistique) entraîné sur des représentations de sacs de mots pondérées (TF-IDF). Deux paramètres clés influençant la capacité de généralisation et l'overfitting ont été optimisés de manière empirique :

1.  **Le seuil de fréquence minimale des termes (`min_df`) :**
    *   Le modèle initial (baseline) utilisait un filtre `min_df=2` (conservant les n-grammes apparaissant au moins 2 fois), produisant un vocabulaire très large de **104 138 caractéristiques**.
    *   Pour limiter le bruit lié aux termes ultra-spécifiques mis en évidence en Section 3.3, nous avons évalué un filtre plus agressif à `min_df=5`. Cette restriction a permis d'éliminer **64,5 % du vocabulaire**, réduisant la dimensionnalité à **36 929 caractéristiques stables**.
2.  **La force de régularisation ($C$) :**
    Nous avons effectué un balayage étendu de l'hyperparamètre de régularisation $C$ (où des valeurs plus élevées réduisent la pénalisation L2) de 1.0 à 10 000 afin de trouver le point d'inflexion exact des performances.

#### Tableau du balayage de régularisation $C$ (TF-IDF, min_df=2)

| Valeur de $C$ | Seuil de décision optimal | F1 Global (Validation) | Hindawi F1 | Spandidos F1 | Others F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1.0 (Baseline)** | 0.37 | 46,91% | 39,15% | 84,00% | 47,77% |
| **10.0** | 0.29 | 52,50% | **41,58%** | 85,11% | 57,63% |
| **30.0** | 0.29 | 53,92% | 40,00% | 85,00% | 62,55% |
| **100.0** | 0.10 | 51,44% | 40,00% | 85,11% | 56,31% |
| **300.0** | 0.15 | **55,30%** | 40,77% | 93,02% | **63,78%** |
| **1 000.0** | 0.08 | 53,07% | 37,40% | 93,02% | 61,13% |
| **3 000.0** | 0.02 | 52,17% | 35,44% | 93,02% | 60,29% |
| **10 000.0** | 0.01 | 52,44% | 35,98% | 93,02% | 60,52% |

*Analyse du point d'inflexion :* Le score F1 global atteint son maximum à $C=300$ (55,30%), poussé par l'amélioration de la strate "Others". Cependant, au-delà de $C=10$, la performance sur le stratum Hindawi décline de manière continue (chutant de 41,58% à 35,98%). De plus, pour des valeurs de $C \ge 100$, le seuil de décision optimal s'effondre vers des valeurs aberrantes (0.10, 0.02, 0.01), indiquant que les probabilités prédites se concentrent anormalement près de zéro à cause d'un surapprentissage massif.

Pour examiner le comportement de régularisation après filtrage du vocabulaire à `min_df=5`, nous avons effectué un sous-balayage de $C$ sur ce nouveau vecteur.

#### Tableau du sous-balayage de régularisation $C$ (TF-IDF, min_df=5)

| Valeur de $C$ | Seuil de décision optimal | F1 Global (Validation) | Hindawi F1 | Spandidos F1 | Others F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.38 | 47,90% | 40,69% | 85,71% | 48,53% |
| **10.0 (Combiné)** | 0.36 | **53,66%** | **41,83%** | **87,80%** | 60,00% |
| **30.0** | 0.32 | 53,67% | 40,93% | 85,00% | **61,54%** |
| **100.0** | 0.21 | 53,26% | 40,62% | 82,93% | 60,74% |
| **300.0** | 0.15 | 52,91% | 40,15% | 82,93% | 60,67% |

*Justification du modèle sélectionné :* Le modèle Combiné (`min_df=5` et `C=10.0`) permet de stabiliser les performances. Il offre le meilleur score F1 sur Hindawi (**41,83%**), le meilleur score sur Spandidos (**87,80%**) et un excellent score sur les autres éditeurs (60,00%), tout en ramenant le seuil de décision optimal à une valeur robuste et naturelle (0.36). Augmenter $C$ au-delà de 10.0 n'apporte aucun gain significatif sur l'ensemble global et dégrade systématiquement la strate Hindawi.

### 4.2. Configuration des modèles testés
Dix architectures distinctes ont été entraînées et comparées :
1.  **Trivial Baseline (Prédiction positive constante) :** Un classifieur naïf prédisant systématiquement la classe positive pour tous les articles du corpus.
2.  **TF-IDF Baseline (min_df=2, C=1.0) :** Représentation de n-grammes de mots (1 et 2) brute avec faible filtrage.
3.  **TF-IDF Combiné (min_df=5, C=10.0) :** Notre modèle proposé combinant réduction de vocabulaire et régularisation ajustée.
4.  **N-grammes de caractères (char_wb 3-5, C=1.0) :** Modélisation par n-grammes de caractères de taille 3 à 5 à l'intérieur des limites de mots pour capturer les racines morphologiques (vocabulaire de 143 989 descripteurs).
5.  **Embeddings Généraux (all-MiniLM-L6-v2) [5, 6] :** Extraction de plongements sémantiques généraux de dimension 384 à partir du texte combiné (titre + résumé), suivie d'une régression logistique.
6.  **Embeddings Biomédicaux (PubMedBERT gelé) [4] :** Extraction de plongements sémantiques spécialisés de dimension 768 à partir d'un modèle BERT pré-entraîné sur la littérature scientifique PubMed (`pritamdeka/S-PubMedBert-MS-MARCO`), suivie d'une régression logistique.
7.  **SVM Linéaire (TF-IDF, min_df=5) :** Machine à vecteurs supports linéaire entraînée sur le même vocabulaire TF-IDF réduit que notre modèle combiné, avec pondération équilibrée des classes.
8.  **XGBoost (TF-IDF, min_df=5) :** Arbres de décision boostés par gradient entraînés sur le même vocabulaire TF-IDF, avec pondération positive de classe (`scale_pos_weight`) pour gérer le déséquilibre.
9.  **PubMedBERT (fine-tuned) :** Fine-tuning de bout en bout du modèle de classification biomédical de base (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) sur notre jeu de données avec un taux d'apprentissage de 2e-5, une taille de batch de 16 (effective) et une précision mixte FP16 pendant 3 époques.
10. **SciBERT (fine-tuned) :** Fine-tuning de bout en bout du modèle scientifique général de base (`allenai/scibert_scivocab_uncased`) avec le même protocole d'entraînement que le modèle PubMedBERT pour une comparaison équitable.

### 4.3. Comparaison des performances globales et baselines triviales
L'évaluation globale de ces modèles sur l'ensemble de validation (1 440 articles) est présentée ci-dessous.

#### Tableau comparatif des performances globales (Validation)

| Modèle | Précision | Rappel | F1 Global | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** | 17,50% | **100,00%** | 29,79% [26,92% - 32,56%] | 50,00% [50,00% - 50,00%] | 17,50% |
| **TF-IDF Baseline (min_df=2, C=1.0)** | 36,30% | 66,27% | 46,91% [42,10% - 51,45%] | 77,86% [74,64% - 80,97%] | 50,97% |
| **TF-IDF Combiné (min_df=5, C=10.0)** | 47,83% | 61,11% | 53,66% [48,62% - 58,20%] | 80,89% [77,84% - 83,68%] | 54,99% |
| **Character N-grams (char_wb 3-5)** | 51,66% | 43,25% | 47,08% [41,28% - 52,49%] | 78,21% [75,20% - 81,04%] | 50,65% |
| **Embeddings Généraux (MiniLM)** | 33,33% | 48,81% | 39,61% [34,75% - 44,48%] | 69,28% [65,74% - 72,86%] | 32,08% |
| **Embeddings Biomédicaux (PubMedBERT gelé)** | 39,09% | 54,76% | 45,62% [40,56% - 50,32%] | 78,22% [75,24% - 81,06%] | 43,92% |
| **SVM Linéaire (TF-IDF, min_df=5)** | 47,52% | 60,71% | 53,31% [48,20% - 58,00%] | 80,83% [77,76% - 83,66%] | 55,24% |
| **XGBoost (TF-IDF, min_df=5)** | 37,04% | 59,52% | 45,66% [40,88% - 50,42%] | 76,08% [72,63% - 79,29%] | 48,04% |
| **PubMedBERT (fine-tuned)†** | **100,00%** | **100,00%** | **100,00% [100,00% - 100,00%]** | **100,00% [100,00% - 100,00%]** | **100,00%** |
| **SciBERT (fine-tuned)†** | **100,00%** | **100,00%** | **100,00% [100,00% - 100,00%]** | **100,00% [100,00% - 100,00%]** | **100,00%** |

*† Voir Section 5.8 pour une mise en garde critique concernant l'interprétation de ce résultat.*

![Comparaison globale des performances](figures/comparaison_globale.png)
*Figure 7 : Comparaison des scores F1 et ROC-AUC globaux des modèles classiques et gelés sur l'ensemble de validation.*

#### Tableau comparatif stratifié par éditeur (Validation F1 / ROC-AUC)

| Modèle | Spandidos F1 (ROC-AUC) | Autres (Regroupés) F1 (ROC-AUC) | Hindawi (In-Pool) F1 (ROC-AUC) |
| :--- | :---: | :---: | :---: |
| **Trivial Baseline** | 49,44% (50,00%) | 19,61% (50,00%) | 42,95% (50,00%) |
| **TF-IDF Baseline** | 84,00% (96,26%) | 47,77% (93,49%) | 39,15% (61,20%) |
| **TF-IDF Combiné** | 87,80% (96,97%) | 60,00% (93,10%) | 41,83% (63,27%) |
| **Character N-grams** | 78,95% (94,34%) | 62,18% (93,01%) | 21,39% (61,79%) |
| **PubMedBERT (gelé)** | 71,11% (88,79%) | 46,74% (86,91%) | 40,15% (65,71%) |
| **SVM Linéaire** | 82,93% (96,87%) | 60,00% (92,52%) | 41,83% (63,54%) |
| **XGBoost** | 73,91% (91,41%) | 46,29% (88,85%) | 40,15% (61,83%) |
| **PubMedBERT (fine-tuned)†** | **100,00% (100,00%)** | **100,00% (100,00%)** | **100,00% (100,00%)** |
| **SciBERT (fine-tuned)†** | **100,00% (100,00%)** | **100,00% (100,00%)** | **100,00% (100,00%)** |

![Comparaison stratifiée par éditeur](figures/comparaison_stratified.png)
*Figure 8 : Comparaison des scores F1 des modèles par strate d'éditeur (Hindawi, Spandidos, Pooled Others) sur l'ensemble de validation.*

*† Voir Section 5.8 pour une mise en garde critique concernant l'interprétation de ce résultat.*


*Analyse des performances globales et classiques :* Tous les modèles entraînés surpassent significativement la baseline triviale au niveau global (F1 de 29,79%). Le modèle **TF-IDF Combiné** s'impose comme le plus performant des modèles lexicaux linéaires classiques avec un F1 global de **53,66%** et un ROC-AUC de **80,89%**. Le SVM Linéaire obtient des performances quasi-identiques (F1 de 53,31%), confirmant la robustesse de l'approche linéaire sur ce type de caractéristiques textuelles. En revanche, le modèle XGBoost s'avère moins performant sur ces caractéristiques (F1 de 45,66%), ce qui s'explique par les difficultés inhérentes aux méthodes basées sur les arbres de décision face aux représentations textuelles très creuses (sparse) à haute dimensionnalité.



*Analyse des modèles fine-tunés (PubMedBERT vs SciBERT) :* Le fine-tuning de bout en bout des transformeurs pré-entraînés produit des résultats parfaits (F1 de 100,00% et ROC-AUC de 1,0000) sur l'ensemble global et sur toutes les strates d'éditeurs (y compris Hindawi). Le protocole de suivi par époque montre une convergence immédiate :
- **PubMedBERT (fine-tuned) :** Époque 1 (F1 = 100,00% | AUC = 1,0000), Époque 2 (F1 = 100,00% | AUC = 1,0000), Époque 3 (F1 = 100,00% | AUC = 1,0000). Le modèle sélectionné est celui de l'époque 1.
- **SciBERT (fine-tuned) :** Époque 1 (F1 = 100,00% | AUC = 1,0000), Époque 2 (F1 = 100,00% | AUC = 1,0000), Époque 3 (F1 = 100,00% | AUC = 1,0000). Le modèle sélectionné est celui de l'époque 1.

Ces résultats suggèrent une capacité d'apprentissage exceptionnelle des représentations neuronales profondes, mais soulèvent des questions méthodologiques cruciales concernant les mécanismes sous-jacents de détection (analysés en Section 5.8).

*Analyse de significativité statistique (Validation) :* Afin de valider si la supériorité du modèle TF-IDF Combiné par rapport aux autres approches classiques et baselines gelées est statistiquement significative, nous avons appliqué le test de McNemar (test apparié sur les prédictions individuelles) et estimé l'intervalle de confiance à 95% par bootstrap (2 000 itérations) de la différence de score F1 (dF1) par rapport à notre modèle sélectionné :
1. **vs. TF-IDF Baseline :** McNemar $\chi^2 = 67,70$ ($p < 0,000001$), ce qui confirme une différence hautement significative. Le bootstrap estime une différence de F1 moyenne de $+6,82\%$ avec un intervalle de confiance à 95% de $[3,64\%, 10,03\%]$, excluant largement 0.
2. **vs. Character N-grams :** McNemar $\chi^2 = 2,45$ ($p = 0,117$), montrant que la répartition des erreurs de classification individuelles n'est pas statistiquement distincte au seuil de $5\%$. Cependant, le bootstrap F1 estime une différence de F1 moyenne de $+6,60\%$ avec un intervalle de confiance à 95% de $[2,04\%, 11,20\%]$, indiquant une amélioration robuste et significative de la métrique F1 globale en faveur du modèle de mots.
3. **vs. Embeddings Biomédicaux (PubMedBERT gelé) :** McNemar $\chi^2 = 14,62$ ($p = 0,000132$), confirmant une différence statistiquement significative. Le bootstrap estime une différence de F1 moyenne de $+8,10\%$ avec un intervalle de confiance à 95% de $[3,65\%, 12,68\%]$, démontrant la supériorité statistique du modèle lexical sur les embeddings gelés.

![Forest plot de significativité statistique](figures/significativite_forest_plot.png)
*Figure 9 : Intervalles de confiance à 95% de la différence de F1 (dF1) par rapport au modèle TF-IDF Combiné et p-values associées au test de McNemar.*


---

## 5. Limites Stratifiées et Analyse Causale : Le cas d'Hindawi

Cette section présente l'analyse la plus critique de ce travail : l'étude de la divergence de performance selon les éditeurs, et plus particulièrement le cas du sous-ensemble Hindawi.

> [!IMPORTANT]
> **Clarification méthodologique essentielle :** Les analyses présentées dans cette section (évaluation, courbes et distributions) concernent exclusivement le sous-ensemble de validation interne **Hindawi (In-Pool) (134 positifs et 356 négatifs)** intégré dans le split principal de 1 440 articles. La réserve étanche de généralisation **Holdout Hindawi (429 positifs et 1 075 négatifs)** est restée totalement inviolée et non évaluée à ce stade.

### 5.1. Échec de la modélisation textuelle sur Hindawi
Lorsque l'on segmente les performances du modèle TF-IDF Combiné et qu'on les compare à la baseline triviale (« toujours prédire positif ») par éditeur sur l'ensemble de validation, un comportement anormal apparaît :

#### Tableau comparatif stratifié par éditeur (Validation)

| Éditeur | Taille de l'échantillon (Pos/Neg) | Baseline Triviale (F1) | TF-IDF Combiné (F1) | Verdict | ROC-AUC (TF-IDF) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Spandidos** | 22 / 45 [Volatile] | 49,44% | **87,80%** | **SURPASSE la baseline** | 96,97% |
| **Autres (Regroupés)** | 96 / 787 | 19,61% | **60,00%** | **SURPASSE la baseline** | 93,10% |
| **Hindawi (In-Pool)** | 134 / 356 | **42,95%** | 41,83% | **ÉCHOUE face à la baseline** | 63,27% |

Alors que le modèle atteint d'excellents résultats sur Spandidos (F1 de 87,80% et AUC de 96,97%) et sur les autres éditeurs (F1 de 60,00% et AUC de 93,10%), **ses performances sur Hindawi s'effondrent à 41,83% de F1, perdant 1,12 point de pourcentage face à la baseline triviale (42,95%).**

L'analyse de la distribution des probabilités prédites montre un chevauchement quasi-total entre les deux classes pour Hindawi :

*   **Hindawi Positifs (134) :** Probabilité moyenne = 0,2996 | Médiane = 0,2448 | Écart-type = 0,2241 (Q25 = 0,1026, Q75 = 0,4743).
*   **Hindawi Négatifs (356) :** Probabilité moyenne = 0,2137 | Médiane = 0,1234 | Écart-type = 0,2186 (Q25 = 0,0480, Q75 = 0,3096).

Un quart (25%) des articles de paper mills Hindawi confirmés se voient attribuer des probabilités de fraude inférieures à 0,10. À titre de comparaison, pour Spandidos et les autres éditeurs, les médianes des probabilités positives se situent à 0,78 contre moins de 0,07 pour les négatifs. Le modèle n'a presque aucune capacité de discrimination textuelle sur le sous-ensemble Hindawi (AUC de 63,27%).

### 5.2. Analyse qualitative des causes de rétractation
Pour comprendre ce phénomène, nous avons extrait les motifs de rétractation officiels de la base de données Retraction Watch pour tous les cas positifs du corpus d'entraînement et de validation. Les résultats indiquent une différence fondamentale dans la nature même de la fraude commise selon les éditeurs.

#### Motifs de rétractation dominants par éditeur

1.  **Strate Spandidos (Fraude axée sur le contenu) :**
    *   **76,6 %** des notices mentionnent explicitement des duplications d'images (`duplication of/in image`).
    *   **0 %** des notices ne mentionnent de fraude au niveau de la relecture par les pairs.
    *   *Interprétation :* La fraude consiste à réutiliser des figures d'expériences (Western Blots, cytométries) à travers des dizaines de papiers décrivant des gènes différents. Pour habiller ces images volées, les paper mills génèrent des textes méthodologiques et des résultats extrêmement stéréotypés. Ces gabarits répétés laissent des traces lexicales évidentes dans le résumé (ex: les micro-ARN et les structures de phrase spécifiques à Spandidos), détectées facilement par le modèle de texte.
2.  **Strate Hindawi (Fraude axée sur le processus) :**
    *   **100,0 %** des notices mentionnent des manipulations systémiques des processus éditoriaux : fraudes à l'évaluation par les pairs (`compromised peer review` / `peer review rings`) et manipulations de citations (`referencing/attributions`).
    *   **61,9 %** des notices font également référence à du contenu assisté par ordinateur (`computer-aided/computer-generated content`), un phénomène lié à la présence de « tournures torturées » (*tortured phrases*) documentées à grande échelle par Cabanac et al. [3].
    *   **0 %** des notices ne mentionnent de duplication d'images.
    *   *Interprétation :* Les manuscrits Hindawi ont été insérés dans des numéros spéciaux via des réseaux de relecteurs et d'éditeurs invités complices. La fraude réside dans le processus de validation académique et non dans une falsification grossière d'images nécessitant un texte d'accompagnement stéréotypé. Le résumé de ces articles ressemble à du texte scientifique légitime rédigé ou reformulé de manière unique, sans structure lexicale répétitive distinctive.

### 5.3. Validation empirique par test dose-réponse et bootstrap
Afin de tester si la présence de contenu assisté par ordinateur (reformulation automatique par IA) expliquait cette absence de signal (en gommant les signatures lexicales), nous avons divisé les 134 articles positifs Hindawi de validation en deux sous-groupes exclusifs :
*   **Sous-groupe A (Contenu assisté par ordinateur, $N=82$) :** Articles contenant la mention `computer-aided/computer-generated content` dans la base RWDB (incluant les articles ayant à la fois cette mention et des fraudes de relecture).
*   **Sous-groupe B (Autres fraudes de processus, $N=52$) :** Articles rétractés uniquement pour manipulation de relecture/citation sans mention de génération automatique.

Nous avons calculé les intervalles de confiance à 95 % par rééchantillonnage bootstrap (2 000 itérations avec remplacement) de l'AUC pour les deux sous-groupes par rapport aux 356 négatifs Hindawi.

#### Résultats des intervalles de confiance bootstrap de l'AUC

*   **Modèle TF-IDF Combiné :**
    *   *Sous-groupe A (IA, N=82) :* $\text{AUC} = 63,39\%$ | $\text{95% CI} = [57,02\%, 70,05\%]$
    *   *Sous-groupe B (Autre, N=52) :* $\text{AUC} = 63,07\%$ | $\text{95% CI} = [55,96\%, 70,39\%]$
    *   *Recouvrement des CI :* **OUI** (plage commune $[57,02\%, 70,05\%]$, largeur de recouvrement = 13,03 pp)
*   **Modèle PubMedBERT :**
    *   *Sous-groupe A (IA, N=82) :* $\text{AUC} = 59,92\%$ | $\text{95% CI} = [53,33\%, 66,49\%]$
    *   *Sous-groupe B (Autre, N=52) :* $\text{AUC} = 59,78\%$ | $\text{95% CI} = [52,01\%, 67,59\%]$
    *   *Recouvrement des CI :* **OUI** (plage commune $[53,33\%, 66,49\%]$, largeur de recouvrement = 13,15 pp)

**Conclusion de l'analyse causale :** Les intervalles de confiance des deux sous-groupes se superposent de manière quasi complète. Cette absence de différence statistique invalide l'hypothèse selon laquelle la reformulation par ordinateur est la cause principale du blocage.

L'explication suggérée par nos résultats est plus profonde : **les fraudes liées aux réseaux d'évaluation par les pairs et aux cartels de citations n'ont pas laissé de signature détectable par les représentations lexicales et sémantiques testées dans le résumé des articles.** Les textes de ces paper mills sont rédigés avec le même vocabulaire et le même style que les articles scientifiques authentiques. L'échec des modèles n'indique pas nécessairement un défaut de capacité des algorithmes (le modèle PubMedBERT sur embeddings gelés échouant également), mais suggère l'absence de signal accessible aux représentations testées dans les abstracts de ce stratum. Pour détecter ces cas, il est indispensable de sortir du cadre textuel et d'analyser les métadonnées de processus.

### 5.4. Analyse qualitative des erreurs du modèle (Phase 11)
Afin d'identifier les limites opérationnelles et les biais cognitifs du modèle de référence, nous avons extrait un échantillon arbitraire représentatif (correspondant à l'ordre d'indexation dans les fichiers) de 12 erreurs de prédiction issues des ensembles de test et de holdout. L'analyse détaillée de ces cas permet de catégoriser deux grands modes de défaillance.

#### A. Le surapprentissage des gabarits micro-ARN/LncRNA (Faux Positifs)
Le principal facteur de faux positifs (articles scientifiques légitimes classés à tort comme frauduleux, avec des probabilités de fraude variant de 0,53 à 0,71) est l'omniprésence de termes à poids positif extrême liés à la biologie moléculaire des ARN non-codants :
*   `microrna` (poids +8,04), `mir` (poids +5,00), `noncoding` (poids +2,93), `long noncoding` (poids +2,85) et `apoptosis` (poids +3,76).

L'audit des abstracts légitimes faussement détectés montre un recoupement textuel massif avec ces termes :
*   **Case 1 (DOI: 10.1155/2020/8891876) :** Contient littéralement `microrna` (+8,04), `mir` (+5,00) et la bigramme `and invasion` (+4,13) pour décrire la régulation de Twist1.
*   **Case 2 (DOI: 10.1155/2020/8124570) :** Contient `mir` (+5,00), `apoptosis` (+3,76) et `long noncoding` (+2,85) pour décrire l'action du LncRNA HAND2-AS1.
*   **Case 8 (DOI: 10.1002/iub.2012) :** Contient `mir` (+5,00), `apoptosis` (+3,76) et `long noncoding` (+2,85) pour décrire le rôle de MEG3.

*Conclusion :* Pour Spandidos et d'autres éditeurs, la fraude documentaire repose sur des résumés sémantiquement standardisés construits autour de micro-ARN/LncRNA. Le modèle a surappris cette corrélation lexicale, ce qui le conduit à rejeter des études oncologiques réelles et valides sur la régulation par ARN simplement parce qu'elles emploient la terminologie scientifique standard de cette discipline.

#### B. La ressemblance avec le style bioinformatique standard (Faux Négatifs)
Le principal facteur de faux négatifs (articles d'usines à articles classés comme sains, avec des probabilités s'effondrant entre 0,01 et 0,17) est l'adoption d'un style de rédaction calqué sur les analyses de données bioinformatiques cliniques publiques :
*   **Case 4 (DOI: 10.1155/2021/5070099) :** Titré *« Development and Validation of a Novel Mitophagy-Related Gene Prognostic Signature... »* (Probabilité prédite = 0,0581).
*   **Case 5 (DOI: 10.1155/2021/4682589) :** Titré *« Establishing and Validating an Aging-Related Prognostic Four-Gene Signature... »* (Probabilité prédite = 0,0170).

Ces articles frauduleux ne décrivent pas des expériences in vitro / in vivo stéréotypées (les Western blots typiques des paper mills à haute signature lexicale), mais simulent des études pronostiques basées sur des données extraites de bases publiques comme TCGA. Leurs résumés emploient des termes comme `Cox regression`, `Kaplan-Meier`, `TCGA database` et `nomogram`. 

*Conclusion :* Comme ces termes de modélisation clinique possèdent des poids neutres ou négatifs dans le classifieur (étant associés aux articles négatifs légitimes d'oncologie clinique et translationnelle), les paper mills qui génèrent des faux profils d'analyse de données contournent entièrement la détection lexicale en se fondant parfaitement dans la masse des études bioinformatiques.

### 5.5. Évaluation finale sur l'ensemble de test (Phase 9)
La performance finale du modèle TF-IDF Combiné a été mesurée sur l'ensemble de test (`cancer_pm_test.json`), qui n'a été accédé qu'une seule fois. Il est important de préciser que le seuil de décision (0,36) a été gelé sur l'ensemble de validation avant d'accéder aux ensembles de test et de holdout, et qu'il n'a jamais été ré-ajusté par la suite. Au seuil ainsi figé de 0,36, les performances globales et par strate se structurent comme suit :

*   **Métriques globales (Test) :** Précision = 44,05 %, Rappel = 54,37 %, F1 = 48,67 %, ROC-AUC = 79,59 %, PR-AUC = 54,70 %.
*   **Sous-ensemble de test Hindawi (in-pool) ($N_{\text{pos}}=134$, $N_{\text{neg}}=357$) :** Précision = 42,86 %, Rappel = 33,58 %, F1 = 37,66 %, ROC-AUC = 65,71 %.
*   **Sous-ensemble de test Spandidos ($N_{\text{pos}}=18$, $N_{\text{neg}}=43$) [Petit échantillon] :** Précision = 92,86 %, Rappel = 72,22 %, F1 = 81,25 %, ROC-AUC = 98,06 %.
*   **Sous-ensemble de test Autres (regroupés) ($N_{\text{pos}}=100$, $N_{\text{neg}}=788$) :** Précision = 41,15 %, Rappel = 79,00 %, F1 = 54,11 %, ROC-AUC = 89,44 %.

Une comparaison directe avec les résultats de validation montre une baisse modérée de F1 global de **−4,99 pp** (de 53,66 % à 48,67 %), tandis que les ROC-AUC restent extrêmement stables (chute de seulement −1,30 pp). Ce comportement montre une généralisation robuste de la capacité de tri du modèle, malgré une sensibilité de la frontière de décision à la variation des populations.

### 5.6. Évaluation finale de généralisation : Holdout Hindawi (Phase 10)
L'évaluation finale sur le pool de généralisation **Holdout Hindawi** ($N_{\text{pos}}=429$ et $N_{\text{neg}}=1075$, non exposé tout au long du projet) fournit des indices importants concernant les limites documentaires du résumé :

*   **Métriques du Holdout (seuil = 0,36) :** Précision = 46,41 %, Rappel = 36,13 %, F1 = 40,63 %, ROC-AUC = 64,47 %, PR-AUC = 41,24 %.
*   **F1 Baseline Triviale (toujours positif) :** 44,39 %.
*   **Verdict :** **Le modèle de texte échoue face à la baseline triviale** avec une perte de −3,76 pp.

L'analyse de la distribution des probabilités sur ce holdout montre la superposition presque parfaite des classes (médiane positive = 0,2444 vs médiane négative = 0,1120). Ce résultat est cohérent avec l'hypothèse d'une absence de signal exploitable par nos représentations textuelles dans le résumé pour ce type de fraude.

### 5.7. Limites méthodologiques et explications alternatives

Bien que nos résultats indiquent l'absence de signal accessible aux représentations classiques testées (TF-IDF, n-grammes de caractères, embeddings sémantiques gelés) sur la strate Hindawi, cette absence de preuve ne constitue pas une preuve absolue d'absence. Plusieurs explications alternatives et limites méthodologiques doivent être soulignées pour interpréter ces résultats avec rigueur :

1. **Manipulation systémique du processus éditorial (réseaux d'évaluation par les pairs ou peer-review rings) :** Dans le cas d'Hindawi, les réseaux d'usines à articles ont massivement exploité la création de numéros spéciaux (*Special Issues*) en corrompant ou en plaçant des éditeurs invités complices. Parce que la validation académique des manuscrits était garantie par cette corruption administrative (les articles étant acceptés d'office par des complices), les usines à articles n'avaient aucune contrainte de standardisation lexicale ou d'habillage stylistique. Contrairement à Spandidos où les manuscrits devaient franchir un filtre éditorial standard et s'appuyaient donc sur des gabarits rédactionnels répétitifs, les manuscrits Hindawi pouvaient présenter une grande diversité lexicale et ressembler en tout point à des articles scientifiques authentiques.
2. **Déficit d'information sémantique (modalité manquante) :** L'analyse sémantique basée uniquement sur le texte des résumés souffre d'un déficit d'information critique. La détection de la fraude éditoriale et de processus (comme les cartels de citations ou la manipulation d'évaluation par les pairs) nécessite des indicateurs non textuels. L'analyse des délais d'acceptation, les graphes de co-autorat ou les cartels de citations constituent des modalités d'analyse bien plus distinctives qui auraient permis d'identifier immédiatement ces anomalies, là où le texte seul s'avère insuffisant.

---

### 5.8. Robustesse et limites du fine-tuning des transformeurs (PubMedBERT et SciBERT)

Le fait que PubMedBERT et SciBERT fine-tunés atteignent une performance parfaite ($100,00\%$ de F1 et de ROC-AUC) sur toutes les strates de validation, y compris le sous-ensemble Hindawi (où tous les modèles classiques plafonnent à $\sim 63-65\%$ d'AUC), exigeait une investigation rigoureuse. Afin d'écarter l'hypothèse d'un bug de fuite de données ou de mémorisation superficielle, nous avons mené six tests diagnostiques approfondis :

1. **Généralisation sur Holdout Externe :** Nous avons évalué le modèle PubMedBERT fine-tuné (époque 1) sur le jeu de données externe et étanche `cancer_pm_holdout.json` ($N = 1\,504$ articles Hindawi, mis de côté avant partitionnement et resté totalement isolé). Le modèle obtient un score réel exceptionnel de **AUC = 99,88%** et **F1 = 99,06%** (Précision = 100,00%, Rappel = 98,14%). Cela prouve que le modèle a appris des caractéristiques généralisables à l'ensemble du domaine et que la performance n'est pas le produit d'une fuite train/val. Une évaluation sur le jeu de test indépendant (`cancer_pm_test.json`) affirme également un **AUC de 100,00%**.
2. **Audit de non-leakage et distribution des proximités (Holdout Hindawi) :** Bien que l'ensemble Holdout Hindawi ait été extrait par partitionnement stratifié temporel avant l'étape de clustering disjoint des co-auteurs et des titres (Section 2.3), nous avons audité la proximité textuelle entre les 429 positifs Hindawi du Holdout et les 157 positifs Hindawi d'Entraînement. Nous avons mesuré la similarité 3-grammes de titre Jaccard et la similarité cosinus TF-IDF (mots unigrammes et bigrammes) sur titre+résumé pour chaque paire :
   * **Distribution Jaccard (Titre) :** L'écrasante majorité des articles présente des proximités très faibles : $22,14\%$ des articles se situent dans la tranche $[0,1 - 0,2)$, $63,64\%$ dans la tranche $[0,2 - 0,3)$, et $12,82\%$ dans la tranche $[0,3 - 0,4)$. Seuls $1,17\%$ (soit 5 articles) atteignent la classe $[0,4 - 0,5)$, et aucun article ne dépasse la valeur de $0,47$ (Zéro doublon au-dessus du seuil standard de $0,70$).
   * **Distribution cosinus TF-IDF (Titre+Résumé) :** $38,23\%$ des articles se situent dans la tranche $[0,1 - 0,2)$, $51,28\%$ dans la tranche $[0,2 - 0,3)$, $8,62\%$ dans la tranche $[0,3 - 0,4)$, $1,17\%$ dans la tranche $[0,4 - 0,5)$ et seulement $0,47\%$ (2 articles) dans la tranche $[0,5 - 0,6)$. Aucun article ne dépasse une similarité de $0,591$ (Zéro doublon au-dessus de $0,70$).
   Ces distributions, centrées sur une faible similarité lexicale baseline (médiane à $\sim 0,21 - 0,23$), démontrent qu'il n'existe pas de familles d'articles à squelette syntaxique cloné ou de variations par substitution d'entités chimiques/biologiques (*entity-swapping templates*) à cheval entre les partitions de test et d'entraînement.
3. **Absence de biais éditorial sur les contrôles négatifs (Vérification des biais éditoriaux) :** Nous avons comparé la distribution par revue des $1\,075$ contrôles négatifs de la partition Holdout Hindawi et des $1\,069$ contrôles négatifs du pool principal ( Train/Val/Test). Les deux populations présentent une composition quasi identique : `BioMed Research International` représente $37,67\%$ du Holdout vs $39,20\%$ du pool, `CMMM` représente $13,49\%$ vs $13,66\%$, et `Evidence-Based Complementary` représente $16,19\%$ vs $15,06\%$. L'écart maximal (skew) observé toutes revues confondues n'étant que de $1,52\%$, cela exclut l'hypothèse d'une classification opportuniste basée sur des signatures d'éditeurs ou une répartition asymétrique de revues spécifiques.
4. **Séparation nette dans l'espace des scores (Sous-groupe B) :** Pour confirmer que la performance parfaite (AUC de $100,00\%$) obtenue sur le Sous-groupe B ($52$ cas de fraudes purement administratives sans indicateur de génération par IA) n'est pas un artefact statistique dû à la taille de l'échantillon, nous avons inspecté les probabilités individuelles de classification (probabilité prédite après application de la fonction sigmoïde) face aux $356$ contrôles négatifs Hindawi :
   * **Positifs du Sous-groupe B ($N=52$) :** Score minimal = **$0,89102298$**, Score maximal = $0,99525672$, Score médian = $0,99127901$.
   * **Contrôles sains Hindawi ($N=356$) :** Score minimal = $0,00095159$, Score maximal = **$0,00631157$**, Score médian = $0,00149582$.
   * **Marge de séparation :** **$+0,88471138$**.
   Il existe une déconnexion totale des distributions sans aucun chevauchement, garantissant une classification parfaite et robuste quel que soit le seuil choisi dans cet intervalle.
5. **Vérification par permutation des labels :** Nous avons ré-entraîné le modèle PubMedBERT sous un protocole de fine-tuning identique mais en associant à chaque document d'entraînement une étiquette factice générée par permutation aléatoire (désaxant la corrélation avec l'étiquette réelle à seulement $0,0188$). Évalué sur les données de validation, de test et de holdout externe, le modèle obtient respectivement des scores de **AUC = 46,95%**, **AUC = 44,96%** et **AUC = 47,38%**. Ce retour à l'aléa strict ($\sim 50\%$) écarte définitivement l'hypothèse d'une fuite de cibles passive ou de canaux cachés (leakage de métadonnées, doublons géométriques, ou ID internes) au sein de nos pipelines.
6. **Analyse des artefacts positionnels :** L'inspection visuelle des entrées tokenisées brutes confirme que le texte d'entrée est vierge de tout identifiant (PMID, DOI, éditeur), de balises XML/HTML résiduelles, ou d'indications de labellisation. Le modèle opère uniquement sur le texte du titre et du résumé.
7. **Shuffling au niveau des mots vs. des phrases et Analyse d'Attribution :** 
   * *Mélange des mots (Word Shuffling) :* Lorsque l'ordre des mots est entièrement mélangé (détruisant la syntaxe locale mais préservant le sac de mots), l'AUC s'effondre à **67,55%** (rejoignant le plafond des modèles classiques).
   * *Mélange des phrases (Sentence Shuffling) :* Lorsque nous mélangeons l'ordre des phrases au sein du résumé tout en conservant la structure interne et la grammaire locale de chaque phrase, l'AUC reste presque inchangée à **98,37%** (pour PubMedBERT) et **99,83%** (pour SciBERT).
   * *Attribution directe (Leave-One-Out) et Quantification :* Un test d'attribution par omission et par fenêtre glissante sur les vrais positifs du Sous-groupe B a permis d'isoler les marqueurs textuels initiant les prédictions positives fortes ($P \approx 1,0$). Nous avons ensuite quantifié de manière systématique la prévalence de ces signatures à l'aide de filtres par expressions régulières sur l'ensemble des jeux :
     1. *Les transitions structurelles d'assemblage ou d'extraction automatisée (concaténation d'en-têtes de sections) :* Nous avons détecté la présence d'en-têtes (ex. `METHODS:`, `RESULTS:`, `CONCLUSION:`) concaténés à l'intérieur du texte sans retour à la ligne ni ponctuation de fin de phrase correcte (ex. `100 RESULT:`, `pathway. METHOD:`, `metadherin ( CONCLUSION:`). Cette signature de formatage automatisé est présente dans **63,46%** du Sous-groupe B (33 sur 52) et **43,82%** de l'ensemble des positifs Hindawi du Holdout (188 sur 429). En comparaison, ce taux est significativement plus faible dans les jeux de contrôle négatifs sains (26,40% dans la validation Hindawi et 27,44% dans le holdout Hindawi). Bien que ce motif de concaténation inline soit en partie induit par les scripts de conversion XML/Abstract de Hindawi (présent chez $\approx 26-28\%$ des négatifs sains), sa forte surreprésentation chez les positifs (jusqu'à 63,46%) en fait un canal discriminant exploité par le transformeur.
     2. *Les tournures types de squelettes de phrases maladroites :* Les tournures automatisées spécifiques repérées lors de l'audit (ex. `attempted to clarify... on carcinogenic and development`, ou `belongs to a type of the most deadly`) ne concernent que **3,85%** du Sous-groupe B (2 sur 52) et co-occurrent systématiquement avec le premier artefact de concaténation (3,85% d'overlap).
      3. *Détection résiduelle :* De manière révélatrice, **36,54%** des cas du Sous-groupe B (19 sur 52) et **56,18%** de l'ensemble des positifs Hindawi du Holdout (241 sur 429) ne contiennent **ni** l'artefact de concaténation sans ponctuation, **ni** les tournures maladroites précitées. Pourtant, le transformeur fine-tuné les classe correctement avec une probabilité quasi-certaine ($P > 99\%$). Savoir si ce résidu s'explique par l'apprentissage de caractéristiques stylistiques et sémantiques diffuses de la fraude ou par un autre artefact structurel encore non identifié (par exemple des anomalies d'espacement des références bibliographiques ou des structures de ponctuation spécifiques) reste une question ouverte pour de futurs travaux. Cette observation est ainsi cohérente avec l'hypothèse que le transformeur ne se limite pas à surapprendre ces deux anomalies évidentes de surface, mais s'adapte à un ensemble composite de signaux.

*Interprétation scientifique et écart gelé vs fine-tuné :* Ces résultats suggèrent que la performance exceptionnelle des transformeurs fine-tunés ne s'explique pas uniquement par un biais de partitionnement ou par la présence d'un artefact unique de surface. La comparaison avec PubMedBERT gelé (qui plafonne à $65,71\%$ d'AUC sur Hindawi, très proche des $63,27\%$ de TF-IDF) révèle un aspect fondamental de la representation de ces modèles. 

Les représentations neuronales gelées (S-PubMedBERT) sont pré-entraînées sur des tâches sémantiques (recherche documentaire, alignement sémantique). Elles apprennent à condenser des documents entiers en résumant leur contenu conceptuel et en gommant les variations stylistiques locales. Parce que le sujet d'un article d'une usine (ex. traitement du cancer du sein par un micro-ARN) est sémantiquement identique à une étude légitime, la représentation condensée gelée est incapable de les dissocier.

En revanche, le **fine-tuning de bout en bout** réoriente les mécanismes d'attention (*attention heads*) du transformeur. Au lieu de modéliser le sujet sémantique, les têtes s'adaptent pour repérer les micro-structures grammaticales stables, les tournures méthodologiques stéréotypées (ex. *"transwell assay was used to"*, *"luciferase reporter assay was performed"*) et l'agencement syntaxique local propre aux usines. Comme ces modèles repèrent le style local et non les mots isolés, ils restent insensibles aux substitutions de synonymes individuelles qui rendaient inefficients les modèles lexicaux comme les n-grammes de caractères ou de mots.


**Mise en garde pour le déploiement et appel à réplication :** Bien que ce signal stylistique soit réel et généralisable au sein du corpus d'évaluation, cette détection repose sur un **surapprentissage de style** (*style-overfitting*). Si les usines à articles modifient leurs patrons de rédaction ou emploient des générateurs de texte plus avancés, cette signature syntaxique locale disparaîtra, rendant le transformeur inefficace. C'est pourquoi, en conditions réelles, le modèle lexical stable (TF-IDF) demeure la référence de base pour l'évaluation de la robustesse sémantique.

![Comparaison F1 avec modèles fine-tunés](figures/modeles_finetunes.png)
*Figure 10 : Performances comparées incluant les modèles profonds fine-tunés avec mise en garde visuelle sur le risque de surapprentissage stylistique (Section 5.8).*
 Les résultats parfaits des modèles fine-tunés doivent donc être interprétés comme une opportunité de détection ciblée à un instant $t$ de la chaîne de fabrication industrielle, mais appellent à des réplications indépendantes sur des corpus étanches plus récents.

---

## 6. Considérations Éthiques et Limites de Déploiement

Le déploiement d'un tel système de détection dans un environnement éditorial réel soulève un enjeu éthique majeur lié au coût d'un faux positif. L'analyse des erreurs (Section 5.4A) a démontré que le modèle peut surapprendre certaines corrélations lexicales légitimes, telles que la terminologie propre à la biologie moléculaire des ARN non-codants (micro-ARN, LncRNA). En conséquence, le système risque d'assimiler par erreur des recherches légitimes et rigoureuses à de la fraude simplement parce qu'elles partagent des caractéristiques thématiques avec les gabarits préfabriqués des usines à articles.

Le coût humain et professionnel d'une telle erreur est dévastateur : le travail intègre d'un chercheur pourrait être injustement rejeté, ou pire, publiquement accusé de fraude. Les performances finales mesurées sur l'ensemble de test (précision globale de 44,05 %) indiquent clairement que, dans un scénario de production, plus de la moitié des alertes positives générées par le modèle s'avéreront être de fausses alarmes. Par conséquent, ce modèle ne doit en aucun cas être utilisé comme un outil de décision autonome pour rejeter des manuscrits ou publier automatiquement des rétractations. Il est conçu exclusivement comme un instrument de **triage** visant à alerter et orienter l'effort d'une investigation éditoriale humaine qualifiée. Toute intégration opérationnelle devra nécessairement s'accompagner d'une capacité humaine de révision proportionnée au volume d'alertes générées.

---

## 7. Perspectives et Travaux Futurs

L'évaluation a mis en évidence une limite fondamentale des approches purement textuelles : sur certains éditeurs (notamment Hindawi), les articles frauduleux sont textuellement indifférenciables des articles légitimes au niveau des résumés. La fraude s'est matérialisée dans les processus éditoriaux et académiques externes plutôt que dans le contenu rédactionnel. 

Cette section présente trois architectures de détection basées sur des métadonnées non textuelles, conçues comme des pistes de travaux futurs pour dépasser ce plafond de performance, en évaluant de manière réaliste les coûts d'acquisition de données et les verrous techniques associés.

### 7.1. Analyse des délais de relecture éditoriale (Peer-Review Timelines)
Les réseaux de fraude à l'évaluation par les pairs (*peer-review rings*) reposent sur un contournement du processus de relecture. Le relecteur complice (souvent contrôlé par l'usine à articles elle-même) accepte l'article immédiatement sans modifications majeures. 
*   **Hypothèse :** Un délai inhabituellement court entre la soumission, les révisions successives et l'acceptation finale de l'article est corrélé positivement avec une probabilité élevée de rétraction pour évaluation compromise.
*   **Besoins en Données et Faisabilité technique :**
    *   *Données requises :* Dates de soumission (`Received`), de révision (`Revised`) et d'acceptation (`Accepted`).
    *   *Coût d'acquisition :* **TRÈS ÉLEVÉ.** La base de données Retraction Watch (RWDB) ne contient que la date de publication originale et la date de rétraction. Elle ne contient aucune date intermédiaire du cycle éditorial. La base PubMed fournit parfois ces dates dans l'historique XML pour certains éditeurs, mais la couverture est incomplète et incohérente. L'acquisition nécessite le développement de scrapers Web spécifiques pour chaque éditeur afin d'extraire ces dates directement depuis les pages HTML ou les PDF des articles, ce qui représente un coût d'ingénierie et de maintenance substantiel en raison des changements fréquents de mise en page des sites d'éditeurs.

### 7.2. Analyse des réseaux de co-autorat (Co-authorship Graphs)
Les usines à articles vendent des places d'auteurs (souvent des positions de co-auteurs secondaires) à des chercheurs cherchant à augmenter artificiellement leur production académique. Cela crée des associations d'auteurs hautement inhabituelles et déconnectées de la géographie ou de la discipline.
*   **Graphique cible :** Un graphe biparti Auteurs-Articles, projeté ensuite sous forme de graphe de co-autorat où les nœuds sont les auteurs et les arêtes représentent les co-publications.
*   **Mesures de détection :** Détection de communautés (ex. Louvain), coefficient de clustering local anormalement élevé pour des auteurs sans liens institutionnels communs, ou densité anormale de relations d'évaluation complices.
*   **Besoins en Données et Faisabilité technique :**
    *   *Données requises :* Liste complète et unique des auteurs par article.
    *   *Coût d'acquisition :* **ÉLEVÉ (Verrou de la désambiguïsation).** RWDB fournit les noms d'auteurs sous forme de chaînes brutes non structurées (ex: "Zhang, Y.; Wang, L."), sans identifiant unique. L'utilisation de chaînes brutes introduit un bruit colossal en raison des homonymes très fréquents. L'intégration nécessite l'interrogation d'APIs tierces comme ORCID, OpenAlex ou Scopus Author ID pour associer chaque nom à un identifiant unique. Ce processus de désambiguïsation automatique est complexe à fiabiliser à grande échelle, représentant un coût de nettoyage de données majeur.

### 7.3. Analyse des réseaux de citations (Citation Cartels)
Les usines à articles gonflent l'impact de leurs productions en créant des réseaux de citation réciproque (*citation cartels*) ou en insérant des citations totalement hors-contexte dans des articles complices pour satisfaire les demandes de clients ou d'éditeurs corrompus (fraude aux citations).
*   **Hypothèse :** Les articles issus d'usines à articles présentent une déconnexion sémantique entre le contenu du paragraphe de citation et l'article cité, ainsi qu'une structure de graphe de citation fermée (boucles de citations croisées anormalement denses).
*   **Besoins en Données et Faisabilité technique :**
    *   *Données requises :* Liens entrants et sortants de citation (graphe de citations global) et contexte sémantique entourant chaque citation.
    *   *Coût d'acquisition :* **MOYEN À ÉLEVÉ.** La structure globale des citations (qui cite qui) peut être récupérée via l'API OpenAlex ou Crossref. Le coût d'ingénierie est modéré pour construire le graphe de citations de base. En revanche, récupérer le *contexte textuel* de la citation (la phrase précise entourant la citation dans le corps du texte) nécessite l'accès au texte intégral (*full-text*) en XML ou PDF des articles citants. Cela se heurte aux paywalls des éditeurs commerciaux et nécessite des outils complexes d'extraction de PDF (ex: Grobid), rendant l'analyse sémantique des citations très difficile d'accès.

---

## 8. Conclusion du Chapitre

Ce chapitre a présenté une méthodologie rigoureuse pour la détection automatique des usines à articles dans la littérature oncologique. La construction du jeu de données CPM-11K (11 106 articles stables) a permis de jeter des bases solides pour l'apprentissage supervisé, en nettoyant les textes de tout label leakage et en s'immunisant contre les biais d'appariement thématique grâce à une recherche par journal, année et validation lexicale locale.

Le protocole de partitionnement par clusters disjoints a démontré son importance pour éliminer les fuites de gabarits textuels et d'auteurs, révélant au passage des pièges structurels majeurs tels que la percolation des patronymes fréquents dans les signatures d'articles.

Sur le plan de la modélisation, le modèle **TF-IDF Combiné (min_df=5, C=10.0)** s'impose comme la baseline la plus robuste, alliant simplicité, interprétabilité et une performance de premier ordre (**F1 global de 53,66% et ROC-AUC de 80,89%**). Cependant, l'évaluation stratifiée a mis en lumière une asymétrie de performance majeure : alors que les paper mills de Spandidos (caractérisées par des fraudes d'images grossières habillées de textes stéréotypés) sont détectées avec un F1 de 87,80%, les paper mills d'Hindawi échouent face à une prédiction positive naïve.

L'analyse causale par motifs de rétractation et tests dose-réponse bootstrap suggère que cet échec n'est pas uniquement attribuable à l'outil linguistique employé, mais plutôt à la nature même de la fraude. Les fraudes de processus (évaluation par les pairs biaisée, manipulation de citations) n'altèrent pas le contenu rédactionnel des résumés de manière évidente pour les modèles testés. Par conséquent, la détection des paper mills ne pourra probablement pas reposer uniquement sur l'analyse textuelle des seuls résumés à l'avenir. Une approche robuste de détection à grande échelle gagnera à combiner l'analyse lexicale avec des modèles multi-modaux exploitant les métadonnées de processus (délais d'acceptation, graphes de co-autorat et structures de citation).



## Annexes

### Annexe A : Coefficients de pondération du modèle final (TF-IDF Combiné)

Ce tableau présente les 30 termes et n-grammes de mots possédant les coefficients les plus fortement positifs au sein du classifieur régularisé final, déterminant les gabarits de détection d'usines à articles :

| Rang | Terme / n-gramme | Coefficient |
| :---: | :--- | :---: |
| **1** | `microrna` | +8.0424 |
| **2** | `mir` | +5.0050 |
| **3** | `apoptosis and` | +4.5122 |
| **4** | `mimic` | +4.4411 |
| **5** | `hispidulin` | +4.2652 |
| **6** | `and invasion` | +4.1282 |
| **7** | `shrna` | +3.9993 |
| **8** | `image` | +3.9704 |
| **9** | `invasion` | +3.8166 |
| **10** | `viability` | +3.7844 |
| **11** | `apoptosis` | +3.7580 |
| **12** | `research` | +3.7462 |
| **13** | `sponging microrna` | +3.7249 |
| **14** | `inhibits` | +3.6506 |
| **15** | `against` | +3.6428 |
| **16** | `conclusion` | +3.5825 |
| **17** | `assay were` | +3.5393 |
| **18** | `kcnq1ot1` | +3.4164 |
| **19** | `moreover` | +3.3646 |
| **20** | `195` | +3.3150 |
| **21** | `of mir` | +3.2892 |
| **22** | `opioid` | +3.2368 |
| **23** | `paper` | +3.2318 |
| **24** | `mir 195` | +3.2096 |
| **25** | `targeting` | +3.2028 |
| **26** | `and cell` | +3.1961 |
| **27** | `sirt1` | +3.1889 |
| **28** | `mediating` | +3.1855 |
| **29** | `cell viability` | +3.1821 |
| **30** | `enhances` | +3.1632 |

---

## Références

[1] Center for Scientific Integrity. (2018). *Retraction Watch Database*. Retrieved from http://retractiondatabase.org/.

[2] National Library of Medicine. (1946). *Medical Literature Analysis and Retrieval System Online (MEDLINE)*. National Institutes of Health. Available at https://www.nlm.nih.gov/medline/.

[3] Cabanac, G., Labbé, C., & Magazinov, A. (2021). Tortured phrases: A new class of scientific misconduct. *Journal of the Association for Information Science and Technology*, 73(11), 1621-1644. DOI: 10.1002/asi.24578.

[4] Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Chiang, D., ... & Poon, H. (2021). Domain-specific language model pretraining for biomedical natural language processing. *ACM Transactions on Computing for Healthcare*, 3(1), 1-23. DOI: 10.1145/3458750.

[5] Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distortion distillation for pre-training of transformers. *Advances in Neural Information Processing Systems*, 33, 13013-13024.

[6] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 3982-3992. DOI: 10.18653/v1/D19-1410.
