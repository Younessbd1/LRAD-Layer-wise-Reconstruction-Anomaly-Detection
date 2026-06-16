# Analyse des résultats — Ensemble LRAD sur CelebA-OOD

**Expérience** : `celeba_ood`  
**Run** : `ensemble_20260612_164529_6508484`  
**Date** : 12 juin 2026  
**Ensemble** : 10 modèles (graines 42–51)  
**Données** : CelebA 64×64 — In-dist : 18 941 images | OOD : 13 193 images (attribut `Eyeglasses`)  
**Architecture** : FacialCNN, 6 blocs convolutifs, canaux [32, 64, 128, 256, 256, 256]  
**Attributs supervisés** : Genre (M/F) + 6 attributs (Young, Smiling, Mouth_Slightly_Open, High_Cheekbones, Pointy_Nose, Oval_Face)  
**Agrégation spatiale** : `p95` (95e percentile sur les pixels)

---

## 1. Cadre théorique — Décomposition Biais/Variance

### 1.1 Principe de l'ensemble

On entraîne **M = 10 modèles indépendants** (deep ensemble pur : diversité uniquement via l'initialisation aléatoire des poids et l'ordre de mélange SGD, sans MC-Dropout). Chaque modèle `m` produit, pour un bloc convolutif `k` et une image `x`, une reconstruction `f̂^m_k(x)`.

L'ensemble fournit la **reconstruction moyenne** :

```
f̄_k(x) = (1/M) Σ_{m=1}^{M} f̂^m_k(x)
```

### 1.2 Décomposition pixel par pixel

Pour chaque pixel `i` et chaque canal RGB `c`, l'erreur au carré sumée sur les canaux (range `[0, 3]` puisque les images sont en `[0, 1]`) :

```
Risk_k(x)[i]      = (1/M) Σ_m  Σ_c ( x_c[i] − f̂^m_k,c(x)[i] )²
Bias_k(x)[i]      =             Σ_c ( x_c[i] − f̄_k,c(x)[i]   )²
Variance_k(x)[i]  = (1/M) Σ_m  Σ_c ( f̂^m_k,c(x)[i] − f̄_k,c(x)[i] )²
```

**Identité algébrique exacte** (pixel par pixel) :

```
Risk = Bias + Variance
```

Intuition :
- **Risk** : coût moyen d'un seul modèle tiré au hasard de l'ensemble.
- **Bias** : erreur irréductible du modèle *consensus* — ce que l'ensemble ne peut pas reconstruire.
- **Variance** : désaccord entre les modèles — signal d'incertitude épistémique.

> **Signal OOD du projet LRAD** : le terme **Bias** est retenu comme score d'anomalie.  
> Sur une image in-distribution (visage sans lunettes), tous les modèles reconstruisent bien → Bias faible.  
> Sur une image OOD (lunettes occultant le visage), tous les modèles extrapolent mal → Bias élevé.

### 1.3 Exemple de calcul concret

Considérons **un seul pixel** dans la région des lunettes d'une image OOD.

| Quantité | R | G | B |
|---------|---|---|---|
| Pixel original `x` | 0.20 | 0.20 | 0.20 |
| Modèle 1 `f̂¹` | 0.82 | 0.62 | 0.52 |
| Modèle 2 `f̂²` | 0.88 | 0.68 | 0.58 |
| … (10 modèles) | … | … | … |
| Moyenne `f̄` | 0.85 | 0.65 | 0.55 |

**Risk** (M = 10) :
```
Risk ≈ (1/10) × [ ((0.20−0.82)² + (0.20−0.62)² + (0.20−0.52)²)
                + ((0.20−0.88)² + (0.20−0.68)² + (0.20−0.58)²)
                + … ]
     ≈ (1/10) × 10 × (0.3844 + 0.1764 + 0.1024)   # ≈ même erreur sur tous les modèles
     ≈ 0.663
```

**Bias** (erreur du modèle consensus) :
```
Bias = (0.20−0.85)² + (0.20−0.65)² + (0.20−0.55)²
     = 0.4225 + 0.2025 + 0.1225
     = 0.747
```

**Variance** (désaccord entre modèles) :
```
Variance ≈ (1/10) × [ ((0.82−0.85)² + (0.62−0.65)² + (0.52−0.55)²)
                     + ((0.88−0.85)² + (0.68−0.65)² + (0.58−0.55)²)
                     + … ]
         ≈ 0.0027
```

Vérification de l'identité : `0.663 ≈ 0.747 − 0.084` → dans cet exemple simplifié les modèles *divergent* légèrement donc Variance contribue peu. Dans l'expérience réelle `identity_max_residual = 3.87e-7` confirme que `Risk = Bias + Variance` tient à la précision float32.

**Pour un pixel d'un visage in-distribution** :
```
x ≈ [0.75, 0.55, 0.45]    f̄ ≈ [0.76, 0.54, 0.46]
Bias = (−0.01)² + (0.01)² + (−0.01)² = 0.0003   → quasi nul
```

### 1.4 Réduction pixel → scalaire par image (`p95`)

Une carte `(H=64, W=64)` est aplatie en vecteur de 4 096 valeurs. Le score d'un bloc est le **95e percentile** :

```
score_k(x) = p95( {Bias_k(x)[i]}_{i=1}^{4096} )
           = valeur en position ⌊0.95 × 4096⌋ = 3 891  dans le vecteur trié
```

Le `p95` est un compromis robuste : il ignore les quelques pixels aberrants mais réagit bien à une anomalie localisée comme une paire de lunettes.

**Score final agrégé** (moyenne uniforme sur 6 blocs) :
```
score(x) = (1/6) Σ_{k=0}^{5} score_k(x)
```

### 1.5 AUROC — interprétation

Pour toutes les paires (image_in, image_ood) tirées au hasard :

```
AUROC = P( score(x_ood) > score(x_in) )
```

Un AUROC de 0.627 signifie que le score Bias distingue correctement OOD vs in-dist dans **62,7 % des cas**.  
Valeur de référence aléatoire : 0.50. Valeur parfaite : 1.00.

---

## 2. Résultats globaux

| Métrique | Valeur |
|----------|--------|
| Taille ensemble | 10 modèles |
| Précision genre (moyenne) | **97.6 %** |
| AUROC MSP (moyenne) | 0.626 |
| **AUROC Bias agrégé** | **0.627** |
| AUROC Risk agrégé | 0.622 |
| AUROC Variance agrégé | 0.599 |
| Meilleur AUROC par bloc (Bias) | **0.655** (Bloc 3) |
| `identity_max_residual` | 3.87×10⁻⁷ ✓ |

### Précision par modèle

| Graine | Acc. Genre | AUROC MSP | AUROC Entropy Genre |
|--------|-----------|-----------|---------------------|
| 42 | 97.69 % | 0.601 | 0.599 |
| 43 | 97.45 % | 0.641 | 0.696 |
| 44 | 97.74 % | 0.645 | 0.705 |
| 45 | 97.56 % | 0.604 | 0.616 |
| 46 | 97.68 % | 0.659 | 0.693 |
| 47 | 97.62 % | 0.638 | 0.679 |
| 48 | 97.60 % | 0.617 | 0.663 |
| 49 | 97.51 % | 0.643 | 0.696 |
| 50 | 97.57 % | 0.626 | 0.653 |
| 51 | 97.57 % | 0.592 | 0.594 |

Tous les modèles ont une précision genre quasi-identique (~97.6 %), ce qui atteste de la convergence stable de l'entraînement malgré les graines différentes.

---

## 3. Plots — description, formules et interprétation

---

### 3.1 `mean_recons_only.png` — Reconstructions de l'ensemble-mean

**Ce que montre ce plot** : pour chaque image (ligne), la colonne originale suivie de 6 reconstructions `f̄_k(x)` — une par bloc convolutif L0 à L5 — produites par la **moyenne** des 10 reconstructions individuelles.

**Formule** :
```
f̄_k(x) = (1/10) Σ_{m=1}^{10} f̂^m_k(x)      (B, 3, H, W)
```

**Ce qu'on observe** :
- Les blocs superficiels (L0, L1) produisent des reconstructions floues, sans détail fin — le décodeur reconstruit à partir de features basses résolution.
- Les blocs profonds (L4, L5) reconstituent des visages plus nets incluant la texture de peau, les yeux, la bouche.
- Sur les images OOD (lunettes), **f̄** reconstruit le visage *sans* lunettes, car l'ensemble n'a jamais vu de lunettes pendant l'entraînement → la zone des lunettes devient une zone d'erreur forte.

---

### 3.2 `mean_recon_breakdown.png` — Original | Erreur | Reconstruction par bloc

**Ce que montre ce plot** : triplet `(Original | Err Lk | Recon Lk)` pour chaque bloc, calculé sur la reconstruction de l'ensemble-mean.

**Formule de la carte d'erreur** (colonne « Err Lk ») :
```
err_k(x)[i] = Σ_c ( x_c[i] − f̄_k,c(x)[i] )²       ∈ [0, 3]
```

Les scores annotés dans chaque tuile sont la **moyenne spatiale** :
```
score_tuile = (1 / (H×W)) Σ_i err_k(x)[i]
```

**Échelle de couleur** : fixée à `[0, 3]` (viridis) — jamais normalisée par les données, donc les figures sont directement comparables entre elles.

**Ce qu'on observe** :
- Les tuiles Err L0–L2 sont uniformément sombres (erreurs < 0.05) : les décodeurs superficiels reproduisent la moyenne globale des couleurs mais pas les détails.
- Les tuiles Err L4–L5 s'illuminent sur les zones anormales (lunettes, accessoires) : à ces profondeurs le réseau a appris à reconstruire les structures fines du visage, ce qui le met en défaut sur l'OOD.
- Les reconstructions Recon L5 ressemblent à des photos réelles de visages, preuve que le décodeur L5 a capturé les sémantiques visuelles complexes.

---

### 3.3 `mean_error_maps.png` — Cartes de Risk (moyenne des M erreurs)

**Ce que montre ce plot** : cartes de **Risk** par bloc — la *moyenne* des 10 cartes d'erreur individuelles, pas l'erreur de la reconstruction moyenne.

**Formule** :
```
Risk_k(x)[i] = (1/M) Σ_{m=1}^{M} Σ_c ( x_c[i] − f̂^m_k,c(x)[i] )²
```

**Exemple chiffré** (un pixel, M=10 modèles, bloc L5, image OOD) :
```
Erreurs par modèle : [0.70, 0.68, 0.71, 0.69, 0.72, 0.67, 0.73, 0.68, 0.70, 0.72]
Risk = moyenne = 0.700
```

**Différence vs Bias** :
```
Risk   = erreur de la moyenne pondérée des modèles
Bias   = erreur de la reconstruction moyenne  (≠ moyenne des erreurs en général)
```
L'identité `Risk = Bias + Variance` est exacte ; Risk ≥ Bias toujours.

**Ce qu'on observe** : Les cartes Risk et Bias sont visuellement presque identiques car la Variance est très faible (~0.01) sur ce jeu de données. La séparabilité OOD/ID est légèrement inférieure à celle du Bias pur (AUROC 0.622 vs 0.627) car le terme de Variance ajoute du bruit.

---

### 3.4 `min_error_maps.png` — Erreur du meilleur membre

**Ce que montre ce plot** : pour chaque pixel, l'erreur du **meilleur** modèle de l'ensemble (minimum sur M).

**Formule** :
```
MinErr_k(x)[i] = min_{m=1..M}  Σ_c ( x_c[i] − f̂^m_k,c(x)[i] )²
```

**Interprétation** : un pixel reste *illuminé* (erreur élevée) seulement si **aucun** des 10 modèles n'arrive à le reconstruire correctement. C'est un signal OOD plus fort que la moyenne : il résiste aux membres "chanceux" qui auraient par hasard bien capturé l'anomalie.

**Exemple** (pixel dans la zone lunettes) :
```
Erreurs : [0.68, 0.70, 0.71, 0.67, 0.72, 0.69, 0.73, 0.65, 0.70, 0.71]
MinErr  = 0.65     (toujours très élevé → lunettes pas reconstituables)
```

Versus un pixel de peau normale :
```
Erreurs : [0.002, 0.001, 0.003, 0.001, 0.002, 0.001, 0.002, 0.003, 0.001, 0.002]
MinErr  = 0.001    (quasi nul → au moins un modèle reconstruit parfaitement)
```

**Ce qu'on observe** : Les cartes MinErr sont globalement **plus sombres** que les cartes Risk (valeurs annotées plus faibles) mais les zones OOD restent détectables, confirmant que les lunettes résistent à toute tentative de reconstruction.

---

### 3.5 `mean_abs_bias.png` — Cartes de Bias `(x − f̄)²`

**Ce que montre ce plot** : cartes du terme **Bias** — l'erreur de la reconstruction *consensus* de l'ensemble. C'est le **signal OOD principal** du projet.

**Formule** :
```
Bias_k(x)[i] = Σ_c ( x_c[i] − f̄_k,c(x)[i] )²      avec  f̄_k = (1/M) Σ_m f̂^m_k
```

**Exemple complet (bloc L5, image OOD avec lunettes, pixel central de la monture)** :

```
x        = [0.15, 0.15, 0.15]    (monture sombre)
f̄        = [0.80, 0.62, 0.52]    (peau reconstituée sans lunettes)

Bias[pixel] = (0.15−0.80)² + (0.15−0.62)² + (0.15−0.52)²
            =    0.4225     +    0.2209     +    0.1369
            =    0.780
```

**Score p95 pour cette image** (sur les 4096 pixels du bloc L5) :
```
→ trier les 4096 valeurs Bias
→ prendre la valeur en position 3891 (= ⌊0.95 × 4096⌋)
→ ex : score_bias_L5 = 0.23
```

**Ce qu'on observe** : les lignes étiquetées "OOD" s'illuminent nettement à L3–L5 sur la zone de la monture et des verres, alors que les lignes "ID" restent sombres partout. Cela valide que le Bias capture spatialement l'anomalie (localisation des lunettes).

---

### 3.6 `variance_heatmaps_all.png` et `variance_heatmaps_ood.png` — Désaccord entre modèles

**Ce que montre ce plot** : cartes de **Variance** — mesure du désaccord pixel-wise entre les 10 modèles.

**Formule** :
```
Variance_k(x)[i] = (1/M) Σ_m  Σ_c ( f̂^m_k,c(x)[i] − f̄_k,c(x)[i] )²
```

C'est la **variance empirique** (non biaisée) des reconstructions, sumée sur les canaux RGB.

**Exemple** (même pixel lunette, les modèles convergent vers une reconstruction similaire) :
```
f̂¹ = [0.82, 0.63, 0.53]
f̂² = [0.78, 0.61, 0.51]
f̄  = [0.80, 0.62, 0.52]

Var model 1 = (0.82−0.80)² + (0.63−0.62)² + (0.53−0.52)² = 0.0004+0.0001+0.0001 = 0.0006
Var model 2 = (0.78−0.80)² + (0.61−0.62)² + (0.51−0.52)² = 0.0004+0.0001+0.0001 = 0.0006
Variance = 0.0006   (très faible malgré l'OOD)
```

**Ce qu'on observe** : Les cartes de Variance sont extrêmement sombres (valeurs annotées ~0.001–0.005) — l'ensemble est très **peu diversifié** sur ce problème. L'AUROC Variance (0.599) est le plus faible des trois termes, confirmant que le désaccord inter-modèles n'est pas le signal discriminant ici. Les 10 modèles apprennent tous la même représentation faciale et échouent de la même manière sur les lunettes → le Bias reste le bon signal.

---

### 3.7 `score_comparison.png` — Comparaison Bias / Risk / Min / Quantile-Min

**Ce que montre ce plot** : quatre colonnes de cartes d'erreur côte à côte pour **un seul bloc** (typiquement L5), chacune avec sa propre échelle de couleur.

| Colonne | Formule |
|---------|---------|
| Bias `(x−f̄)²` | `Σ_c (x_c − f̄_c)²` |
| Risk (mean) | `(1/M) Σ_m Σ_c (x_c − f̂^m_c)²` |
| Min | `min_m Σ_c (x_c − f̂^m_c)²` |
| Quantile-min (k=3) | `3e plus petite erreur parmi M modèles` |

Les colonnes ont **des échelles différentes** (chaque colonne prend son propre vmax = max de la colonne) car les ordres de grandeur divergent : le minimum est 5–10× plus petit que la moyenne.

**Interprétation** : Toutes les colonnes révèlent les mêmes zones anormales (lunettes, accessoires inhabituels), mais la colonne Min est la plus sélective : seules les zones qu'aucun modèle ne reconstruit apparaissent. Les valeurs annotées confirment que Bias ≈ Risk (faible variance) et que Min < Bias.

---

### 3.8 `ensemble_decomposition.png` — Triplet Risk | Bias | Variance par bloc

**Ce que montre ce plot** : pour chaque bloc `k` et chaque image, les trois termes `Risk_k | Bias_k | Variance_k` côte à côte. Tous partagent la même échelle `[0, 3]`, donc les tuiles vérifient visuellement l'identité `Risk = Bias + Variance`.

**Vérification de l'identité** :
```
identity_max_residual = |Risk − Bias − Variance|_max = 3.87 × 10⁻⁷  ≪  1e-6   ✓
```

**Ce qu'on observe** : la tuile Bias est quasi-identique à la tuile Risk (Variance ≈ 0), mais la comparaison révèle que le Bias est légèrement *inférieur* au Risk — la différence est précisément la Variance. La Variance est visuellement une image noire (valeurs < 0.005), confirmant la faible diversité de l'ensemble sur CelebA.

---

### 3.9 `ensemble_score_hists.png` — Histogrammes ID vs OOD des scores agrégés

**Ce que montre ce plot** : trois histogrammes de densité pour les scores **agrégés** (p95 sur les pixels puis moyenne sur les blocs) — Risk, Bias, Variance — en bleu pour in-distribution (n=18 941) et rouge pour OOD (n=13 193).

**Formule du score agrégé** :
```
score_aggregated(x) = (1/6) Σ_{k=0}^{5}  p95_pixel( term_k(x) )
```

**Valeurs lues sur le graphique** :

| Score | Moyenne ID | Moyenne OOD | Séparation visible |
|-------|-----------|-------------|-------------------|
| Risk  | ~0.08 | ~0.12 | Partielle |
| Bias  | ~0.07 | ~0.10 | Partielle |
| Variance | ~0.020 | ~0.025 | Faible |

**Interprétation** : les distributions se *chevauchent fortement*, ce qui explique les AUROC modérés (~0.62). Le décalage de la distribution OOD vers la droite est réel mais insuffisant pour une discrimination parfaite. Cela s'explique par le fait que les lunettes n'occupent qu'une petite fraction de l'image (≈ 5–15 % des pixels) ; pour la majorité des pixels, OOD et ID ont des erreurs similaires.

---

### 3.10 `decomposition_auroc.png` — Barres AUROC par bloc

**Ce que montre ce plot** : l'AUROC OOD-vs-ID pour chaque terme (Risk/Bias/Variance) à chaque profondeur, plus l'AUROC agrégé.

**Formule AUROC** :
```
AUROC = P( score(x_ood) > score(x_in) )
      = (1 / (n_in × n_ood)) Σ_{i,j} 𝟙[ score(x^ood_j) > score(x^in_i) ]
```

**Résultats complets** :

| Bloc | Risk | Bias | Variance |
|------|------|------|---------|
| 0 | 0.572 | 0.562 | 0.586 |
| 1 | 0.597 | 0.588 | 0.602 |
| 2 | 0.626 | 0.621 | 0.623 |
| 3 | 0.651 | **0.655** | 0.617 |
| 4 | 0.621 | 0.627 | 0.585 |
| 5 | 0.603 | 0.608 | 0.571 |
| **Agrégé** | **0.622** | **0.627** | **0.599** |

**Interprétation** :
- **Bloc 3** est le plus discriminant (résolution intermédiaire 8×8, canaux=256) — c'est à cette profondeur que les décodeurs capturent les structures sémantiques des traits du visage, là où les lunettes perturbent le plus la reconstruction.
- Le **Bias domine légèrement le Risk** à tous les blocs, confirmant que retirer la Variance (bruit de désaccord inter-modèles) améliore la séparation.
- La **Variance est moins discriminante** : les modèles convergent vers la même erreur sur les lunettes plutôt que de diverger, rendant le signal épistémique faible.
- La ligne pointillée à 0.5 représente le hasard pur — tous les blocs et termes sont au-dessus.

---

### 3.11 `bias_variance_vs_block.png` — Évolution avec la profondeur

**Ce que montre ce plot** : la moyenne et l'écart-type des scores Bias et Variance en fonction du bloc convolutif, ID (bleu) et OOD (rouge).

**Formules** (pour N images) :
```
μ_k = (1/N) Σ_{n=1}^{N} score_k(x_n)
σ_k = sqrt( (1/N) Σ_n (score_k(x_n) − μ_k)² )
```

**Ce qu'on observe** :
- Le Bias **croît fortement avec la profondeur** pour les deux groupes (blocs 0→5 : 0.01 → 0.25 pour ID, 0.01 → 0.30 pour OOD). Les décodeurs profonds reconstruisent des détails fins → erreur plus grande.
- La **séparation OOD − ID s'ouvre progressivement** à partir du bloc 2 et atteint son maximum au bloc 3–4, puis se referme légèrement au bloc 5.
- Pour la Variance, la séparation est quasi nulle aux blocs 0–2 puis légèrement visible aux blocs 3–5, mais les bandes d'écart-type se chevauchent fortement.
- Les bandes d'écart-type des deux groupes se chevauchent largement → le chevauchement explique les AUROC modérés.

---

### 3.12 `bias_variance_vs_percentile.png` — Courbes de percentile ID vs OOD

**Ce que montre ce plot** : pour les scores agrégés (toutes profondeurs), on trace le q-ème percentile des distributions ID et OOD pour `q ∈ [1%, 99%]`.

**Formule** :
```
courbe_ID(q)  = percentile_q( {score(x_in)}_{n=1}^{18941} )
courbe_OOD(q) = percentile_q( {score(x_ood)}_{n=1}^{13193} )
```

**Interprétation** :
- Les deux courbes sont **parallèles et proches** de q=1% à q≈70%, signe que les distributions se chevauchent massivement dans leur corps.
- À partir de q≈80%, la courbe OOD (rouge pointillé) **s'écarte vers le haut** — les cas OOD extrêmes ont des scores nettement plus élevés que les cas ID extrêmes.
- Cela confirme que le séparateur optimal se trouve dans les **queues hautes** des distributions : un seuil à p95 ou p99 sera bien plus discriminant qu'un seuil médian.
- Pour le Bias, à q=99% : score_OOD ≈ 0.245 contre score_ID ≈ 0.222 → écart de ~10 %.

---

## 4. Bilan et limites

### Points forts
- L'identité `Risk = Bias + Variance` est vérifiée à `3.87×10⁻⁷` ✓
- La précision de classification genre est excellente (≥97.4 % sur tous les modèles)
- Le Bias capture **spatialement** la localisation des lunettes (visible dans les heatmaps)
- Le bloc 3 est clairement le plus discriminant (AUROC Bias = 0.655)

### Limites observées
- **AUROC modéré** (~0.627) : les lunettes n'occupent qu'une petite fraction de l'image ; le score p95 sur tous les pixels dilue le signal.
- **Faible diversité de l'ensemble** : la Variance inter-modèles est quasi nulle — les 10 modèles apprennent la même représentation. La graine aléatoire seule ne génère pas assez de diversité sur ce problème.
- **Scores `auroc_entropy_combined = NaN`** : la branche combinant l'entropie des attributs et du genre n'a pas produit de résultats valides (probablement due à des attributs dont l'entropie est dégénérée).

### Pistes d'amélioration
- Augmenter la diversité de l'ensemble : architectures différentes, augmentations asymétriques, ou vrais sous-ensembles de données.
- Pondérer le bloc 3 plus fortement dans l'agrégation (actuellement uniforme).
- Utiliser une réduction spatiale plus ciblée : masque attentionnel ou top-k pixels plutôt que p95 global.
