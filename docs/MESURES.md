# MESURES — le rapport détaillé de toutes les mesures OOD du projet

*Dernière mise à jour : 18 juillet 2026 — couvre les runs `ensemble_20260707_152254_6754917` (baseline, 10 modèles 64 px), `ensemble_20260716_170930_6774881` (validation splits complets) et `gridsearch_6781780` (recherche d'hyperparamètres CutPaste).*

Ce document explique **chaque mesure** calculée dans le projet : sa formule, son
implémentation, son interprétation, et un **exemple numérique complet** à chaque
fois. Il se lit de haut en bas : chaque section s'appuie sur les précédentes.

## Sommaire

1. [Notation et vue d'ensemble](#1-notation-et-vue-densemble)
2. [L'AUROC — la métrique de toutes les mesures](#2-lauroc)
3. [Scores du classifieur : MSP et entropies](#3-scores-du-classifieur)
4. [La décomposition Risque = Biais + Variance](#4-la-décomposition-risque--biais--variance)
5. [Incertitude prédictive : totale / aléatoire / épistémique](#5-incertitude-prédictive)
6. [L'énergie](#6-lénergie)
7. [Score localisé : z-score par pixel + patch-max](#7-score-localisé)
8. [locfre — l'erreur de features localisée](#8-locfre)
9. [Fusion par rang](#9-fusion-par-rang)
10. [Fusion supervisée (régression logistique)](#10-fusion-supervisée)
11. [CutPaste et le signal cutpaste_prob](#11-cutpaste)
12. [Le grid search multi-métriques](#12-le-grid-search-multi-métriques)
13. [Historique des résultats](#13-historique-des-résultats)
14. [Où vit chaque mesure dans le code](#14-où-vit-chaque-mesure-dans-le-code)

---

## 1. Notation et vue d'ensemble

- `x` : une image d'entrée `(3, H, W)`, pixels dans `[0, 1]`.
- `M` : la taille de l'ensemble (10 modèles au run de référence, 8 pour cp128).
- `f̂ᵐ(x)` : la reconstruction de `x` par les décodeurs du membre `m`
  (un décodeur par bloc convolutif ; `f̂ᵐₖ` = reconstruction au bloc `k`).
- `f̄(x) = (1/M) Σₘ f̂ᵐ(x)` : la reconstruction **moyenne** de l'ensemble
  (le « consensus »).
- `aⱼᵐ(x)` : les activations du bloc `j` du tronc du membre `m`.
- **in-dist** : visages sans lunettes (label 0) ; **OOD** : visages avec
  lunettes (label 1). L'objectif de chaque *score* est d'être **plus grand
  sur les OOD**.

Le pipeline complet :

```
                    ┌── têtes classif ──► MSP, entropies, énergie, incertitudes   (§3, §5, §6)
image ──► tronc ────┤
                    ├── tête CutPaste ──► cutpaste_prob                            (§11)
                    │
                    └── activations ──► décodeurs ──► reconstructions
                                                        │
                          ┌─────────────────────────────┤
                          ▼                             ▼
              Risque = Biais + Variance        erreur de features (locfre)
              + z-score + patch-max (§4, §7)          (§8)
                          │                             │
                          └───────────┬─────────────────┘
                                      ▼
                        FUSION (rang §9, supervisée §10) ──► score final
```

Chaque score produit **un scalaire par image**. On mesure sa qualité avec
l'AUROC (§2) : on calcule le score sur toutes les images de `test_in` et de
`test_ood`, et on regarde s'il classe bien les OOD au-dessus.

---

## 2. L'AUROC

### 2.1 Définition

L'AUROC (*Area Under the ROC Curve*) d'un score `s` est :

```
AUROC = P( s(X_ood) > s(X_in) )
```

la probabilité qu'une image OOD tirée au hasard reçoive un score **plus
grand** qu'une image in-dist tirée au hasard. C'est un nombre entre 0 et 1 :

| AUROC | Lecture |
|---|---|
| 0.5 | le score est aléatoire — aucune information |
| 0.64 | 64 paires sur 100 bien classées — signal faible |
| 0.80 | signal exploitable |
| 0.90+ | détecteur de qualité opérationnelle |
| < 0.5 | le score est **inversé** (informatif mais dans le mauvais sens) |

### 2.2 Le calcul par comptage de paires — exemple complet

3 images normales (A, B, C) et 2 images OOD (D, E), score de chacune :

| Image | Type | Score |
|---|---|---|
| A | in | 0.2 |
| B | in | 0.5 |
| C | in | 0.9 |
| D | ood | 0.6 |
| E | ood | 0.8 |

On forme **toutes les paires** (in, ood) — il y en a `3 × 2 = 6` — et on
compte celles où l'OOD gagne :

```
(A,D): 0.2 < 0.6 ✓     (A,E): 0.2 < 0.8 ✓
(B,D): 0.5 < 0.6 ✓     (B,E): 0.5 < 0.8 ✓
(C,D): 0.9 > 0.6 ✗     (C,E): 0.9 > 0.8 ✗

AUROC = 4/6 ≈ 0.67
```

(En cas d'égalité exacte des deux scores, la paire compte pour ½.)
Ici, l'image C — une normale au score anormalement haut, p. ex. un visage
aux cheveux mal reconstruits — coûte à elle seule 2 paires. C'est
**exactement** le mécanisme qui limitait le score p95 global à 0.64.

### 2.3 La courbe ROC

Équivalence : pour chaque seuil `t`, on classe « OOD » toute image avec
`s > t`, et on trace le point (taux de faux positifs, taux de vrais
positifs) :

```
TPR(t) = #{ood : s > t} / #ood        FPR(t) = #{in : s > t} / #in
```

En faisant varier `t` de +∞ à −∞ on obtient une courbe de (0,0) à (1,1) ;
**l'aire sous cette courbe est l'AUROC** — les deux définitions coïncident.
La diagonale `y = x` est le hasard. C'est ce que trace le panneau du milieu
de `plots/fused_auroc.png`.

### 2.4 Propriétés importantes

1. **Invariance monotone** : appliquer une fonction croissante au score
   (`log`, `exp`, ×1000…) ne change pas l'AUROC — seul **l'ordre** compte.
   C'est ce qui autorise la fusion par rang (§9).
2. **Pas de seuil** : l'AUROC juge le classement entier, pas une décision.
3. Dans le code : `sklearn.metrics.roc_auc_score(labels, scores)` avec
   `labels = [0]*n_in + [1]*n_ood` — voir `_auroc_entry` dans
   [lrad/evaluate.py](../lrad/evaluate.py).

---

## 3. Scores du classifieur

Le classifieur a une tête *gender* (softmax 2 classes) et une tête *attrs*
(6 sigmoïdes). L'intuition commune : **sur une image OOD, le modèle hésite**.

### 3.1 MSP — Maximum Softmax Probability

```
score_msp(x) = 1 − max_c p(c|x)
```

Exemple : `p = (0.98, 0.02)` (très sûr) → `score = 0.02`.
`p = (0.55, 0.45)` (hésite) → `score = 0.45`. Plus le modèle hésite,
plus le score monte.

### 3.2 Entropie de la tête gender (softmax)

```
H(p) = − Σ_c p(c) · ln p(c)
```

Exemple : `p = (0.9, 0.1)` → `H = −0.9·ln 0.9 − 0.1·ln 0.1 = 0.095 + 0.230
= 0.325` nats. `p = (0.5, 0.5)` → `H = ln 2 ≈ 0.693` (le maximum).
L'entropie mesure l'hésitation de façon plus fine que le MSP.

### 3.3 Entropie des attributs (Bernoulli)

Pour chaque attribut de probabilité `p` :

```
H_b(p) = −p·ln p − (1−p)·ln(1−p)
```

moyennée sur les 6 attributs. **Piège numérique résolu dans le code** : en
float32, `1 − 10⁻¹²` s'arrondit à `1.0`, donc un attribut saturé (`p = 1.0`)
produisait `ln(0) = −∞` et un score NaN. La version stable calcule
l'entropie **depuis les logits** : `H = p·softplus(−z) + (1−p)·softplus(z)`
(voir `_bernoulli_entropy_logits` dans [lrad/evaluate.py](../lrad/evaluate.py)).

### 3.4 Résultats mesurés (run de référence, par membre)

| Score | AUROC (meilleur membre / pire) |
|---|---|
| entropie gender | 0.79 / 0.45 |
| MSP | 0.76 / 0.44 |
| entropie attrs | ~0.50 partout (bruit pur) |

Leçon : l'entropie attrs ne détecte rien (les 6 attributs restent
prédictibles avec des lunettes) et **diluait** le score combiné — d'où le
passage aux fusions qui pèsent chaque signal.

---

## 4. La décomposition Risque = Biais + Variance

### 4.1 Les formules

Pour un bloc `k`, un pixel `i` (erreur L2 **sommée** sur les 3 canaux RGB,
donc un pixel vit dans `[0, 3]`) :

```
Risque_k(x)[i]   = (1/M) Σₘ ( x[i] − f̂ᵐₖ(x)[i] )²     erreur moyenne des membres
Biais_k(x)[i]    = ( x[i] − f̄ₖ(x)[i] )²               erreur du consensus
Variance_k(x)[i] = (1/M) Σₘ ( f̂ᵐₖ(x)[i] − f̄ₖ(x)[i] )²  désaccord des membres
```

avec l'**identité algébrique exacte**, pixel par pixel :

```
Risque = Biais + Variance
```

(vérifiée dans le code à ~2×10⁻⁷ près, le bruit float32).

### 4.2 Exemple numérique

Un pixel, 3 modèles. La vraie valeur est `x = 0.8`. Les reconstructions :
`f̂¹ = 0.5`, `f̂² = 0.6`, `f̂³ = 0.7`.

```
f̄ = (0.5 + 0.6 + 0.7)/3 = 0.6

Risque   = [(0.8−0.5)² + (0.8−0.6)² + (0.8−0.7)²]/3
         = [0.09 + 0.04 + 0.01]/3          = 0.0467
Biais    = (0.8 − 0.6)²                    = 0.04
Variance = [(0.5−0.6)² + (0.6−0.6)² + (0.7−0.6)²]/3
         = [0.01 + 0 + 0.01]/3             = 0.0067

Vérification : 0.04 + 0.0067 = 0.0467 ✓
```

Interprétation : le **Biais** est ce qu'aucun membre ne sait reconstruire
(l'anomalie — des lunettes absentes du train) ; la **Variance** est là où
les membres divergent (l'incertitude épistémique).

### 4.3 Du pixel au score par image : mean / max / p95

La carte `(H, W)` est réduite en un scalaire par :

- `mean` — sensible à toute l'image (dilue les anomalies locales) ;
- `max` — le pixel le plus surprenant (sensible au bruit) ;
- `p95` — le 95ᵉ percentile, compromis robuste. **C'était le score
  historique** ; ses AUROC : biais 0.638, risque 0.636, variance 0.627.

**Pourquoi p95 plafonne** : les lunettes couvrent ~500 pixels sur 4096
(64×64) ≈ 12 % de l'image dans le meilleur cas, souvent moins. Le p95
regarde « les 5 % de pixels les pires » — or les cheveux et le fond
fournissent déjà des milliers de pixels à erreur haute sur **chaque**
visage normal. Le pic local des lunettes bouge à peine ce percentile
(in_mean ≈ 0.95 vs ood_mean ≈ 1.10, +15 % seulement).

---

## 5. Incertitude prédictive

On applique la même décomposition aux **probabilités** des têtes de
classification (pas aux pixels). Pour la tête gender :

```
Totale     = H( (1/M) Σₘ pᵐ )        entropie de la prédiction moyenne
Aléatoire  = (1/M) Σₘ H(pᵐ)          moyenne des entropies individuelles
Épistémique = Totale − Aléatoire      ≥ 0, par concavité de H
```

### Exemple numérique (2 modèles)

**Cas 1 — les modèles sont d'accord et hésitent** (incertitude aléatoire) :
`p¹ = (0.6, 0.4)`, `p² = (0.6, 0.4)`.

```
moyenne = (0.6, 0.4)      Totale     = H(0.6, 0.4) = 0.673
                          Aléatoire  = [H(0.6,0.4) + H(0.6,0.4)]/2 = 0.673
                          Épistémique = 0.673 − 0.673 = 0        ← désaccord nul
```

**Cas 2 — les modèles sont sûrs mais se contredisent** (épistémique) :
`p¹ = (0.9, 0.1)`, `p² = (0.1, 0.9)`.

```
moyenne = (0.5, 0.5)      Totale     = H(0.5, 0.5) = 0.693
                          Aléatoire  = [H(0.9,0.1) + H(0.1,0.9)]/2 = 0.325
                          Épistémique = 0.693 − 0.325 = 0.368     ← désaccord fort
```

C'est **exactement** le signal OOD recherché : sur une image jamais vue,
chaque membre extrapole différemment → ils se contredisent → l'épistémique
monte. `unc_epistemic_combined` (gender + attrs) = **0.740 AUROC**, le
meilleur signal individuel de l'ensemble de référence.

---

## 6. L'énergie

Pour la tête gender de logits `(z₀, z₁)` :

```
E(x) = − log( e^{z₀} + e^{z₁} )        (− logsumexp, moyenné sur les membres)
```

Intuition : sur une image d'entraînement, la bonne classe pousse son logit
très haut → `logsumexp` grand → énergie très **négative**. Sur une image
OOD, les logits restent petits → l'énergie **monte**.

Exemple : logits `(8, −2)` (in-dist confiant) → `E ≈ −8.0`.
Logits `(1.2, 0.8)` (OOD, mou) → `E = −log(e^{1.2}+e^{0.8}) ≈ −1.7`.
`−1.7 > −8.0` → l'OOD score plus haut. ✓

Mesuré : 0.72–0.73 sur l'ensemble de référence, et **jusqu'à 0.80 après
entraînement CutPaste** (§12) — le pretext réorganise les features de sorte
que les logits s'effondrent davantage sur les vraies occlusions.

---

## 7. Score localisé

Deux défauts du p95 global (§4.3) sont corrigés par deux transformations,
appliquées **dans cet ordre** ([lrad/localized.py](../lrad/localized.py)) :

### 7.1 Le z-score par pixel

Sur des images de référence in-dist (val, ou un slice du train), on estime
**pour chaque position de pixel** `i` la moyenne et l'écart-type de la
carte de biais :

```
μ(i) = E_ref[ Biais(x)[i] ]        σ(i) = √Var_ref[ Biais(x)[i] ]
z(x)[i] = ( Biais(x)[i] − μ(i) ) / max(σ(i), σ_min)      σ_min = 10⁻³
```

**Pourquoi ça marche** : CelebA est aligné — le pixel (30, 40) est toujours
à peu près le même endroit du visage. Les cheveux/fond ont un μ élevé → ils
ne dominent plus ; la zone des yeux, toujours bien reconstruite (σ minuscule),
transforme le moindre écart en z énorme. On obtient l'effet « région des
yeux » **sans jamais coder la région**.

Exemple : pixel du fond : `biais = 0.9`, `μ = 0.85`, `σ = 0.3` →
`z = 0.17` (banal). Pixel de l'œil : `biais = 0.25`, `μ = 0.05`,
`σ = 0.02` → `z = 10` (très anormal). Le fond avait une erreur brute
**4× plus grande**, mais c'est l'œil qui ressort. C'est toute l'idée.

### 7.2 Le patch-max multi-échelle

Sur la carte z-scorée : average-pooling avec des fenêtres de 4, 8 et 16 px
(stride = fenêtre/2, donc chevauchantes), maximum spatial, puis maximum sur
les 3 échelles :

```
score(x) = max_taille max_position  moyenne_fenêtre( z(x) )
```

**Pourquoi** : une anomalie localisée *n'importe où* remplit exactement une
fenêtre → le max la capte, qu'elle soit sur les yeux, un chapeau ou une
main. Un pixel isolé bruité est écrasé par la moyenne de sa fenêtre. Le
test unitaire clé (`tests/test_localized.py`) plante une tache de 3.5 % des
pixels : elle est **invisible pour le p95** (< 5 %) et **détectée par le
patch-max**.

Résultat : biais 0.638 → **0.686** sur splits complets. Mieux, mais
l'espace pixel reste limité par le flou des décodeurs → §8.

---

## 8. locfre

*Localized Feature Reconstruction Error* — le meilleur signal individuel
(**0.779** sur splits complets). [lrad/feature_error.py](../lrad/feature_error.py).

### 8.1 La construction, étape par étape

1. Chaque membre reconstruit l'image au bloc **le plus profond** ; on prend
   le consensus `f̄(x)`. Point crucial : les décodeurs n'ont vu que des
   visages sans lunettes → **`f̄(x)` est un visage sans lunettes**, même si
   `x` en porte. La reconstruction « efface » l'anomalie.
2. On ré-encode `f̄(x)` dans le tronc de chaque membre → activations
   `aⱼᵐ(f̄(x))`.
3. Carte d'erreur par position spatiale `u` du bloc `j` (les vecteurs de
   canaux sont normalisés L2, notés `ĉ`) :

```
locfre_j(x)[u] = (1/M) Σₘ ‖ ĉ(aⱼᵐ(x)[u]) − ĉ(aⱼᵐ(f̄(x))[u]) ‖²
```

4. z-score par position contre des stats de référence in-dist (comme §7.1),
   puis patch-max (fenêtres 2/4/8 adaptées à la résolution du bloc).

### 8.2 Pourquoi c'est plus fort que le pixel

L'erreur **pixel** compare des couleurs : le flou du décodeur crée une
erreur de fond énorme partout, qui noie l'anomalie. L'erreur **features**
compare des *concepts* : la normalisation L2 par canal ignore le contraste
et le flou, mais un objet **sémantiquement absent** de la reconstruction
(les lunettes) change la direction du vecteur d'activations → distance
forte, exactement à la position de l'objet.

Exemple conceptuel à une position sur les yeux : `x` porte des lunettes →
le vecteur d'activations pointe vers « monture, verre, reflet ». `f̄(x)`
montre des yeux nus → le vecteur pointe vers « œil, sourcil, peau ». Ces
deux directions unitaires sont quasi orthogonales → `‖·‖² ≈ 2` (le max est
4). Sur un visage sans lunettes, les deux vecteurs coïncident → ≈ 0.

Les blocs 1 (16×16, texture fine) et 3 (4×4, sémantique) sont les plus
forts et complémentaires — d'où `blocks = (1, 3)` par défaut.

---

## 9. Fusion par rang

Les signaux (locfre 0.78, épistémique 0.74, énergie 0.72…) se trompent sur
des **images différentes**. La fusion la plus simple qui ne demande aucune
calibration : remplacer chaque score par son **rang normalisé** dans le pool
d'évaluation, puis moyenner :

```
fused(x) = (1/S) Σ_s  rang_s(x) / (N−1)
```

### Exemple complet

4 images (2 in : A, B ; 2 ood : C, D), 2 signaux :

| Image | s₁ (locfre) | rang₁ | s₂ (énergie) | rang₂ | fused |
|---|---|---|---|---|---|
| A (in) | 1.2 | 0/3 = 0.00 | −8.1 | 1/3 = 0.33 | 0.17 |
| B (in) | 3.0 | 2/3 = 0.67 | −9.0 | 0/3 = 0.00 | 0.33 |
| C (ood) | 2.1 | 1/3 = 0.33 | −5.0 | 3/3 = 1.00 | 0.67 |
| D (ood) | 4.5 | 3/3 = 1.00 | −6.2 | 2/3 = 0.67 | 0.83 |

Chaque signal seul fait une erreur (s₁ classe B au-dessus de C ; s₂ classe
A au-dessus de B, peu importe) — mais le **fused** classe parfaitement :
`{A, B} < {C, D}` → AUROC = 1.0. Les erreurs des signaux ne coïncident pas,
la moyenne des rangs les annule.

Pourquoi les rangs et pas les scores bruts ? Parce que locfre vit dans
`[0, 20]`, l'énergie dans `[−12, −1]` : une moyenne brute serait dominée
par l'échelle, pas par l'information. Le rang est sans échelle (§2.4).

Recette validée : `locfre_b1 + locfre_b3 + épistémique + énergie`
(+ `cutpaste_prob` si les modèles ont la tête §11) → **0.803** splits
complets, sans aucune donnée OOD ni étiquette.

---

## 10. Fusion supervisée

La fusion par rang pèse tout également. Une **régression logistique**
apprend les poids sur des exemples étiquetés :

```
fused_sup(x) = Σ_s w_s · (score_s(x) − μ_s)/σ_s + b
```

les `w_s` maximisant la séparation in/ood sur un jeu de **calibration**.

### 10.1 Le protocole anti-fuite (crucial)

```
pool OOD (13 193)  ──split seed 42──►  moitié A (6 596)  → CALIBRATION (label 1)
                                       moitié B (6 597)  → ÉVALUATION seulement
train in-dist      ──slice────────►    6 596 négatifs    → CALIBRATION (label 0)
test_in (18 941)   ────────────────►   jamais touché     → ÉVALUATION seulement
```

L'AUROC finale est mesurée sur `test_in` vs **moitié B** — des images que
la régression n'a jamais vues. Le split est fait par
`split_loader` ([lrad/dataset.py](../lrad/dataset.py)) : permutation seedée,
déterministe, disjointe.

### 10.2 Les poids appris (run 6774881) et leur lecture

```
unc_epistemic_combined  +6.26   ← le signal porteur
ens_msp_gender          +2.40
locfre_b3               +0.95
locfre_b1               +0.37
ens_energy_gender       +0.48
unc_epistemic_gender    −4.44   ← corrections de redondance
unc_total_combined      −2.46
```

Les poids **négatifs** sont l'avantage clé sur la fusion par rang :
épistémique *combined* et épistémique *gender* sont très corrélés ; la
régression garde l'un à fond et **soustrait** l'autre pour ne conserver que
l'information non redondante — ce qu'une moyenne ne peut pas faire.
Résultat : **0.810** vs 0.803 (rang). Le prix : il faut ~quelques milliers
d'exemples OOD étiquetés, et les poids sont spécialisés « lunettes ».

---

## 11. CutPaste

Tout ce qui précède est *post-hoc* (les modèles n'ont jamais rien appris
sur les occlusions). CutPaste ([lrad/cutpaste.py](../lrad/cutpaste.py))
attaque l'entraînement lui-même, **sans aucune donnée OOD réelle**.

### 11.1 L'augmentation

Pendant l'entraînement, chaque image du batch est altérée avec probabilité
`prob` : un rectangle est découpé dans une image **donneuse** (l'image
suivante du batch — donc de la vraie texture de visage) et collé à une
position aléatoire. Deux formes :

- **patch** : aire `∈ area_range` × l'image (ex. 5–15 %), ratio d'aspect
  `∈ aspect_range` ;
- **scar** : sliver fin (2–8 px × 10–45 % de l'image), horizontal ou
  vertical — mélangés par `scar_prob`.

Exemple avec `area_range = [0.05, 0.15]` sur du 64×64 : aire tirée 10 % →
410 px² ; aspect tiré 2.0 → `h = √(410×2) ≈ 29`, `w = √(410/2) ≈ 14` →
un rectangle 29×14 collé quelque part. C'est, en gros, la taille d'une
paire de lunettes.

### 11.2 La tête pretext et la perte

Le modèle reçoit une 3ᵉ tête binaire (`model.cutpaste_head: true`) qui
prédit *intact (0) vs altéré (1)*. La perte totale d'un batch :

```
L = CE_gender(intacts) + 2·BCE_attrs(intacts) + w_cp · CE_cutpaste(tout le batch)
```

Les pertes supervisées ne portent que sur les images **intactes** (un patch
peut occulter les sourcils dont dépendent les labels) ; la perte pretext
porte sur tout. `w_cp` = `loss_weight` (le grid search a montré que 0.5
vaut mieux que 2 — trop de pretext dégrade le reste, §12).

### 11.3 Le signal à l'évaluation

```
cutpaste_prob(x) = (1/M) Σₘ P_m(altéré | x)
```

Le modèle n'a vu que des patchs synthétiques, mais « quelque chose recouvre
le visage » **généralise** : de vraies lunettes font monter P(altéré).
Mesuré au grid search : jusqu'à **0.746** AUROC pour un seul petit modèle.
Le signal s'ajoute automatiquement aux deux fusions quand les membres
portent la tête.

---

## 12. Le grid search multi-métriques

[scripts/run_gridsearch.py](../scripts/run_gridsearch.py) — comment choisir
les hyperparamètres CutPaste **sans** entraîner 30 ensembles complets.

### 12.1 La grille et son élagage

```
scar_prob   ∈ {0, 0.5, 1.0}     forme des patchs
area_range  ∈ {(2–8 %), (5–15 %)}   taille des boîtes
prob        ∈ {0.3, 0.5}        fraction du batch altérée
loss_weight ∈ {0.5, 1.0, 2.0}   poids de la perte pretext
```

Produit brut : 36 configs. **Élagage** : quand `scar_prob = 1.0` (scars
uniquement), `area_range` n'est jamais lu → les configs qui ne diffèrent
que par lui sont identiques ; on n'en garde qu'une → **30 configs**.

### 12.2 Ce que fait chaque config (≈ 18 min sur 2080 Ti)

1. **Même seed pour toutes** — elles ne diffèrent que par les boutons
   CutPaste, pas par l'init ni l'ordre des batchs (comparaison propre).
2. Entraîne le classifieur **6 epochs** (vs 20 en vrai) avec la tête
   pretext, un checkpoint par epoch.
3. Recharge chaque checkpoint et mesure l'AUROC cutpaste par epoch → la
   courbe `gridsearch_epochs.png` (répond à « combien d'epochs ? » sans en
   faire une dimension de grille — gratuit).
4. Entraîne les décodeurs **8 epochs** (vs 25) — nécessaires pour mesurer
   les signaux de reconstruction.
5. Évalue **5 AUROC** sur des splits de test plafonnés (~5 000 images/côté) :
   `cutpaste`, `bias_p95` (§4), `locfre_b3` (§8), `energy` (§6), et
   **`fused`** = fusion par rang des 4 (l'épistémique est exclue : avec un
   seul modèle elle vaut identiquement 0).
6. **La sélection se fait sur `fused`** : c'est le score final du projet,
   donc on choisit les hyperparamètres qui optimisent *l'ensemble du
   stack*, pas une métrique isolée — c'était la demande.

### 12.3 Pourquoi des schedules courts suffisent

On ne cherche pas la performance absolue, on cherche un **classement** de
configs. L'hypothèse (standard) : si la config X bat la config Y à 6
epochs, elle la bat aussi à 20. Le risque de mauvais classement existe mais
il est faible entre configs proches, et le coût est divisé par ~5.

### 12.4 Les résultats (job 6781780) et leur lecture

| Rang | Config | fused | cutpaste | bias | locfre | energy |
|---|---|---|---|---|---|---|
| 1 | box, 5–15 %, p=.5, w=.5 | **0.827** | 0.746 | 0.634 | 0.760 | 0.803 |
| 2 | box, 5–15 %, p=.5, w=1 | 0.822 | 0.676 | 0.638 | 0.752 | 0.806 |
| … | | | | | | |
| 30 | scar-only, p=.5, w=.5 | 0.659 | 0.592 | 0.639 | 0.758 | 0.406 |

Lectures :

- **Gros patchs sans scar gagnent** (rangs 1-2-3-5) : un gros rectangle de
  texture ressemble plus à une vraie occlusion qu'un trait fin.
- **`bias_p95` est plat (~0.64 partout)** : les décodeurs s'entraînent sur
  des reconstructions propres quelle que soit l'augmentation — c'est le
  128 px qui doit l'aider, pas CutPaste. Le grid le **prouve** au lieu de
  le supposer.
- **L'énergie varie de 0.41 à 0.80 selon la config** : la découverte
  majeure. Le pretext restructure le tronc au point de transformer un
  signal existant. C'est pour ça qu'il fallait une sélection
  multi-métriques : une config choisie sur `cutpaste` seul aurait pu tuer
  l'énergie sans qu'on le voie.
- 0.827 avec **un seul** modèle court à 64 px > 0.810 de l'ensemble complet
  de 10 → l'ensemble 128 px final part de très haut.

---

## 13. Historique des résultats

Splits complets sauf mention. La progression, mesure par mesure :

| Étape | Score | AUROC |
|---|---|---|
| Baseline historique | biais p95 global (§4) | 0.638 |
| + incertitude | épistémique combined (§5) | 0.740 |
| + localisation | z-score pixel + patch-max (§7) | 0.686 |
| + espace features | locfre_b3 (§8) | 0.779 |
| + fusion | rang à 4 signaux (§9) | **0.803** |
| + supervision | logistique calibrée (§10) | **0.810** |
| + CutPaste (grid, 1 modèle court, sous-échantillon) | fused (§12) | **0.827** |
| ensemble 8×128 px + CutPaste (job à lancer) | fused | objectif 0.85–0.90 |

---

## 14. Où vit chaque mesure dans le code

| Mesure | Module | Script d'évaluation |
|---|---|---|
| AUROC / ROC | `lrad/evaluate.py` (`_auroc_entry`, `ood_auroc`) | tous |
| MSP, entropies | `lrad/evaluate.py` (`collect_predictions`) | `run_celeba.py` |
| Risque/Biais/Variance | `lrad/ensemble.py` (`decomposition_maps`) | `run_ensemble.py` |
| Incertitudes | `lrad/ensemble.py` (`_uncertainty_scores`) | `run_ensemble.py` |
| z-score + patch-max | `lrad/localized.py` | `run_localized.py` |
| locfre | `lrad/feature_error.py` | `run_fused.py` |
| énergie, cutpaste_prob | `lrad/fusion.py` (`collect_fusion_signals`) | `run_fused.py` |
| fusion rang / supervisée | `lrad/fusion.py` | `run_fused.py --supervised` |
| CutPaste | `lrad/cutpaste.py` + `lrad/train.py` | (entraînement) |
| grid search | — | `run_gridsearch.py` |

Jobs OAR : `oar_run_gridsearch.sh` (24 h), `oar_run_cutpaste128.sh` (48 h,
gruss), `oar_run_fused.sh` (4 h, éval seule sur checkpoints existants).
