# Run `cp128_20260719_060210_6782570` — rapport complet

*Ensemble profond de 8 membres, CelebA 128 px, CutPaste activé, OOD = attribut `Eyeglasses`.
Job OAR 6782570, GPU NVIDIA A40, torch 2.5.1+cu121, du 19 juillet 2026 06:03 au 16:37 (≈ 10 h 34 min).

Ce document explique **ce qui a été calculé**, **avec quelles formules**, **comment lire chaque
figure produite**, et **ce que les chiffres veulent dire**. Les mathématiques dures sont
accompagnées d'un petit exemple entièrement chiffré.

Document frère : [`MESURES.md`](MESURES.md) (formulation générale des mesures, tous runs confondus).
Ce fichier-ci est spécifique à ce run.

## Sommaire

1. [Résumé en une page](#1-résumé-en-une-page)
2. [Données et protocole de découpe](#2-données-et-protocole-de-découpe)
3. [Architectures](#3-architectures)
4. [Entraînement](#4-entraînement)
5. [Le socle : Risque = Biais + Variance](#5-le-socle--risque--biais--variance)
6. [Les scores OOD, un par un](#6-les-scores-ood-un-par-un)
7. [Fusion](#7-fusion)
8. [AUROC — la métrique](#8-auroc--la-métrique)
9. [Interprétation de chaque figure](#9-interprétation-de-chaque-figure)
10. [Tableaux de résultats](#10-tableaux-de-résultats)
11. [Lecture critique et anomalies](#11-lecture-critique-et-anomalies)
12. [Reproduire ce run](#12-reproduire-ce-run)

---

## 1. Résumé en une page

**La question.** Un classifieur de visages est entraîné uniquement sur des visages **sans
lunettes**. Au test, on lui présente des visages **avec lunettes**. Peut-on détecter
automatiquement que ces images sont hors-distribution (OOD), sans jamais avoir montré de lunettes
au modèle pendant l'entraînement ?

**La méthode.** Huit modèles indépendants (architectures différentes, graines différentes) sont
entraînés. Chacun porte :

- un tronc convolutif à 5 blocs, avec trois têtes : genre, 6 attributs faciaux, et une tête
  **CutPaste** auto-supervisée (« ce visage a-t-il été altéré ? ») ;
- cinq **décodeurs**, un par bloc, qui reconstruisent l'image d'entrée depuis les activations de ce
  bloc.

On en tire une famille de signaux OOD, puis on les fusionne.

**Le résultat.**

| Score | AUROC | Commentaire |
|---|---|---|
| Biais pixel p95 (ligne de base historique) | **0.6237** | plafonne — les décodeurs sont flous |
| z-score + patch-max sur le risque | **0.7016** | la normalisation par pixel aide beaucoup |
| `locfre_b3` (erreur de features, bloc 4) | **0.7898** | meilleur signal isolé |
| Énergie de la tête genre | **0.7387** | complémentaire, gratuit |
| Fusion par rang, 6 signaux, sans étiquettes | **0.8120** | ce qu'un détecteur déployable atteindrait |
| Fusion supervisée (régression logistique) | **0.8838** | borne haute, voit des étiquettes OOD à la calibration |

**Le message.** Le gain ne vient pas d'un score magique mais de trois idées empilées :
(1) quitter l'espace des pixels pour l'espace des **features**, (2) **normaliser par position**
avant de réduire, (3) **fusionner** des signaux qui se trompent sur des images différentes.

---

## 2. Données et protocole de découpe

### 2.1 Définition de l'OOD

Source : `lrad/dataset.py`. CelebA « all » = 202 599 visages, chacun décrit par 40 attributs
binaires. Une image est déclarée OOD si l'attribut `Eyeglasses` est activé :

$$
\text{OOD}(x) = \mathbb{1}\big[\,\texttt{Eyeglasses}(x) = 1\,\big]
$$

Ce qui donne, dans le log du run :

```
in-dist = 189 406 images     ood = 13 193 images
```

### 2.2 Découpe

Le pool in-distribution est découpé avec `train_ratio = 0.9`, `val_ratio = 0.0`, seed 42 :

| Split | Taille | Rôle |
|---|---|---|
| `train` | 170 465 | entraînement classifieur + décodeurs |
| `val` | **0** | *aucun* — donc pas d'early stopping |
| `test_in` | 18 941 | négatifs de l'évaluation (label 0) |
| `test_ood` | 13 193 | positifs de l'évaluation (label 1) |

`val_ratio = 0` a deux conséquences directes, visibles dans le code (`lrad/train.py`) :

- **pas d'early stopping** : les 20 époques sont exécutées et les poids de la dernière époque sont
  conservés tels quels (`best_state` reste `None`) ;
- les **statistiques de référence** de tous les scores normalisés doivent venir de `train`, jamais
  de `test_in` — sinon on fuiterait le split d'évaluation. `run_fused.py` respecte cela
  explicitement.

### 2.3 Étiquettes prédites

Genre (`Male`) plus six attributs :

```
Arched_Eyebrows, Bushy_Eyebrows, Narrow_Eyes, Bags_Under_Eyes, Smiling, Young
```

**Quatre sur six sont péri-oculaires**, et c'est délibéré. Le classifieur, entraîné exclusivement
sur des visages sans lunettes, doit donc apprendre à regarder la région des yeux et des sourcils.
Les lunettes occultent précisément l'évidence dont ces têtes dépendent — c'est ce qui maximise
l'écart d'activation entre ID et OOD dans cette zone.

Taux de positifs sur `train` (log du run) :

```
Male=39.0%  Arched_Eyebrows=28.4%  Bushy_Eyebrows=14.9%
Narrow_Eyes=11.9%  Bags_Under_Eyes=20.9%  Smiling=48.7%  Young=79.9%
```

### 2.4 Découpe supplémentaire pour la fusion supervisée

`scripts/run_fused.py --supervised` refait une découpe déterministe (seed 42, `ood_cal_frac = 0.5`),
via `lrad.dataset.split_loader` :

| Rôle | Provenance | Taille |
|---|---|---|
| Négatifs de calibration | tranche de `train` | 6 596 |
| Positifs de calibration | moitié du pool OOD | 6 596 |
| Statistiques de référence `locfre` | reste de `train` (plafonné à 40 batches) | 5 120 |
| Évaluation, négatifs | `test_in` entier | 18 941 |
| Évaluation, positifs | **autre** moitié du pool OOD | 6 597 |

> ⚠️ **Conséquence à retenir.** Les AUROC du fichier `localized_auroc.json` sont mesurées sur
> **13 193** images OOD ; celles de `fused_auroc.json` sur **6 597**. Les deux ensembles de
> chiffres ne sont pas strictement comparables entre eux. Comparez toujours à l'intérieur d'un
> même fichier.

---

## 3. Architectures

Trois schémas SVG, exacts au code, accompagnent ce rapport :

| Fichier | Contenu |
|---|---|
| [`diagrams/pipeline_complet.svg`](diagrams/pipeline_complet.svg) | Le pipeline de bout en bout : données → 8 membres → 5 familles de signaux → fusion → AUROC |
| [`diagrams/pipeline_classifier.svg`](diagrams/pipeline_classifier.svg) | Le tronc `FacialCNN` : 5 blocs, formes exactes, 3 têtes, CutPaste, pertes |
| [`diagrams/pipeline_decoder.svg`](diagrams/pipeline_decoder.svg) | Les 5 `BlockDecoder` : progression canaux × résolution, étage par étage |

### 3.1 Le classifieur

Source : `lrad/model.py`. Chaque bloc est
`Conv(k×k, padding = k//2) → BatchNorm → ReLU → MaxPool(2×2)`, le pooling étant **omis sur le
dernier bloc** (`pool = i < len(channels) - 1`). Le padding conserve H et W dans la convolution,
donc seul le MaxPool divise la résolution.

Pour le membre 0 (`channels = [32, 64, 128, 256, 256]`, `kernel_size = 3`), entrée 128×128 :

> **Convention de numérotation.** Ce rapport numérote les blocs **1 à 5**. Le code les indexe
> **0 à 4** et les figures les étiquettent `L0…L4`. « Bloc 4 » ici = `L3` dans les figures.

| Bloc | Sortie | Rôle intuitif |
|---|---|---|
| 1 | 32 × 64 × 64 | bords, texture fine |
| 2 | 64 × 32 × 32 | motifs locaux |
| 3 | 128 × 16 × 16 | parties du visage |
| 4 | 256 × 8 × 8 | sémantique |
| 5 | 256 × 8 × 8 | sémantique, sans perte de résolution |

Puis `AdaptiveAvgPool2d(1)` → $h \in \mathbb{R}^{256}$ → trois têtes linéaires.
**981 802 paramètres** entraînables (confirmé dans le log).

Une propriété importante du design : `kernel_size` ne change que le champ réceptif, jamais la
géométrie spatiale. Les 8 membres ont donc **exactement les mêmes 5 résolutions de blocs** malgré
des largeurs et des noyaux différents — c'est ce qui permet d'aligner la décomposition par bloc
entre membres.

### 3.2 Les décodeurs

Source : `lrad/decoder.py`. Un `BlockDecoder` par bloc, empilant
$n_{\text{up}} = \log_2(128 / \text{taille du bloc})$ étages
`ConvTranspose2d(4×4, stride 2, padding 1) → BN → ReLU`, canaux divisés par 2 à chaque étage avec
un plancher de 16, puis `Conv1×1 → 3` et une **sigmoïde** (la sortie vit donc dans $[0,1]$, comme
l'image).

Le noyau 4×4 / stride 2 / padding 1 double H et W exactement, sans ambiguïté de taille de sortie :
c'est la raison de ce choix plutôt qu'un 3×3.

| Décodeur | Entrée | Étages | Paramètres |
|---|---|---|---|
| $d_0$ | 32 × 64 × 64 | 1 | 8 275 |
| $d_1$ | 64 × 32 × 32 | 2 | 41 107 |
| $d_2$ | 128 × 16 × 16 | 3 | 172 307 |
| $d_3$ | 256 × 8 × 8 | 4 | 696 851 |
| $d_4$ | 256 × 8 × 8 | 4 | 696 851 |
| | | **Total** | **1 615 391** |

### 3.3 Les 8 membres

`ensemble.member_variants` donne à chaque membre sa propre architecture. La diversité de
l'ensemble vient donc de **trois** sources cumulées : graine d'initialisation, ordre de mélange
SGD, et architecture.

| # | Graine | `channels` | `k` |
|---|---|---|---|
| 0 | 42 | 32, 64, 128, 256, 256 | 3 |
| 1 | 43 | 24, 48, 96, 192, 384 | 3 |
| 2 | 44 | 48, 96, 192, 256, 256 | 3 |
| 3 | 45 | 16, 48, 96, 192, 320 | 5 |
| 4 | 46 | 40, 80, 160, 320, 320 | 3 |
| 5 | 47 | 32, 64, 128, 256, 256 | 5 |
| 6 | 48 | 64, 96, 160, 224, 288 | 3 |
| 7 | 49 | 24, 64, 128, 192, 256 | 5 |

Ce n'est **pas** du MC-Dropout : il n'y a aucune couche Dropout dans le modèle. C'est un *deep
ensemble* au sens strict.

---

## 4. Entraînement

### 4.1 Étape A — le classifieur (20 époques, Adam 3e-4)

$$
\mathcal{L} \;=\; \underbrace{\mathrm{CE}(z_g,\, y_g)}_{\text{genre}}
\;+\; 2.0 \cdot \underbrace{\mathrm{BCE}(z_a,\, y_a)}_{\text{6 attributs}}
\;+\; 0.5 \cdot \underbrace{\mathrm{CE}(z_{cp},\, y_{cp})}_{\text{CutPaste}}
$$

**CutPaste** (`lrad/cutpaste.py`, d'après Li et al., CVPR 2021) : à chaque batch, chaque image est
altérée avec probabilité 0.5 en y collant un rectangle découpé dans une **autre image du batch**
(le donneur est $x_{i+1 \bmod B}$, donc jamais soi-même). Configuration de ce run :

```yaml
prob: 0.5           # une image sur deux est altérée
area_range: [0.05, 0.15]   # le patch couvre 5 à 15 % de l'image
aspect_range: [0.3, 3.3]
scar_prob: 0.0      # que des rectangles, pas de « balafres » fines
loss_weight: 0.5
```

Le patch fait donc entre $0.05 \times 128^2 = 819$ et $0.15 \times 128^2 = 2458$ pixels.

Deux détails du code qui comptent :

1. **Les pertes genre/attributs ne sont calculées que sur les images intactes** du batch
   (`intact = cp_labels == 0`). Un patch collé peut recouvrir exactement l'évidence que
   l'étiquette décrit — l'entraîner dessus injecterait du bruit d'étiquette.
2. `scar_prob = 0` : ce run n'utilise que des rectangles. Les « balafres » (fines bandes, plus
   proches de la forme d'une monture de lunettes) sont désactivées. C'est un axe d'amélioration
   évident pour un run suivant.

**L'idée derrière CutPaste.** On ne peut pas apprendre « lunettes » sans jamais en voir. Mais on
peut apprendre « quelque chose recouvre une partie de ce visage », en le fabriquant. Au test,
$P(\text{altéré} \mid x)$ se déclenche sur des occlusions réelles jamais vues — les lunettes
incluses. Le run confirme : ce signal seul atteint **AUROC 0.6918**.

### 4.2 Étape B — les décodeurs (25 époques, Adam 1e-3, classifieur gelé)

$$
\mathcal{L}_{\text{dec}} \;=\; \sum_{k=0}^{4} \mathrm{MSE}\big(d_k(a_k(x)),\; x\big)
$$

Le classifieur est mis en `eval()` et ses paramètres passent en `requires_grad = False`. Les cinq
décodeurs sont optimisés ensemble par un unique Adam.

MSE par bloc du membre 0 (`decoders_history.json`) :

| | bloc 1 | bloc 2 | bloc 3 | bloc 4 | bloc 5 |
|---|---|---|---|---|---|
| époque 1 | 0.00340 | 0.00539 | 0.00769 | 0.01174 | 0.02017 |
| époque 25 | 0.00043 | 0.00150 | 0.00327 | 0.00532 | 0.01124 |

**Lecture.** L'erreur croît monotonement avec la profondeur, d'un facteur ≈ 26 entre le bloc 1 et
le bloc 5. C'est mécanique : les MaxPool successifs ont détruit l'information spatiale, et un
décodeur partant d'une carte 8×8 ne peut plus produire qu'un visage flou. **Ce plancher d'erreur
est la raison pour laquelle tous les scores en espace pixel plafonnent autour de 0.62–0.70** :
sur un visage propre, les cheveux et le fond fournissent déjà des milliers de pixels à forte
erreur, tandis qu'une paire de lunettes n'en touche que quelques centaines.

### 4.3 Résultats par membre

| # | Graine | Précision genre (`test_in`) | AUROC MSP | AUROC entropie genre |
|---|---|---|---|---|
| 0 | 42 | **0.6303** ⚠️ | **0.2602** ⚠️ | **0.2958** ⚠️ |
| 1 | 43 | 0.9822 | 0.7609 | 0.7632 |
| 2 | 44 | 0.9846 | 0.7710 | 0.7777 |
| 3 | 45 | 0.9786 | 0.7990 | 0.8111 |
| 4 | 46 | 0.9760 | 0.7850 | 0.7919 |
| 5 | 47 | 0.9825 | 0.7074 | 0.7166 |
| 6 | 48 | 0.9780 | 0.7005 | 0.7062 |
| 7 | 49 | 0.9752 | 0.6368 | 0.6491 |

Le membre 0 est aberrant — voir [§11](#11-lecture-critique-et-anomalies).

---

## 5. Le socle : Risque = Biais + Variance

C'est l'identité centrale du projet. Source : `lrad/ensemble.py:decomposition_maps`.

### 5.1 Notation

- $M = 8$ membres, indexés par $m$.
- $x \in [0,1]^{3 \times 128 \times 128}$ l'image d'entrée.
- $\hat f_k^m(x)$ la reconstruction du membre $m$ depuis le bloc $k$.
- $p$ un pixel (position spatiale), $c \in \{R, G, B\}$ un canal.

L'erreur par pixel est la **somme** sur les canaux RVB des carrés — pas la moyenne, pas de racine :

$$
e^m_k(x)[p] \;=\; \sum_{c \in \{R,G,B\}} \big( x_c[p] - \hat f^m_{k,c}(x)[p] \big)^2
\;\in\; [0, 3]
$$

Reconstruction consensus (la « moyenne des modèles ») :

$$
\bar f_k(x) \;=\; \frac{1}{M} \sum_{m=1}^{M} \hat f^m_k(x)
$$

### 5.2 Les trois termes

$$
\begin{aligned}
\textbf{Risque}_k(x)[p]   &= \frac{1}{M} \sum_{m} \big( x[p] - \hat f^m_k(x)[p] \big)^2
&&\text{la moyenne des erreurs} \\[4pt]
\textbf{Biais}_k(x)[p]    &= \big( x[p] - \bar f_k(x)[p] \big)^2
&&\text{l'erreur de la moyenne} \\[4pt]
\textbf{Variance}_k(x)[p] &= \frac{1}{M} \sum_{m} \big( \hat f^m_k(x)[p] - \bar f_k(x)[p] \big)^2
&&\text{le désaccord entre modèles}
\end{aligned}
$$

### 5.3 L'identité et sa preuve

$$
\boxed{\;\text{Risque} \;=\; \text{Biais} \;+\; \text{Variance}\;}
$$

*Preuve.* Fixons un pixel et un canal, notons $a = x[p]$, $f_m = \hat f^m[p]$,
$\bar f = \frac{1}{M}\sum_m f_m$. On écrit $a - f_m = (a - \bar f) + (\bar f - f_m)$ et on développe :

$$
\frac{1}{M}\sum_m (a - f_m)^2
= \underbrace{(a-\bar f)^2}_{\text{constant en } m}
+ \frac{2(a-\bar f)}{M}\underbrace{\sum_m (\bar f - f_m)}_{=\;0}
+ \frac{1}{M}\sum_m (\bar f - f_m)^2
$$

Le terme croisé s'annule **exactement**, par définition de la moyenne. Il reste
Biais + Variance. L'identité est vraie canal par canal, donc elle survit à la somme sur RVB. ∎

Le run vérifie numériquement cette identité (`identity_residual`) :

```
Risk = Bias + Variance — max abs residual sur 25 échantillons : 2.12e-07
```

soit le niveau de bruit du float32. Si ce nombre dépassait $10^{-5}$, il y aurait un bug.

### 5.4 Exemple chiffré

$M = 2$, un pixel, un canal. $x = 0.5$, $\hat f^1 = 0.3$, $\hat f^2 = 0.9$.

- $\bar f = (0.3 + 0.9)/2 = 0.6$
- Risque $= \frac{(0.5-0.3)^2 + (0.5-0.9)^2}{2} = \frac{0.04 + 0.16}{2} = 0.10$
- Biais $= (0.5 - 0.6)^2 = 0.01$
- Variance $= \frac{(0.3-0.6)^2 + (0.9-0.6)^2}{2} = \frac{0.09+0.09}{2} = 0.09$
- Vérification : $0.01 + 0.09 = 0.10$ ✓

### 5.5 Ce que chaque terme signifie

- **Biais** = l'erreur qui **survit à l'ensemble**. Même en moyennant 8 modèles, on n'arrive pas à
  reconstruire ces pixels. C'est le candidat naturel pour « anomalie » : sur des lunettes, aucun
  décodeur entraîné sur des visages propres ne sait dessiner une monture.
- **Variance** = le **désaccord épistémique** en espace pixel. Hors distribution, les modèles
  extrapolent différemment, donc en principe la variance monte.
- **Risque** = ce que coûte un modèle moyen pris isolément.

**Résultat observé, et il est instructif** : sur cette tâche, la variance (0.6431) fait légèrement
**mieux** que le biais (0.6237), et le risque se situe entre les deux (0.6310). Autrement dit,
l'hypothèse de départ du projet — « le biais est le bon score d'anomalie » — n'est pas confirmée
ici. L'explication est dans le commentaire du code (`lrad/ensemble.py`, §uncertainty) : pour une
tâche d'**occlusion**, tous les membres échouent *au même endroit*, donc l'erreur est grande mais
**corrélée** → biais élevé, variance faible. Le désaccord qui croît réellement hors distribution
vit dans les **prédictions**, pas dans les pixels.

---

## 6. Les scores OOD, un par un

Convention commune à tous les scores : **grand = plus OOD**.

### 6.1 Réduction pixel → scalaire (p95)

Source : `lrad/anomaly_score.py:_reduce_over_pixels`.

Une carte $(H, W)$ doit devenir un nombre. Trois options, `p95` retenue ici :

$$
s(A) = \operatorname{quantile}_{0.95}\big(\operatorname{vec}(A)\big)
$$

- `mean` : sensible à toute l'image → une anomalie locale se dilue.
- `max` : sensible au pixel unique le plus surprenant → très bruité.
- `p95` : compromis robuste. Ignore les pixels chauds isolés, mais se déclenche dès que ~5 % des
  pixels bougent.

Puis moyenne uniforme sur les 5 blocs :
$S(x) = \frac{1}{5}\sum_{k=0}^{4} s\big(A_k(x)\big)$.

**Exemple.** Sur 20 valeurs de pixels triées, le p95 est la valeur au rang $0.95 \times 19 = 18.05$,
donc entre le 19ᵉ et le 20ᵉ élément — en pratique, la 2ᵉ plus grande. Sur 128×128 = 16 384 pixels,
le p95 est le 819ᵉ plus grand. Une paire de lunettes touche de l'ordre de 500 à 1 500 pixels : elle
est donc *juste à la limite* d'influencer le p95. C'est précisément pourquoi ce score plafonne.

### 6.2 Score localisé : z-score par pixel + patch-max

Source : `lrad/localized.py`. Deux corrections composées, sans jamais coder en dur où se trouve
l'anomalie.

**(1) z-score par pixel.** Les visages CelebA sont **alignés** : la position d'un pixel a un sens
sémantique stable. On ajuste $\mu[p]$ et $\sigma[p]$ sur des images in-distribution de référence
(5 120 images de `train` ici), puis :

$$
z_k(x)[p] \;=\; \frac{A_k(x)[p] - \mu_k[p]}{\max\big(\sigma_k[p],\, \varepsilon\big)},
\qquad \varepsilon = 10^{-3}
$$

Effet : les zones **toujours** mal reconstruites (cheveux, fond → grand $\mu$) sont annulées, et
les zones normalement faciles (le visage aligné → petit $\sigma$) voient toute déviation amplifiée.

Le plancher $\varepsilon$ est nécessaire : un pixel dont l'écart-type de référence s'effondre vers 0
(fond saturé) transformerait sinon du bruit float en z-score non borné.

*Calcul en streaming.* Les statistiques sont accumulées en float64 par somme et somme des carrés,
donc la mémoire reste $O(H \cdot W)$ par bloc, jamais $O(N \cdot H \cdot W)$ :

$$
\mu = \frac{\sum_i A_i}{N}, \qquad
\sigma^2 = \max\left(\frac{\sum_i A_i^2}{N} - \mu^2,\; 0\right)
$$

**(2) patch-max multi-échelle.**

$$
s(z) \;=\; \max_{w \in \{4, 8, 16\}}\;\;
\max_{\text{positions}}\;\; \operatorname{AvgPool}_{w,\, \text{stride}\, w/2}(z)
$$

Le stride $w/2$ crée un recouvrement, donc une anomalie à cheval sur une frontière de fenêtre est
quand même vue en entier. Le moyennage à l'intérieur de la fenêtre écrase le bruit d'un pixel
isolé, tandis que le max spatial permet de déclencher **n'importe où** dans l'image.

**Exemple chiffré.** Un pixel dont la carte de biais vaut 0.05, avec $\mu[p] = 0.03$ et
$\sigma[p] = 0.01$ :

$$
z = \frac{0.05 - 0.03}{0.01} = 2.0
$$

Ce même pixel dans les cheveux, où $\mu[p] = 0.20$ et $\sigma[p] = 0.08$, donnerait
$z = (0.05 - 0.20)/0.08 = -1.875$ : négatif, donc ignoré. Le z-score fait exactement le travail
attendu.

Supposons maintenant une fenêtre 4×4 dont les 16 z-scores valent tous 2.0 sauf un à 8.0 :
moyenne $= (15 \times 2 + 8)/16 = 2.375$. Un pixel chaud isolé sur fond de z=0 donnerait
$8/16 = 0.5$. La fenêtre privilégie bien les régions **cohérentes**, pas les pics isolés.

**Résultats du run** (`localized_auroc.json`, 18 941 ID vs 13 193 OOD) :

| Terme | z + patch-max, agrégé | Ligne de base p95 | Gain |
|---|---|---|---|
| Risque | **0.7016** | 0.6310 | +0.071 |
| Variance | **0.6936** | 0.6431 | +0.051 |
| Biais | **0.6802** | 0.6237 | +0.056 |

La normalisation par position vaut, à elle seule, entre 5 et 7 points d'AUROC.

### 6.3 `locfre` — erreur de features localisée

Source : `lrad/feature_error.py`. **C'est le meilleur signal isolé du run.**

**L'idée.** L'espace pixel est condamné par le flou des décodeurs. L'espace des **features**
contourne le problème : on ne compare plus des images, on compare des *descriptions*.

**Le calcul, en trois temps.**

1. Reconstruire l'entrée au bloc le plus profond et moyenner sur l'ensemble :
   $\bar f_4 = \frac{1}{M}\sum_m d^m_4(a^m_4(x))$.
   Cette reconstruction consensus est un « visage propre » : les lunettes en sont **absentes**,
   parce qu'aucun décodeur entraîné sur des visages sans lunettes ne sait en dessiner.

2. **Ré-encoder** $\bar f_4$ avec le tronc de chaque membre, et comparer position par position les
   activations **normalisées par canal** :

$$
\hat a_j = \frac{a_j}{\lVert a_j \rVert_2} \quad \text{(norme sur l'axe des canaux)}
$$

$$
E_j(x)[u,v] \;=\; \frac{1}{M}\sum_{m=1}^{M}
\big\lVert\, \hat a^m_j(x)[u,v] \;-\; \hat a^m_j(\bar f_4)[u,v] \,\big\rVert_2^2
$$

3. z-score contre des statistiques de référence par position (mêmes formules qu'en §6.2), puis
   patch-max avec des fenêtres adaptées à la résolution de la carte de features
   (`{2, 4, 8}` filtrées à $\leq \min(h,w)$).

**Pourquoi normaliser les activations ?** Parce que la distance entre deux **vecteurs unitaires**
ne mesure que l'**angle**, jamais l'amplitude :

$$
\lVert u - v \rVert^2 = \lVert u \rVert^2 + \lVert v \rVert^2 - 2\,u^\top v
= 2 - 2\cos\theta \;\in\; [0, 4]
$$

Le signal mesure donc « le contenu sémantique a-t-il changé de nature », et non « l'activation
est-elle plus forte ». C'est robuste aux différences de contraste et d'échelle entre membres.

**Exemple chiffré.** Deux vecteurs de features unitaires séparés de 60° :
$\lVert u - v \rVert^2 = 2 - 2\cos 60° = 2 - 1 = 1$.
Séparés de 90° : $2 - 0 = 2$. Diamétralement opposés : $2 + 2 = 4$ (le maximum).

**Résultats** (`fused_auroc.json`, blocs 1, 2, 3 — soit les blocs 2, 3 et 4 du tronc) :

| Signal | Résolution de la carte | AUROC | moyenne ID | moyenne OOD |
|---|---|---|---|---|
| `locfre_b1` | 32 × 32 | 0.7838 | 3.023 | 4.016 |
| `locfre_b2` | 16 × 16 | 0.7735 | 2.387 | 3.511 |
| `locfre_b3` | 8 × 8 | **0.7898** | 1.570 | 2.592 |

Le bloc 3 (sémantique, 8×8) gagne de peu, mais les trois sont proches et **complémentaires** —
c'est pour cela que la fusion les prend tous les trois.

### 6.4 Incertitude prédictive : Total = Aléatoire + Épistémique

Source : `lrad/ensemble.py:_uncertainty_scores`. Ici on quitte les pixels pour les **prédictions**.

Pour un ensemble profond, la décomposition standard de l'incertitude prédictive est :

$$
\begin{aligned}
\text{Total} &= H\!\left(\frac{1}{M}\sum_m p_m\right) && \text{entropie de la prédiction moyenne} \\
\text{Aléatoire} &= \frac{1}{M}\sum_m H(p_m) && \text{entropie moyenne des membres} \\
\text{Épistémique} &= \text{Total} - \text{Aléatoire} \;=\; \mathrm{IM} && \text{information mutuelle (BALD)}
\end{aligned}
$$

avec $H(p) = -\sum_c p_c \ln p_c$ (en nats) pour la tête genre, et l'entropie de Bernoulli
$H(p) = -p\ln p - (1-p)\ln(1-p)$ moyennée sur les 6 attributs pour la tête attributs. La tête
« combinée » somme les deux.

**Interprétation.**
- **Aléatoire** = l'ambiguïté intrinsèque de l'image (un visage réellement androgyne). Tous les
  membres sont d'accord *pour être incertains*.
- **Épistémique** = le **désaccord** entre membres. Hors distribution, chaque membre extrapole
  différemment, donc ce terme devrait monter. C'est exactement la « variance » des prédictions.

Par concavité de $H$ (inégalité de Jensen), $\text{Total} \geq \text{Aléatoire}$, donc
l'épistémique est **toujours $\geq 0$**.

**Exemple chiffré.** $M = 2$, tête genre. Membre 1 : $p_1 = (0.9,\, 0.1)$. Membre 2 :
$p_2 = (0.1,\, 0.9)$. Désaccord maximal.

- Moyenne : $\bar p = (0.5,\, 0.5)$
- Total $= H(0.5, 0.5) = \ln 2 = 0.693$
- $H(p_1) = -(0.9\ln 0.9 + 0.1\ln 0.1) = 0.0948 + 0.2303 = 0.325$, idem pour $H(p_2)$
- Aléatoire $= 0.325$
- Épistémique $= 0.693 - 0.325 = \mathbf{0.368}$ — élevé, comme attendu.

Si les deux membres avaient tous deux prédit $(0.9, 0.1)$ : Total $= 0.325$, Aléatoire $= 0.325$,
Épistémique $= 0$. Aucun désaccord → aucun signal épistémique.

**Résultats du run** :

| Tête | Total | Aléatoire | Épistémique |
|---|---|---|---|
| genre | 0.4999 | 0.5622 | **0.4587** ⚠️ |
| attributs | 0.6739 | 0.5924 | **0.7036** ✓ |
| combinée | 0.5623 | 0.6435 | 0.5383 |

**Lecture.** L'épistémique sur la tête **attributs** (0.7036) est un bon signal — cohérent avec le
design du dataset : quatre des six attributs sont péri-oculaires, donc occultés par les lunettes,
donc les membres se disputent. À l'inverse, l'épistémique sur le **genre** est à 0.4587, c'est-à-dire
**sous le hasard** : les membres s'accordent *davantage* sur les visages à lunettes que sur les
visages propres. C'est contre-intuitif mais explicable — le membre 0 défaillant (§11) pollue
l'estimation du désaccord.

La tête « combinée » (0.5383), qui somme genre et attributs, est donc **tirée vers le bas par la
composante genre**. Utiliser `unc_epistemic_attrs` plutôt que `unc_epistemic_combined` dans la
fusion par rang est une piste d'amélioration directe pour un prochain run.

### 6.5 Énergie et MSP

Source : `lrad/evaluate.py`, `lrad/fusion.py`.

**MSP** (Maximum Softmax Probability) — la ligne de base historique de la détection OOD :

$$
s_{\text{MSP}}(x) = 1 - \max_c\, p_g(x)_c
$$

Un modèle confiant (proba proche de 1) donne un score bas ; un modèle hésitant (proba proche de
0.5) donne un score haut. AUROC dans ce run : **0.5025** — le hasard. Le MSP est saturé : le
classifieur reste confiant même face à des lunettes.

**Énergie** — beaucoup plus informative, pour un coût identique :

$$
E(x) = -\log \sum_{c} \exp\big(z_g(x)_c\big)
$$

Le code calcule cela via `-np.logaddexp(z_0, z_1)` moyenné sur les membres, ce qui est la forme
numériquement stable pour 2 classes.

**Pourquoi ça marche là où le MSP échoue.** Le softmax **normalise**, et détruit donc l'information
d'amplitude des logits. L'énergie, elle, la conserve. Une entrée hors distribution produit des
logits globalement plus **petits** (le motif ne ressemble à aucune classe apprise), donc une
énergie plus **grande** (moins négative).

**Exemple chiffré.**
- Entrée ID, logits $z = (6, -6)$ : $E = -\log(e^6 + e^{-6}) \approx -6.0$. MSP $= 1 - 0.99999 \approx 10^{-5}$.
- Entrée OOD, logits $z = (2, -1)$ : $E = -\log(e^2 + e^{-1}) = -\log(7.389 + 0.368) = -2.049$.
  MSP $= 1 - 0.953 = 0.047$.

L'énergie a bougé de **4 unités**, le MSP de 0.047 sur une échelle bornée à 1. C'est cette
dynamique qui fait la différence.

Résultat du run : **AUROC 0.7387** pour l'énergie contre 0.5025 pour le MSP. Même tête, mêmes
logits, +0.24 d'AUROC — uniquement en changeant la façon de lire la sortie.

### 6.6 La probabilité CutPaste

$$
s_{\text{cp}}(x) = \frac{1}{M}\sum_{m=1}^{M} \mathrm{softmax}\big(z^m_{cp}(x)\big)_1
= \frac{1}{M}\sum_m P_m(\text{altéré} \mid x)
$$

Résultat : **AUROC 0.6918**, avec moyenne ID = 0.00153 et moyenne OOD = 0.00543.

**Lecture, et elle est importante.** Les deux moyennes sont *minuscules* — le modèle est très sûr
que les vraies images ne sont pas altérées, dans les deux cas. Mais le **rapport** est de 3.5×, et
l'AUROC ne dépend que de l'**ordre** des scores, pas de leur échelle. Le signal est donc réel bien
que faiblement calibré. C'est aussi pourquoi la fusion par rang lui convient parfaitement : elle
n'utilise que l'ordre.

### 6.7 Le biais péri-oculaire (classement des visages à lunettes)

Source : `lrad/ensemble.py:collect_eye_region_bias`. Sert uniquement à **classer** les visages OOD
pour la figure `top_ood_glasses.png` ; ce n'est pas un score OOD évalué en AUROC.

Fenêtre oculaire, en fractions de l'image : lignes 42–62 %, colonnes 15–85 %.

$$
\text{score} \;=\; \bar b_{\text{œil}} \times \frac{\bar b_{\text{œil}}}{\bar b_{\text{global}}}
\;=\; \frac{\bar b_{\text{œil}}^{\,2}}{\bar b_{\text{global}}}
$$

où $\bar b_{\text{œil}}$ est la moyenne de la carte de biais dans la fenêtre et
$\bar b_{\text{global}}$ sa moyenne sur toute l'image.

**Pourquoi cette forme.** Le premier facteur récompense un biais **fort** dans la région des yeux.
Le second est un facteur de **concentration** : un visage dont la reconstruction est mauvaise
partout a un ratio proche de 1 et ne reçoit aucun bonus. Le produit n'est grand que si le biais
est à la fois fort **et** localisé — c'est-à-dire exactement « les lunettes sont bien détectées
comme l'anomalie ».

**Exemple, le visage classé n°1 du run** (`top_ood_glasses.json`) :

$$
\bar b_{\text{œil}} = 0.1536, \quad \bar b_{\text{global}} = 0.0353
\;\Longrightarrow\;
\text{score} = \frac{0.1536^2}{0.0353} = 0.668
$$

Le biais est 4.35× plus fort dans la région des yeux que sur l'ensemble du visage.

---

## 7. Fusion

Source : `lrad/fusion.py`. Aucun signal seul ne franchit 0.80. Mais ils se trompent sur des images
**différentes** — donc les combiner aide.

### 7.1 Fusion par rang (sans étiquettes)

$$
r(s)_i = \frac{\operatorname{rang}(s_i)}{N - 1} \in [0, 1],
\qquad
\text{fusion}_i = \frac{1}{K}\sum_{j=1}^{K} r\big(s^{(j)}\big)_i
$$

Implémentation : `s.argsort().argsort()` — un double argsort donne le rang de chaque élément.

**Trois propriétés qui font tout l'intérêt de cette méthode.**

1. C'est une transformation **monotone** par signal → l'AUROC de chaque signal pris isolément est
   **inchangée**. On ne détruit aucune information de classement.
2. Elle est **sans étiquettes** — aucune donnée de calibration nécessaire.
3. Elle est **immune aux échelles**. Comparez : `locfre_b1` vit vers 3.0, `ens_energy_gender` vers
   −6.5, `cutpaste_prob` vers 0.003. Une moyenne brute serait entièrement dominée par l'énergie.
   Après passage aux rangs, les trois pèsent pareil.

**Exemple chiffré.** 4 images, 2 signaux.

| Image | $s^{(1)}$ | rang | $r$ | $s^{(2)}$ | rang | $r$ | fusion |
|---|---|---|---|---|---|---|---|
| A | 0.1 | 0 | 0.000 | 900 | 2 | 0.667 | 0.333 |
| B | 0.9 | 3 | 1.000 | 950 | 3 | 1.000 | **1.000** |
| C | 0.4 | 1 | 0.333 | 100 | 0 | 0.000 | 0.167 |
| D | 0.7 | 2 | 0.667 | 500 | 1 | 0.333 | 0.500 |

L'image B, en tête sur les deux signaux, remporte la fusion. L'image A, forte sur $s^{(2)}$ mais
dernière sur $s^{(1)}$, est ramenée au milieu. Notez que l'échelle absurdement différente des deux
signaux (0–1 contre 100–950) n'a aucun effet.

**Les 6 signaux fusionnés dans ce run** (`rank_signals`) :

```
locfre_b1, locfre_b2, locfre_b3, unc_epistemic_combined,
ens_energy_gender, cutpaste_prob
```

**Résultat : AUROC 0.8120**, contre 0.7898 pour le meilleur signal isolé. Le gain est modeste
(+0.022) mais réel, et obtenu **sans aucune étiquette OOD**. C'est le chiffre à citer pour un
détecteur réellement déployable.

### 7.2 Fusion supervisée (régression logistique)

$$
\tilde x_j = \frac{x_j - \mu_j}{\sigma_j},
\qquad
\text{score}(x) = \sum_{j=1}^{10} w_j\,\tilde x_j + b
$$

Le score est le **logit** de $P(\text{OOD} \mid x)$. Comme l'AUROC est invariante par
transformation monotone, travailler sur le logit plutôt que sur la probabilité ne change rien au
résultat.

Le protocole de découpe (§2.4) garantit que les positifs de calibration sont **disjoints** des
positifs d'évaluation, et que les négatifs viennent de `train`, jamais de `test_in`.

**Poids appris** (`fused_auroc.json`) :

| Signal | $\mu$ | $\sigma$ | $w$ |
|---|---|---|---|
| `ens_msp_gender` | 0.0671 | 0.1027 | **+2.756** |
| `ens_energy_gender` | −6.462 | 2.578 | +1.221 |
| `unc_epistemic_combined` | 0.2759 | 0.1710 | +1.051 |
| `locfre_b1` | 3.510 | 1.026 | +0.779 |
| `unc_total_combined` | 0.5357 | 0.2274 | +0.485 |
| `locfre_b3` | 2.044 | 1.014 | +0.257 |
| `cutpaste_prob` | 0.00336 | 0.01806 | +0.226 |
| `locfre_b2` | 2.928 | 1.190 | +0.038 |
| `unc_epistemic_gender` | 0.1255 | 0.1631 | **−1.906** |
| `unc_total_gender` | 0.1737 | 0.2117 | **−2.777** |

Intercept $b = 0.0603$. **Résultat : AUROC 0.8838.**

**Trois choses à lire dans ce tableau.**

1. **`ens_msp_gender` reçoit le plus gros poids positif (+2.756) alors que son AUROC isolée est de
   0.5025 — le hasard.** Ce n'est pas une contradiction : la régression logistique exploite
   l'information *conditionnelle*. Le MSP est inutile seul, mais utile *sachant* l'énergie et
   l'incertitude — il corrige les autres signaux plutôt que de discriminer lui-même.

2. **Deux poids fortement négatifs**, tous deux sur la tête genre. Le modèle a appris que ces
   signaux sont **anti-corrélés** avec l'OOD, ce qui recoupe exactement l'observation de §6.4 :
   `unc_epistemic_gender` a une AUROC de 0.4587, donc **sous** 0.5, donc inverser son signe le rend
   informatif. La régression fait automatiquement cette correction de signe ; la fusion par rang,
   qui suppose « grand = plus OOD », ne le peut pas. C'est une grande partie de l'écart 0.812 →
   0.884.

3. `locfre_b2` reçoit un poids quasi nul (+0.038) malgré une AUROC isolée de 0.7735 — il est
   presque entièrement **redondant** avec `locfre_b1` et `locfre_b3`.

**Exemple d'application.** Une image dont `locfre_b1` vaut 5.0 :
$\tilde x = (5.0 - 3.510)/1.026 = 1.452$, contribution $= 0.779 \times 1.452 = +1.131$ au logit.
On additionne les 10 contributions et l'intercept ; le résultat est le score final.

> ⚠️ La fusion supervisée voit des étiquettes OOD pendant la calibration. C'est une **borne
> supérieure optimiste**. Un détecteur déployé face à un type d'anomalie inconnu n'aurait pas ces
> étiquettes et retomberait sur la fusion par rang (0.8120).

---

## 8. AUROC — la métrique

Source : `lrad/evaluate.py:_auroc_entry`, via `sklearn.metrics.roc_auc_score`.

On empile les scores de `test_in` (label 0) et `test_ood` (label 1), puis on lit l'aire sous la
courbe ROC. La définition la plus utile pour l'intuition est probabiliste :

$$
\mathrm{AUROC} \;=\; P\big(\, s(X_{\text{OOD}}) > s(X_{\text{ID}}) \,\big)
\;+\; \tfrac12\, P\big(\, s(X_{\text{OOD}}) = s(X_{\text{ID}}) \,\big)
$$

pour une paire tirée uniformément au hasard, une image OOD et une image ID.

- 0.5 = hasard pur.
- 1.0 = séparation parfaite.
- **Sous 0.5 = le score est informatif mais dans le mauvais sens** — l'inverser donne $1 - \text{AUROC}$.

**Exemple chiffré.** $s_{\text{ID}} = \{0.1,\, 0.4\}$, $s_{\text{OOD}} = \{0.3,\, 0.6\}$.
Les $2 \times 2 = 4$ paires :

| paire | OOD > ID ? |
|---|---|
| 0.3 vs 0.1 | ✓ |
| 0.3 vs 0.4 | ✗ |
| 0.6 vs 0.1 | ✓ |
| 0.6 vs 0.4 | ✓ |

$\mathrm{AUROC} = 3/4 = 0.75$.

**Pourquoi l'AUROC et pas l'exactitude.** Elle est indépendante du **seuil** (on n'a pas à en
choisir un) et insensible au **déséquilibre des classes** (18 941 contre 6 597 ici). Interpréter
`auroc_msp = 0.2602` pour le membre 0 : ce membre classe les visages à lunettes comme *plus
confiants* que les visages propres, dans 74 % des paires. Inversé, ce serait un score de 0.74 —
symptôme direct de la défaillance décrite en §11.

---

## 9. Interprétation de chaque figure

### 9.1 `ensemble/plots/` — les figures d'ensemble

Les figures en grille utilisent 25 échantillons tirés au hasard — **10 ID** (seed 1234) puis
**15 OOD** (seed 1235) —, étiquetés `ID 1…10` / `OOD 1…15` en marge de ligne.

---

**`ensemble_decomposition.png`** — *la figure fondamentale*

Pour chaque image (une ligne) et chaque bloc, trois tuiles : Risque | Biais | Variance.

*Comment lire.* Vérifiez visuellement l'identité : la tuile Risque doit être la somme pixel à pixel
des deux autres. Sur les lignes OOD, cherchez une zone chaude sur la monture des lunettes. Vous
constaterez qu'elle est présente mais **noyée** dans l'erreur des cheveux et du fond — c'est le
diagnostic visuel du plafond à 0.62 des scores pixel.

*Maths.* Voir §5. Cartes calculées par `decomposition_maps`, échelle de couleur en $[0, 3]$.

---

**`decomposition_auroc.png`** — barres d'AUROC par bloc

Trois groupes de barres (Risque / Biais / Variance), une barre par bloc, plus la valeur agrégée.

*Comment lire.* La tendance est monotone croissante jusqu'au bloc 4, puis redescend au bloc 5 :

| | bloc 1 | bloc 2 | bloc 3 | bloc 4 | bloc 5 |
|---|---|---|---|---|---|
| Risque | 0.5730 | 0.5944 | 0.6288 | **0.6461** | 0.6232 |
| Biais | 0.5547 | 0.5763 | 0.6083 | **0.6349** | 0.6209 |
| Variance | 0.5783 | 0.6055 | 0.6540 | **0.6667** | 0.6217 |

*Pourquoi ce pic au bloc 4.* Les blocs superficiels reconstruisent trop bien : ils copient la
texture, lunettes comprises, donc ne détectent rien. Les blocs profonds reconstruisent trop mal :
tout est flou, l'erreur est dominée par le flou et non par l'anomalie. Le bloc 4 est le compromis —
assez sémantique pour ne pas savoir dessiner des lunettes, assez résolu pour localiser où elles
sont.

> Attention à la numérotation. Le code indexe les blocs de **0 à 4** et les figures les étiquettent
> `L0…L4` ; ce rapport les numérote de **1 à 5**. Le pic « bloc 4 » du tableau ci-dessus correspond
> donc à `L3` dans les figures. Les figures par instance et `score_comparison.png`, elles, sont
> produites au bloc `L4` — le cinquième et dernier.

---

**`ensemble_score_hists.png`** — histogrammes des scores agrégés

Une paire d'histogrammes ID/OOD (densité normalisée) par terme.

*Comment lire.* Le **recouvrement** des deux distributions est le complément visuel de l'AUROC. Ici
le recouvrement est large (AUROC ≈ 0.62–0.64), ce qui se traduit par deux cloches presque
superposées, la cloche OOD légèrement décalée à droite. Comparez avec le panneau droit de
`fused_auroc.png` (AUROC 0.884) pour voir à quoi ressemble une vraie séparation.

---

**`bias_variance_vs_block.png`** — évolution avec la profondeur

Deux panneaux (Biais, Variance). Chacun trace la moyenne du score par bloc, ID contre OOD, avec une
bande de ±1 écart-type.

*Comment lire.* Le score augmente avec la profondeur dans les **deux** classes (les reconstructions
profondes sont plus mauvaises pour tout le monde). Ce qui compte n'est pas la hauteur des courbes
mais **l'écart entre les deux**, rapporté à la largeur des bandes. Une bande qui recouvre l'autre
courbe signifie que ce bloc ne sépare pas de façon fiable image par image, même si les moyennes
diffèrent.

---

**`bias_variance_vs_percentile.png`** — séparation en fonction du percentile

Pour $q$ balayant $[1, 99]$, on trace le $q$-ième percentile de la distribution des scores ID et
celui des scores OOD.

*Maths.* Si $F_{\text{ID}}$ et $F_{\text{OOD}}$ sont les fonctions de répartition, on trace les
fonctions quantiles $F^{-1}_{\text{ID}}(q)$ et $F^{-1}_{\text{OOD}}(q)$. C'est un **Q–Q plot** entre
les deux distributions, déplié le long de $q$.

*Comment lire.* L'écart vertical entre les deux courbes est la séparabilité. Un écart qui se creuse
vers $q \to 99$ signifie que le score ne distingue que les cas **extrêmes** ; un écart constant
signifie un décalage uniforme de toute la distribution, ce qui est bien meilleur pour un détecteur.

---

**`mean_recon_breakdown.png`** — Original | Err Lk | Recon Lk, sur la reconstruction consensus

Pour chaque bloc : l'erreur $(x - \bar f_k)^2$ (le terme de biais) et la reconstruction moyenne
$\bar f_k$ elle-même.

*Comment lire.* Suivez une ligne OOD de gauche à droite : les reconstructions se dégradent
progressivement en visages de plus en plus génériques. C'est visuellement l'affirmation centrale du
projet : **$\bar f$ est un visage propre**, il ne contient pas de lunettes. C'est ce qui rend
$x - \bar f$ informatif, et c'est ce que `locfre` exploite en repartant de $\bar f_4$.

---

**`mean_recons_only.png`** — les reconstructions consensus seules

Même contenu, sans les cartes d'erreur. Utile pour vérifier d'un coup d'œil la qualité de
reconstruction bloc par bloc, ou pour une planche de présentation.

---

**`mean_abs_bias.png`** — le terme de biais seul, par bloc

Cartes $(x - \bar f_k)^2$ uniquement.

*Comment lire.* La vue la plus propre pour juger si les lunettes émergent, sans être distrait par
le risque et la variance. Échelle de couleur partagée entre toutes les tuiles → les blocs sont
directement comparables entre eux.

---

**`mean_error_maps.png`** — le terme de risque, par bloc

$\frac{1}{M}\sum_m (x - \hat f^m_k)^2$ : la moyenne des cartes d'erreur.

*Distinction cruciale à ne pas confondre avec la figure précédente.*

- `mean_error_maps` = **moyenne des erreurs** = Risque
- `mean_abs_bias` = **erreur de la moyenne** = Biais

L'écart entre les deux figures, pixel par pixel, **est** la variance. Les regarder côte à côte est
la façon la plus directe de voir où les modèles sont d'accord et où ils divergent.

---

**`min_error_maps.png`** — l'erreur du meilleur membre

$\min_m (x - \hat f^m_k)^2$, pixel par pixel.

*Comment lire.* Répond à « à quel point le **meilleur** membre reconstruit-il ce pixel ». Une
région ne reste claire que si **aucun** modèle de l'ensemble n'y arrive — c'est un signal OOD plus
fort que la moyenne, qu'un seul mauvais membre peut gonfler. Note d'affichage : cette tuile a sa
**propre** échelle de couleur, car le minimum vit à une magnitude bien inférieure au biais ; une
échelle partagée la rendrait uniformément noire.

---

**`variance_heatmaps_ood.png`** et **`variance_heatmaps_all.png`** — le désaccord

Cartes de variance, sur les OOD seuls puis sur ID + OOD.

*Comment lire.* Cherchez si le désaccord se concentre sur les lunettes. Le résultat du run dit qu'il
s'y concentre partiellement (variance agrégée : AUROC 0.6431, le meilleur des trois termes pixel),
mais le mécanisme reste faible : sur une **occlusion**, tous les membres échouent au même endroit,
donc l'erreur est corrélée. Le désaccord utile est ailleurs — dans les prédictions (§6.4).

---

**`score_comparison.png`** — quatre scores au bloc 5, côte à côte

Colonnes : Biais $(x-\bar f)^2$ | Risque $\text{mean}_m$ | $\min_m$ | 3ᵉ plus petit.

*Maths de la 4ᵉ colonne.* Le **minimum robuste** : les $M$ erreurs par modèle sont triées par ordre
croissant et on garde la $k$-ième plus petite (`torch.kthvalue`, $k = 3$ par défaut). $k=1$
redonne le minimum exact ; $k > 1$ rend le minimum robuste à quelques membres chanceux — un pixel
ne reste sombre que si **au moins $k$** modèles le reconstruisent bien.

*Comment lire.* Chaque colonne garde sa **propre** échelle de couleur (les quatre scores n'ont pas
la même distribution). Ne comparez donc pas les intensités absolues entre colonnes, seulement les
**motifs spatiaux**.

---

**`top_ood_glasses.png`** — les 10 visages à lunettes les mieux détectés

Trois lignes par visage : Original | Biais | Superposition, classés par rang.

*Maths du classement.* Voir §6.7 : $\text{score} = \bar b_{\text{œil}}^2 / \bar b_{\text{global}}$,
sur les 13 193 images OOD complètes, restreint à celles portant réellement l'attribut `Eyeglasses`.
Le classement complet est dans `top_ood_glasses.json`.

*Comment lire.* **C'est une figure de meilleurs cas, pas une évaluation.** Elle montre à quoi
ressemble le pipeline **quand il fonctionne** — et il fonctionne visiblement bien sur ces
dix-là (facteur 4.35 entre biais oculaire et biais global pour le n°1). Elle ne dit rien de la
performance moyenne, qui est donnée par les AUROC.

*Note d'affichage.* La superposition applique `smooth_cam` : rectification à $\geq 0$,
normalisation par le pic, flou gaussien $\sigma = 1.5$, renormalisation, puis alpha
$= \text{cam}^{0.8}$. Ce sont des réglages **purement cosmétiques**, jamais présents dans le score.

---

**`fused_auroc.png`** — le panneau récapitulatif de la fusion

Trois panneaux : barres d'AUROC par signal (les fusions en couleur d'accent, ligne de hasard à 0.5)
| courbes ROC de la meilleure fusion et du meilleur signal isolé | histogrammes ID/OOD du score
fusionné supervisé.

*Comment lire.* **C'est la figure à montrer en premier.** Le panneau de gauche donne la hiérarchie
complète des signaux d'un coup d'œil, du hasard (`ens_msp_gender`, 0.502) au sommet
(`fused_supervised`, 0.884). Le panneau de droite montre une séparation ID/OOD nettement visible —
à comparer avec `ensemble_score_hists.png` pour mesurer le chemin parcouru.

---

### 9.2 `ensemble/plots/instances_in/` et `instances_ood/` — étude par visage

20 visages ID et 20 visages OOD, tirés au hasard (seeds 1334 et 1335), chacun dans son dossier
`ID_XX/` ou `OOD_XX/`, reconstruits au **bloc L4**.

| Fichier | Contenu |
|---|---|
| `model_01.png` … `model_08.png` | Pour **un** membre : Original \| Reconstruction \| Erreur $(x - \hat f^m)^2$. Échelle fixe $[0, 0.5]$ partagée → les 8 fichiers sont directement comparables. |
| `summary.png` | La vue consensus, 5 tuiles : Original \| Biais $(x-\bar f)^2$ \| Erreur moyenne \| Erreur min \| Superposition du biais lissé. |
| `all_models.png` | Superposition d'erreur de **chaque** membre côte à côte, puis les trois résumés (biais / moyenne / min). |

*Comment les utiliser.* `all_models.png` est l'outil de diagnostic : il rend le **désaccord**
directement visible. Si les 8 tuiles membres se ressemblent, la variance est faible et l'échec est
systématique (typique d'une occlusion). Si elles diffèrent, la variance est forte et le signal
épistémique devrait se déclencher.

Les dossiers frères **`instances_in_raw/`** et **`instances_ood_raw/`** contiennent les mêmes cartes
en PNG nus — `original.png`, `bias.png`, `bias_overlay.png`, `mean_error.png`, `min_error.png` —
sans titres, axes ni barres de couleur. Ils sont **pixel-identiques** aux tuiles de `summary.png`
(mêmes dérivations via `_instance_summary_maps`), prêts à être placés dans un article ou une
présentation.

### 9.3 `model_i/plots/` — les figures par membre (8 × 10 fichiers)

| Fichier | Contenu et lecture |
|---|---|
| `training_history.png` | Perte et exactitude par époque. Sans split de validation, seules les courbes d'entraînement existent. Vérifiez que la perte descend sans plateau brutal. |
| `batch_accuracy.png` | Exactitude genre et attributs **par batch**, avec des marqueurs de fin d'époque. Beaucoup plus fin que la courbe par époque : révèle l'instabilité intra-époque. |
| `per_block_breakdown.png` | Original \| Err Lk \| Recon Lk pour ce membre seul. À comparer avec `mean_recon_breakdown.png` de l'ensemble : un membre isolé reconstruit visiblement moins bien que le consensus. |
| `recons_only.png` | Les 5 reconstructions de ce membre, sans cartes d'erreur. |
| `activations.png` | Moyenne sur les canaux des activations de chaque bloc, superposée en `inferno` sur le visage. **Montre où le modèle met son énergie à chaque profondeur.** Chaque bloc a sa propre échelle (les magnitudes profondes diffèrent d'un ordre de grandeur). |
| `fusion_overlay.png` | Original \| Err L0…L4 \| Fusionné (max) \| Superposition. La carte fusionnée est le **max par pixel sur les blocs**, échelle fixe $[0, 0.5]$. |
| `fusion_auroc.png` | Le score de fusion mono-modèle : histogramme + ROC. Ce score est $\max_p \max_k e_k(x)[p]$ — le pixel le plus surprenant, tous blocs confondus. Membre 0 : AUROC 0.6144. |
| `roc_ood.png` | Les 4 courbes ROC du classifieur (MSP, entropie genre / attributs / combinée) sur une même figure. |
| `score_dist_score_msp.png` | Histogramme ID vs OOD du score MSP. Pour les membres sains, la cloche OOD est décalée à droite ; pour le membre 0, elle est décalée **à gauche** (d'où AUROC 0.26). |
| `score_dist_score_entropy_combined.png` | Idem pour l'entropie combinée. |

---

## 10. Tableaux de résultats

### 10.1 Décomposition pixel — `ensemble/summary.json`

*18 941 ID vs 13 193 OOD, réduction p95, moyenne uniforme des blocs.*

| Terme | bloc 1 | bloc 2 | bloc 3 | bloc 4 | bloc 5 | **agrégé** |
|---|---|---|---|---|---|---|
| Risque | 0.5730 | 0.5944 | 0.6288 | 0.6461 | 0.6232 | **0.6310** |
| Biais | 0.5547 | 0.5763 | 0.6083 | 0.6349 | 0.6209 | **0.6237** |
| Variance | 0.5783 | 0.6055 | 0.6540 | 0.6667 | 0.6217 | **0.6431** |

### 10.2 Score localisé — `ensemble/localized_auroc.json`

*Mêmes splits complets. Fenêtres patch-max 4/8/16 px, référence : 5 120 images de `train`.*

| Terme | b1 | b2 | b3 | b4 | b5 | **z agrégé** | ligne de base p95 |
|---|---|---|---|---|---|---|---|
| Risque | 0.6763 | 0.6488 | 0.6704 | 0.6954 | 0.6855 | **0.7016** | 0.6310 |
| Biais | 0.5882 | 0.6395 | 0.6596 | 0.6888 | 0.6759 | **0.6802** | 0.6237 |
| Variance | 0.6486 | 0.6411 | 0.6675 | 0.6834 | 0.6759 | **0.6936** | 0.6431 |

### 10.3 Signaux de fusion — `ensemble/fused_auroc.json`

*18 941 ID vs 6 597 OOD (moitié d'évaluation). Blocs locfre 1, 2, 3.*

| Signal | AUROC | moyenne ID | moyenne OOD |
|---|---|---|---|
| **`fused_supervised`** | **0.8838** | −1.382 | +1.644 |
| **`fused_rank`** | **0.8120** | 0.443 | 0.663 |
| `locfre_b3` | 0.7898 | 1.570 | 2.592 |
| `locfre_b1` | 0.7838 | 3.023 | 4.016 |
| `locfre_b2` | 0.7735 | 2.387 | 3.511 |
| `ens_energy_gender` | 0.7387 | −7.509 | −5.334 |
| `cutpaste_prob` | 0.6918 | 0.00153 | 0.00543 |
| `unc_total_combined` | 0.5632 | 0.516 | 0.569 |
| `unc_epistemic_combined` | 0.5386 | 0.2757 | 0.2820 |
| `unc_total_gender` | 0.5025 | 0.1769 | 0.1825 |
| `ens_msp_gender` | 0.5025 | 0.0616 | 0.0801 |
| `unc_epistemic_gender` | 0.4609 ⚠️ | 0.1412 | 0.1137 |

### 10.4 Incertitude prédictive — `ensemble/summary.json`

| Tête | Total | Aléatoire | Épistémique |
|---|---|---|---|
| genre | 0.4999 | 0.5622 | 0.4587 |
| attributs | 0.6739 | 0.5924 | **0.7036** |
| combinée | 0.5623 | 0.6435 | 0.5383 |

---

## 11. Lecture critique et anomalies

### 11.1 Le membre 0 est défaillant à l'évaluation

C'est l'observation la plus importante de ce run.

| | à l'entraînement (époque 20) | à l'évaluation (`test_in`) |
|---|---|---|
| Exactitude genre | **99.5 %** | **63.0 %** |
| `Arched_Eyebrows` | 86.7 % | 75.0 % |
| `Smiling` | 94.7 % | 71.0 % |

Les sept autres membres évaluent tous entre 97.5 % et 98.5 %. Le membre 0 est le seul à s'effondrer,
et il s'effondre **uniquement en mode `eval()`**, alors que ses métriques d'entraînement (mode
`train()`) sont excellentes — les meilleures de l'ensemble, même.

**Hypothèse la plus probable : divergence des statistiques courantes de BatchNorm.** En mode
`train()`, BatchNorm normalise avec les statistiques **du batch courant** ; en mode `eval()`, avec
les moyennes courantes accumulées (`running_mean` / `running_var`). Si ces dernières ont divergé
des statistiques réelles, le modèle fonctionne parfaitement pendant l'entraînement et s'effondre à
l'inférence. La `momentum` par défaut de PyTorch (0.1) combinée à l'augmentation CutPaste — qui
change la distribution des activations d'un batch à l'autre — est un mécanisme plausible, mais
ceci reste une hypothèse : **elle n'a pas été vérifiée** dans ce run.

**Conséquences mesurables :**

- `auroc_msp = 0.2602` pour ce membre : ses prédictions sont **anti-corrélées** avec l'OOD.
- `unc_epistemic_gender = 0.4587` au niveau de l'ensemble, sous le hasard — un membre qui prédit
  n'importe quoi de façon *cohérente* fausse l'estimation du désaccord.
- La régression logistique le détecte et lui applique un poids **négatif** (−1.906), ce qui est
  précisément pourquoi la fusion supervisée dépasse la fusion par rang de 7 points.

**Vérification suggérée** avant tout autre travail : recharger `model_0/weights/model.pt`,
recalculer les statistiques BatchNorm sur une passe de `train` en mode `train()` sans gradient,
puis réévaluer. Si l'exactitude remonte à ~98 %, l'hypothèse est confirmée et tous les chiffres
d'ensemble de ce run sont à reprendre — ils sont pessimistes.

### 11.2 L'hypothèse de départ n'est pas confirmée

Le projet postule (documenté dans `lrad/anomaly_score.py`) que le **biais** est le bon score
d'anomalie. Sur ce run, il est le **plus faible** des trois termes pixel (0.6237 contre 0.6431 pour
la variance). Ce n'est pas un bug — c'est un résultat, et il est cohérent avec la nature de la
tâche : sur une occlusion, l'échec est corrélé entre membres. Les docstrings du code reflètent
encore la hiérarchie attendue plutôt que celle mesurée.

### 11.3 Deux jeux de chiffres non comparables

`localized_auroc.json` évalue sur 13 193 OOD ; `fused_auroc.json` sur 6 597. Il est tentant de
comparer 0.7016 (z-risque) à 0.7898 (`locfre_b3`) — c'est **approximatif** : les deux moitiés OOD
étant issues d'une permutation aléatoire du même pool, elles devraient être statistiquement
similaires, mais l'écart n'est pas mesuré sur les mêmes images.

### 11.4 Réglages sous-exploités

- **`scar_prob = 0`.** Aucune « balafre » fine n'a été générée pendant l'entraînement CutPaste,
  alors que c'est la forme la plus proche d'une monture de lunettes. Le remettre à 0.5 est le test
  le plus prometteur à moindre coût.
- **`unc_epistemic_attrs` (0.7036) n'est pas dans la fusion par rang**, qui utilise
  `unc_epistemic_combined` (0.5383). Le simple échange devrait faire monter la fusion par rang.
- **`locfre_b2` est redondant** (poids appris +0.038). Le retirer allégerait le calcul sans coût
  mesurable.

### 11.5 Le membre 0 tire l'ensemble vers le bas

Les scores d'ensemble sont calculés sur les 8 membres, membre 0 inclus. Sa reconstruction entre
dans $\bar f$, ses prédictions entrent dans l'estimation de l'incertitude. Un run de contrôle à 7
membres (le membre 0 exclu) donnerait une idée directe du coût de cette défaillance.

---

## 12. Reproduire ce run

```bash
# 1. Entraînement de l'ensemble + décomposition (≈ 10 h 30 sur A40)
python scripts/run_ensemble.py \
    --config configs/celeba_ood_cutpaste128.yaml \
    --output-dir outputs/celeba_ood/cp128_<horodatage>

# 2. Score localisé (z-score + patch-max), pas de réentraînement
python scripts/run_localized.py \
    --output-dir outputs/celeba_ood/cp128_<horodatage>

# 3. Fusion supervisée + panneau récapitulatif
python scripts/run_fused.py \
    --output-dir outputs/celeba_ood/cp128_<horodatage> \
    --blocks 1 2 3 --supervised
```

Ne relancez que l'étape 2 et 3 pour ré-évaluer un ensemble déjà entraîné ; ajoutez `--eval-only` à
l'étape 1 pour recalculer la décomposition et les figures depuis les poids sauvegardés.

### Carte des fichiers de sortie

```
cp128_20260719_060210_6782570/
├── config.resolved.yaml          configuration effective, après résolution
├── logs/                         un log par job OAR
├── model_0/ … model_7/           un dossier par membre
│   ├── config.resolved.yaml      l'architecture de CE membre
│   ├── weights/{model,decoders}.pt
│   ├── history.json              courbes d'entraînement du classifieur
│   ├── decoders_history.json     MSE par bloc et par époque
│   ├── summary.json              exactitude + AUROC de ce membre
│   └── plots/                    10 figures (voir §9.3)
└── ensemble/
    ├── summary.json              décomposition + incertitude, par bloc et agrégé
    ├── localized_auroc.json      scores z + patch-max
    ├── fused_auroc.json          signaux, calibration, AUROC fusionnées
    ├── top_ood_glasses.json      classement des 10 meilleurs cas
    └── plots/                    15 figures + 40 dossiers d'instances (voir §9.1, §9.2)
```

### Carte du code

| Module | Rôle |
|---|---|
| `lrad/dataset.py` | Découpe CelebA in/OOD, chargeurs, `split_loader` |
| `lrad/model.py` | `FacialCNN` — tronc 5 blocs + 3 têtes |
| `lrad/cutpaste.py` | Augmentation CutPaste (rectangles et balafres) |
| `lrad/decoder.py` | `BlockDecoder` — un décodeur par bloc |
| `lrad/train.py` | Les deux boucles d'entraînement (classifieur, décodeurs) |
| `lrad/ensemble.py` | Décomposition Risque/Biais/Variance, incertitude prédictive |
| `lrad/anomaly_score.py` | Réductions pixel → scalaire (mean / max / p95) |
| `lrad/localized.py` | z-score par pixel + patch-max multi-échelle |
| `lrad/feature_error.py` | Le signal `locfre` |
| `lrad/fusion.py` | Fusion par rang et fusion logistique supervisée |
| `lrad/evaluate.py` | MSP, entropies, AUROC |
| `lrad/plots.py` | Toutes les figures |
