# Run `baseline` (ablation) — rapport complet

*Ensemble profond de **10 membres à architecture identique**, CelebA 128 px, sans CutPaste,
OOD = attribut `Eyeglasses`. Job OAR 6866854, GPU NVIDIA L40S, torch 2.5.1+cu121, du
19 août 2026 13:38 au 19 août 2026 21:39 (≈ 8 h 01 min). Répertoire :
`outputs/celeba_ood/ablation/baseline_20260819_133802_6866854/`, config
[`configs/ablation_baseline.yaml`](../../../../configs/ablation_baseline.yaml).*

Ce document explique **ce qui a été calculé**, **avec quelles formules**, **comment lire chaque
figure produite**, et **ce que les chiffres veulent dire**. Les mathématiques dures sont
accompagnées d'un petit exemple entièrement chiffré.

Tous les nombres cités proviennent des JSON du run (`ensemble/summary.json`,
`ensemble/localized_auroc.json`, `ensemble/fused_auroc.json`, `ensemble/top_ood_glasses.json`,
`model_i/summary.json`, `model_i/decoders_history.json`) et du log
`logs/celeba_ood_celeba_ood_abl_baseline_ensemble_oar6866854.log`.

> **C'est le bras de contrôle de l'étude d'ablation** — et, à ce jour, le **meilleur run du
> dépôt** sur les scores fusionnés. Il n'a *aucun* des deux ingrédients que l'ablation teste :
> pas de diversité d'architecture (`ensemble.member_variants` absent), pas de tâche prétexte
> CutPaste. Les dix membres ne diffèrent que par leur graine. Voir [§11.6](#116-comparaison-avec-lastof_results-et-ce-quon-ne-peut-pas-en-conclure).

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

**La méthode.** Dix modèles indépendants — **même architecture, graines différentes** (42…51) —
sont entraînés. Chacun porte :

- un tronc convolutif à 5 blocs, avec deux têtes : genre et 6 attributs faciaux ;
- cinq **décodeurs**, un par bloc, qui reconstruisent l'image d'entrée depuis les activations de ce
  bloc, le tronc étant gelé.

On en tire une famille de signaux OOD, puis on les fusionne.

**Le résultat.**

| Score | AUROC | Commentaire |
|---|---|---|
| Biais pixel p95 (ligne de base historique) | **0.6246** | plafonne — les décodeurs sont flous |
| Variance pixel p95 | **0.6510** | meilleur des trois termes pixel |
| z-score + patch-max sur le risque | **0.6988** | la normalisation par pixel aide beaucoup |
| `unc_epistemic_combined` (désaccord des têtes) | **0.7336** | sain ici, contrairement au run archivé |
| `locfre_b1` (erreur de features, bloc 2) | **0.7833** | meilleur des trois `locfre` |
| Énergie de la tête genre | **0.8112** | **meilleur signal isolé**, et gratuit |
| Fusion par rang, 5 signaux, sans étiquettes | **0.8420** | ce qu'un détecteur déployable atteindrait |
| Fusion supervisée (régression logistique) | **0.8730** | borne haute, voit des étiquettes OOD à la calibration |

**Le message.** Le gain ne vient pas d'un score magique mais de trois idées empilées :
(1) quitter l'espace des pixels pour l'espace des **features** et des **logits**, (2) **normaliser
par position** avant de réduire, (3) **fusionner** des signaux qui se trompent sur des images
différentes. Le passage 0.625 → 0.842 **sans aucune étiquette OOD** est le résultat de ce run.

**Ce qui distingue ce run.** C'est le seul du dépôt où les dix membres sont architecturalement
identiques. Trois conséquences mesurables :

1. **Aucun membre ne s'effondre.** Exactitude genre entre 0.941 et 0.982, AUROC de reconstruction
   entre 0.6290 et 0.6399 — une étendue de **1.1 point**. Le run archivé `LASTOF_RESULTS` avait
   deux membres dégénérés à l'évaluation.
2. **La chaîne « têtes » devient exploitable.** L'énergie genre passe à 0.8112 (contre 0.7127) et
   l'épistémique combinée à 0.7336 (contre 0.5387) — précisément parce que la tête genre n'est
   plus polluée par des membres défaillants.
3. **La variance pixel reste faible** (0.6510), comme attendu : dix membres quasi identiques
   reconstruisent quasi identiquement. Elle bat pourtant le biais.

---

## 2. Données et protocole de découpe

### 2.1 Définition de l'OOD

`dataset.ood_attrs: [Eyeglasses]`. Une image CelebA est **OOD** si et seulement si l'attribut
`Eyeglasses` vaut +1 — ce qui inclut les lunettes de soleil. Tout le reste est
**in-distribution (ID)**.

```
OOD attributes: Eyeglasses  →  in-dist=189406  ood=13193 images
```

L'anomalie est donc une **occlusion locale et sémantique** : un objet réel, absent de tout
l'entraînement, posé sur une région précise du visage. Ce n'est ni du bruit, ni un changement de
domaine global.

### 2.2 Découpe

| Split | Taille | Rôle |
|---|---|---|
| `train` | 170 465 | entraînement classifieur **et** décodeurs (visages sans lunettes) |
| `val` | 0 | `val_ratio = 0.00` — aucun split de validation |
| `test_in` | 18 941 | négatifs de l'évaluation OOD |
| `test_ood` | 13 193 | positifs de l'évaluation OOD |

`train_ratio = 0.90` sur les 189 406 images ID. **`val_ratio = 0` est un choix délibéré** : sans
validation, pas d'arrêt anticipé, le schedule complet (20 + 25 époques) est exécuté et les poids
de la dernière époque sont conservés. Le log le dit explicitement :

```
val_ratio=0 → no validation split: 170465 in-dist images train, 18941 held out for
test_in (no early stopping, full schedule).
```

Conséquence à garder en tête : **rien ne surveille le sur-apprentissage**. C'est acceptable ici
parce que la métrique finale (AUROC OOD) ne dépend pas de l'exactitude du classifieur, mais cela
interdit de lire les courbes d'entraînement comme des courbes de généralisation.

### 2.3 Étiquettes prédites

Tête genre : `Male`, taux de positifs 39.0 % dans `train`.

Tête attributs, 6 cibles, dont **4 péri-oculaires** (les deux dernières sont globales) :

| Attribut | Péri-oculaire | Taux de positifs (train) |
|---|---|---|
| `Arched_Eyebrows` | ✓ | 28.4 % |
| `Bushy_Eyebrows` | ✓ | 14.9 % |
| `Narrow_Eyes` | ✓ | 11.9 % |
| `Bags_Under_Eyes` | ✓ | 20.9 % |
| `Smiling` | — | 48.7 % |
| `Young` | — | 79.9 % |

**C'est le cœur du design.** Le classifieur ne voit que des visages sans lunettes, mais doit lire
la région des yeux pour satisfaire quatre de ses six cibles. Au test, les lunettes occultent
exactement cette évidence. L'écart d'activation ID/OOD que tout le pipeline mesure ensuite est la
conséquence directe de ce choix d'étiquettes.

Notez le déséquilibre de `Young` (79.9 % de positifs) : une tête qui prédit toujours « oui »
obtient 0.80 sans rien apprendre. Ce détail explique l'anomalie de [§11.3](#113-deux-têtes-dattributs-instables-selon-la-graine).

### 2.4 Découpe supplémentaire pour la fusion supervisée

La régression logistique de [§7.2](#72-fusion-supervisée-régression-logistique) a besoin
d'étiquettes. `ood_cal_frac = 0.5`, `split_seed = 42` :

- les 13 193 images OOD sont coupées en deux → **6 596 pour la calibration**, **6 597 pour
  l'évaluation** ;
- les négatifs de calibration viennent de `train` (5 120 images de référence), **jamais** de
  `test_in`.

Deux garanties : les positifs de calibration sont **disjoints** des positifs d'évaluation, et
`test_in` reste intégralement vierge. C'est pourquoi les tableaux de
[§10.3](#103-signaux-de-fusion--ensemblefused_aurocjson) portent sur **18 941 ID vs 6 597 OOD**,
alors que ceux de [§10.1](#101-décomposition-pixel--ensemblesummaryjson) et
[§10.2](#102-score-localisé--ensemblelocalized_aurocjson) portent sur **18 941 ID vs 13 193 OOD**.
Les deux jeux de chiffres ne sont pas directement comparables.

---

## 3. Architectures

Les quatre schémas de [`docs/diagrams/`](../../../../docs/diagrams/) décrivent exactement cette
configuration (`.pdf` vectoriel + `.png`) :

| Schéma | Ce qu'il montre |
|---|---|
| `pipeline_classifier.pdf` | un membre, de l'image d'entrée aux deux têtes |
| `encoder_decoder.pdf` | le tronc déroulé, avec les cinq points de branchement des décodeurs |
| `pipeline_decoder.pdf` | les cinq `BlockDecoder`, une voie par bloc, avec leur coût |
| `ensemble_architecture.pdf` | les dix membres — un seul panneau d'architecture, dix graines |

### 3.1 Le classifieur

`FacialCNN` ([`lrad/model.py`](../../../../lrad/model.py)) — cinq blocs
`Conv(3×3, pad 1, bias=False) + BatchNorm2d + ReLU`, `MaxPool2×2` après chaque bloc **sauf le
dernier**, puis `AdaptiveAvgPool2d(1×1)`.

```text
3×128×128 → 32×64×64 → 64×32×32 → 128×16×16 → 256×8×8 → 256×8×8 → GAP → h ∈ R^256
```

Ni dropout, ni poids pré-entraînés, ni connexions résiduelles.

**Pourquoi le pooling s'arrête un bloc trop tôt.** Le dernier bloc conserve une carte 8×8, donc
le GAP moyenne sur une vraie étendue spatiale et non sur une seule cellule. Surtout, les blocs 4
et 5 **partagent leur résolution**, ce qui rend les deux décodeurs les plus profonds
architecturalement identiques et leurs reconstructions directement comparables.

Deux têtes linéaires lisent le vecteur de 256 dimensions :

- **`head_gender` (256→2)** — softmax + CE sur `Male`. Son rôle n'est pas l'exactitude en soi,
  mais de produire un vecteur de logits dont l'**énergie** et l'**entropie** bougent quand le
  classifieur hésite. Dans ce run, c'est le **meilleur signal isolé** (0.8112).
- **`head_attrs` (256→6)** — sigmoïde + BCE, pondérée `attr_loss_weight: 2.0`.

Perte totale : `CE(genre) + 2.0 · BCE(attributs)`, un seul `.backward()` par pas. **Aucune** tête
CutPaste dans ce bras.

**981 288 paramètres** par membre — 979 232 dans le tronc, 2 056 dans les deux têtes. Le tronc
pèse 99.8 % du modèle. (La variante avec tête CutPaste en compte 981 802 : 514 de plus.)

### 3.2 Les décodeurs

Après l'entraînement du classifieur, le tronc est **gelé** (`eval()`, `requires_grad = False`) et
un `BlockDecoder` ([`lrad/decoder.py`](../../../../lrad/decoder.py)) est entraîné par bloc pour
reconstruire l'image d'entrée depuis les seules activations de ce bloc.

Chaque décodeur empile `n_up = log₂(image_size / block_size)` étages ×2 apprenables —
`ConvTranspose2d(4×4, stride 2, pad 1) + BN + ReLU` — en divisant les canaux par deux à chaque
étage jusqu'à un plancher de 16, fermé par `Conv1×1 → 3` + `Sigmoid` pour retomber dans `[0,1]`.

Le noyau 4×4 / stride 2 / pad 1 est choisi parce qu'il double H et W **exactement**, sans
ambiguïté de taille de sortie — c'est ce qui garantit que le bloc *k* désigne la même échelle
spatiale chez tous les membres.

| Décodeur | Activation d'entrée | `n_up` | Chemin des canaux | Paramètres |
|---|---|---|---|---|
| `dec L0` | 32×64×64 | 1 | 32→16 | 8 275 |
| `dec L1` | 64×32×32 | 2 | 64→32→16 | 41 107 |
| `dec L2` | 128×16×16 | 3 | 128→64→32→16 | 172 307 |
| `dec L3` | 256×8×8 | 4 | 256→128→64→32→16 | 696 851 |
| `dec L4` | 256×8×8 | 4 | 256→128→64→32→16 | 696 851 |
| | | | **total** | **1 615 391** |

Les deux décodeurs les plus profonds portent **86 %** des paramètres : ils partent de l'activation
la plus large *et* demandent le plus d'étages de sur-échantillonnage. `dec L3` et `dec L4` sont
identiques — c'est la résolution partagée des blocs 4 et 5 qui réapparaît. La pile de décodeurs
pèse **1.65×** le classifieur qu'elle inverse.

Les cinq décodeurs sont optimisés conjointement par un seul Adam sur
$\sum_k \mathrm{MSE}\big(d_k(a_k(x)),\, x\big)$.

### 3.3 Les 10 membres

**C'est ici que ce run diffère de tous les autres du dépôt.** `ensemble.member_variants` est
absent de la config, donc `resolve_member_configs` construit les dix membres à partir de la même
section `model`. La seule diversité est la graine `base_seed + i` = 42…51, qui pilote
l'initialisation des poids et l'ordre de mélange SGD.

| # | Graine | Canaux | Noyau | Paramètres |
|---|---|---|---|---|
| 1…10 | 42…51 | 32-64-128-256-256 | 3×3 | 981 288 |

Il n'y a rien de plus à tabuler — et c'est le propos du bras. La trace spatiale
`128 → 64 → 32 → 16 → 8 → 8` est partagée **par identité** et non par la contrainte que
`resolve_member_configs` impose quand `member_variants` est actif : le bloc *k* désigne la même
échelle partout, les décompositions par bloc restent alignées, et moyenner les reconstructions
entre membres est trivialement bien défini.

L'argument standard pour aller plus loin — « des initialisations indépendantes produisent des
membres qui échouent sur les *mêmes* entrées, ce qui écrase le terme de variance » — est
exactement ce que les bras `arch` et `cutpaste` de l'ablation existent pour tester **contre ce
contrôle**. Ce que ce run mesure, lui, c'est le point de départ :

| Grandeur | Étendue sur les 10 membres |
|---|---|
| Paramètres | 981 288 — identiques par construction |
| Exactitude genre | 0.941 … 0.982 |
| AUROC de reconstruction (`p95`, par membre) | 0.6290 … 0.6399 (**étendue 0.011**) |
| MSE décodeur `L0` (dernière époque) | 0.00040 … 0.00048 |
| MSE décodeur `L4` (dernière époque) | 0.01110 … 0.01190 |

Dix membres qui reconstruisent à 4 % près les uns des autres : c'est le régime où le terme de
variance a le moins à dire. Il obtient pourtant 0.6510, **au-dessus** du biais (0.6246).

---

## 4. Entraînement

Séquence par membre, répétée 10 fois : classifieur → gel du tronc → décodeurs. Durée observée
≈ **47 min par membre** (≈ 17 min classifieur + ≈ 34 min décodeurs) sur L40S, soit ≈ 8 h pour
l'ensemble complet, évaluation comprise.

### 4.1 Étape A — le classifieur (20 époques, Adam 3e-4)

Perte `CE(genre) + 2.0 · BCE(attributs)`, batch 128, 1 332 pas par époque.

Membre 1 (graine 42), premières et dernières époques :

| Époque | `train_loss` | genre | attributs (6 têtes) |
|---|---|---|---|
| 1 | 0.8537 | 94.1 % | 77.5 / 89.0 / 87.7 / 81.2 / 86.4 / 83.9 |
| 20 | 0.3181 | 99.8 % | 90.4 / 95.3 / 93.3 / 90.4 / 96.5 / 95.1 |

La première époque coûte 328 s (préchauffage des workers et du cache disque), les suivantes ≈ 36 s.

Les dix membres convergent de façon quasi superposable — membre 2 finit à `train_loss = 0.3187`
contre 0.3181 pour le membre 1. **C'est attendu et c'est le point** : même architecture, même
données, même schedule ; seule la graine change. Les écarts qui subsistent à l'**évaluation**
(0.941 vs 0.982 d'exactitude genre) sont donc entièrement imputables à l'initialisation et à
l'ordre des batchs.

### 4.2 Étape B — les décodeurs (25 époques, Adam 1e-3, classifieur gelé)

MSE par bloc, moyennée sur les dix membres, dernière époque :

| Bloc | `L0` | `L1` | `L2` | `L3` | `L4` |
|---|---|---|---|---|---|
| MSE finale (moyenne des 10) | 0.00044 | 0.00130 | 0.00330 | 0.00538 | 0.01163 |
| MSE époque 1 (membre 1) | 0.00280 | 0.00455 | 0.00763 | 0.01057 | 0.02143 |

La qualité de reconstruction se dégrade **monotonement** avec la profondeur du branchement, d'un
facteur **26** entre `L0` et `L4`. C'est mécanique : les max-pools successifs détruisent le détail
spatial, donc un décodeur partant d'une carte 8×8 ne peut peindre qu'un visage flou.

**Ce plancher d'erreur est la raison pour laquelle tous les scores en espace pixel plafonnent vers
0.62–0.65.** Sur un visage propre, les cheveux et le fond contribuent déjà des milliers de pixels
à forte erreur, alors qu'une paire de lunettes en touche quelques centaines.

### 4.3 Résultats par membre

| # | Graine | Exactitude genre | AUROC `entropy_gender` | AUROC `entropy_attrs` | AUROC reconstruction |
|---|---|---|---|---|---|
| 1 | 42 | 0.9815 | 0.7908 | 0.5178 | 0.6290 |
| 2 | 43 | 0.9797 | 0.7220 | 0.5973 | 0.6323 |
| 3 | 44 | 0.9410 | 0.8120 | 0.4460 | 0.6399 |
| 4 | 45 | 0.9449 | 0.7931 | 0.5955 | 0.6352 |
| 5 | 46 | 0.9819 | 0.7633 | 0.5789 | 0.6351 |
| 6 | 47 | 0.9820 | 0.7782 | 0.5707 | 0.6323 |
| 7 | 48 | 0.9788 | 0.7856 | 0.5978 | 0.6347 |
| 8 | 49 | 0.9677 | 0.6649 | 0.4760 | 0.6382 |
| 9 | 50 | 0.9787 | 0.7805 | 0.5470 | 0.6337 |
| 10 | 51 | 0.9810 | 0.7632 | 0.4228 | 0.6378 |

**Aucun membre n'est défaillant.** Les dix entropies genre sont bien au-dessus du hasard (0.665 à
0.812) et les dix AUROC de reconstruction tiennent dans 1.1 point. C'est le contraste le plus net
avec le run archivé `LASTOF_RESULTS`, où deux membres tombaient sous 0.5 sur `entropy_gender`.

La colonne `entropy_attrs` est en revanche bruitée (0.423 à 0.598) — voir
[§11.3](#113-deux-têtes-dattributs-instables-selon-la-graine).

---

## 5. Le socle : Risque = Biais + Variance

C'est l'identité centrale du projet. Source : `lrad/ensemble.py:decomposition_maps`.

### 5.1 Notation

- $M = 10$ membres, indexés par $m$.
- $x \in [0,1]^{3 \times 128 \times 128}$ l'image d'entrée.
- $\hat f_k^m(x)$ la reconstruction du membre $m$ depuis le bloc $k$.
- $p$ un pixel (position spatiale), $c \in \{R, G, B\}$ un canal.

L'erreur par pixel est la **somme** sur les canaux RVB des carrés — pas la moyenne, pas de racine :

$$
e^m_k(x)[p] \;=\; \sum_{c \in \{R,G,B\}} \big( x_c[p] - \hat f^m_{k,c}(x)[p] \big)^2
\;\in\; [0, 3]
$$

Reconstruction consensus :

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

Le terme croisé s'annule **exactement**, par définition de la moyenne. Il reste Biais + Variance.
L'identité est vraie canal par canal, donc elle survit à la somme sur RVB. ∎

Le run vérifie numériquement cette identité :

```
Risk = Bias + Variance identity — max abs residual on 25 samples (10 ID + 15 OOD): 2.09e-07
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

- **Biais** = l'erreur qui **survit à l'ensemble**. Même en moyennant 10 modèles, on n'arrive pas
  à reconstruire ces pixels. C'est le candidat naturel pour « anomalie » : sur des lunettes, aucun
  décodeur entraîné sur des visages propres ne sait dessiner une monture.
- **Variance** = le **désaccord épistémique** en espace pixel.
- **Risque** = ce que coûte un modèle moyen pris isolément.

**Résultat observé.** Comme dans le run archivé, la hiérarchie attendue n'est **pas** celle
observée :

| Terme | AUROC agrégée |
|---|---|
| Variance | **0.6510** |
| Risque | 0.6339 |
| Biais | 0.6246 |

L'hypothèse de départ du projet — « le biais est le bon score d'anomalie » — n'est donc pas
confirmée ici non plus, et cette fois **elle ne peut pas être imputée à des membres défaillants**,
puisqu'il n'y en a aucun (§4.3). C'est un point important : le résultat est structurel, pas
accidentel. L'explication reste celle du commentaire de `lrad/ensemble.py` : pour une tâche
d'**occlusion**, tous les membres échouent *au même endroit*, donc l'erreur est grande mais
**corrélée** → biais élevé, variance faible en valeur absolue. Ce qui subsiste de désaccord est
néanmoins mieux *ordonné* que le biais, d'où l'AUROC légèrement supérieure.

Le désaccord qui croît réellement hors distribution vit dans les **prédictions**, pas dans les
pixels — et ce run le montre bien mieux que le précédent (§6.4).

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

**Exemple.** Sur 128×128 = 16 384 pixels, le p95 est le 819ᵉ plus grand. Une paire de lunettes
touche de l'ordre de 500 à 1 500 pixels : elle est donc *juste à la limite* d'influencer le p95.
C'est précisément pourquoi ce score plafonne.

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

Le plancher $\varepsilon$ est nécessaire : un pixel dont l'écart-type de référence s'effondre vers
0 transformerait sinon du bruit float en z-score non borné.

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
$\sigma[p] = 0.01$ : $z = (0.05 - 0.03)/0.01 = 2.0$. Ce même pixel dans les cheveux, où
$\mu[p] = 0.20$ et $\sigma[p] = 0.08$, donnerait $z = (0.05 - 0.20)/0.08 = -1.875$ : négatif, donc
ignoré. Le z-score fait exactement le travail attendu.

Supposons maintenant une fenêtre 4×4 dont les 16 z-scores valent tous 2.0 sauf un à 8.0 :
moyenne $= (15 \times 2 + 8)/16 = 2.375$. Un pixel chaud isolé sur fond de z=0 donnerait
$8/16 = 0.5$. La fenêtre privilégie bien les régions **cohérentes**, pas les pics isolés.

**Résultats du run** (`localized_auroc.json`, 18 941 ID vs 13 193 OOD) :

| Terme | z + patch-max, agrégé | Ligne de base p95 | Gain |
|---|---|---|---|
| Risque | **0.6988** | 0.6339 | +0.065 |
| Variance | **0.6939** | 0.6510 | +0.043 |
| Biais | **0.6850** | 0.6246 | +0.060 |

La normalisation par position vaut, à elle seule, entre 4 et 7 points d'AUROC — pour un coût nul
en entraînement, puisqu'elle ne fait que ré-évaluer des reconstructions déjà calculées.

### 6.3 `locfre` — erreur de features localisée

Source : `lrad/feature_error.py`.

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
est-elle plus forte ». Ici, où les dix membres ont **exactement la même largeur**, cette
normalisation est moins critique que dans le bras `arch` — mais elle reste utile contre les
différences de contraste entre images.

**Exemple chiffré.** Deux vecteurs de features unitaires séparés de 60° :
$\lVert u - v \rVert^2 = 2 - 2\cos 60° = 1$. Séparés de 90° : $2$. Diamétralement opposés : $4$
(le maximum).

**Résultats** (`fused_auroc.json`, blocs 1, 2, 3 — soit les blocs 2, 3 et 4 du tronc) :

| Signal | Résolution de la carte | AUROC | moyenne ID | moyenne OOD |
|---|---|---|---|---|
| `locfre_b1` | 32 × 32 | **0.7833** | 3.106 | 4.161 |
| `locfre_b2` | 16 × 16 | 0.7406 | 2.442 | 3.368 |
| `locfre_b3` | 8 × 8 | 0.7710 | 1.578 | 2.490 |

**Différence notable avec le run archivé** : là-bas, le bloc le plus profond (`locfre_b3`)
gagnait ; ici c'est le plus superficiel (`locfre_b1`), et l'écart entre les trois est plus large
(0.043 contre 0.018). Interprétation prudente : sans diversité d'architecture, les activations
profondes des dix membres sont presque redondantes, donc moyenner sur $m$ apporte moins au bloc
profond qu'au bloc superficiel, où le bruit d'initialisation reste visible.

### 6.4 Incertitude prédictive : Total = Aléatoire + Épistémique

Source : `lrad/ensemble.py:_uncertainty_scores`. Ici on quitte les pixels pour les **prédictions**.

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
- **Aléatoire** = l'ambiguïté intrinsèque de l'image. Tous les membres sont d'accord *pour être
  incertains*.
- **Épistémique** = le **désaccord** entre membres. Hors distribution, chaque membre extrapole
  différemment, donc ce terme devrait monter.

Par concavité de $H$ (inégalité de Jensen), $\text{Total} \geq \text{Aléatoire}$, donc
l'épistémique est **toujours $\geq 0$**.

**Exemple chiffré.** $M = 2$, tête genre. Membre 1 : $p_1 = (0.9,\, 0.1)$. Membre 2 :
$p_2 = (0.1,\, 0.9)$. Désaccord maximal.

- Moyenne : $\bar p = (0.5,\, 0.5)$
- Total $= H(0.5, 0.5) = \ln 2 = 0.693$
- $H(p_1) = -(0.9\ln 0.9 + 0.1\ln 0.1) = 0.325$, idem pour $H(p_2)$ → Aléatoire $= 0.325$
- Épistémique $= 0.693 - 0.325 = \mathbf{0.368}$ — élevé, comme attendu.

Si les deux membres avaient tous deux prédit $(0.9, 0.1)$ : Épistémique $= 0$.

**Résultats du run** :

| Tête | Total | Aléatoire | Épistémique |
|---|---|---|---|
| genre | 0.7864 | 0.7870 | **0.7832** ✓ |
| attributs | 0.6450 | 0.5675 | 0.6584 |
| combinée | 0.7421 | 0.7089 | **0.7336** ✓ |

**C'est le tableau qui change tout par rapport au run archivé.** Là-bas, l'épistémique genre était
à 0.4680 — *sous le hasard* — parce que deux membres défaillants polluaient l'estimation du
désaccord, ce qui tirait la version combinée à 0.5385. Ici, les dix membres étant sains, la tête
genre porte 0.7832 et la combinée 0.7336.

Or c'est la **combinée**, et elle seule, que `collect_fusion_signals` exporte sous le nom
`unc_epistemic_combined`. Dans le run archivé, la fusion recevait donc un signal quasi inutile
(0.539) ; ici elle reçoit un signal solide (0.734). C'est une des deux causes mécaniques de
l'écart entre les deux fusions par rang (0.804 → 0.842).

À noter : sur la tête genre, `Total`, `Aléatoire` et `Épistémique` sont presque égaux
(0.786 / 0.787 / 0.783). Les trois quantités classent les images quasi dans le même ordre — le
désaccord et l'incertitude intrinsèque sont ici fortement corrélés, ce qui limite ce que la
décomposition apporte au-delà de l'entropie brute.

### 6.5 Énergie (et pourquoi le MSP a disparu du code)

Source : `lrad/evaluate.py`, `lrad/fusion.py`.

**Énergie de la tête genre** — le meilleur signal isolé de ce run, pour un coût nul :

$$
E(x) = -\log \sum_{c} \exp\big(z_g(x)_c\big)
$$

Le code calcule cela via `-np.logaddexp(z_0, z_1)` moyenné sur les membres, forme numériquement
stable pour 2 classes.

Résultat du run : **AUROC 0.8112**, moyenne ID = −9.885, moyenne OOD = −5.325.

**Pourquoi ça marche.** Une entrée hors distribution produit des logits globalement plus **petits**
(le motif ne ressemble à aucune classe apprise), donc une énergie plus **grande** (moins négative).
Le softmax, lui, **normalise**, et détruit donc cette information d'amplitude.

**Exemple chiffré.**
- Entrée ID, logits $z = (6, -6)$ : $E = -\log(e^6 + e^{-6}) \approx -6.0$. MSP $\approx 10^{-5}$.
- Entrée OOD, logits $z = (2, -1)$ : $E = -\log(7.389 + 0.368) = -2.049$. MSP $= 0.047$.

L'énergie a bougé de **4 unités**, le MSP de 0.047 sur une échelle bornée à 1. C'est cette
dynamique qui fait la différence.

L'écart ID/OOD observé ici (−9.885 → −5.325, soit **4.56 unités**) est presque le double de celui
du run archivé (−7.320 → −5.422, soit 1.90). Un classifieur genre correctement entraîné sur les
dix membres produit des logits ID beaucoup plus saturés, donc une énergie ID beaucoup plus basse,
donc une meilleure séparation. C'est la seconde cause mécanique de l'écart entre les deux runs.

> **Note sur le MSP.** Le score $1 - \max_c p_g$ et l'entropie **combinée** ont été retirés du code
> (l'en-tête de `lrad/evaluate.py` documente la décision). Raison : sur deux classes,
> $1 - \max_c p_c$ est une fonction monotone de $H(p)$, donc le MSP est **rang-équivalent** à
> l'entropie genre — même AUROC, aucune information supplémentaire. Il ne reste que deux scores de
> tête par membre : `score_entropy_gender` et `score_entropy_attrs`.

### 6.6 Ce que ce bras n'a pas : la probabilité CutPaste

Le run archivé disposait d'un sixième signal de fusion,
$s_{\text{cp}}(x) = \frac{1}{M}\sum_m P_m(\text{altéré} \mid x)$, produit par la tête prétexte
CutPaste (AUROC isolée 0.6738 là-bas).

**Ce bras n'a pas cette tête** (`model.cutpaste_head` absent), donc la fusion travaille sur
**cinq** signaux au lieu de six. C'est l'une des deux différences qui rendent la comparaison
directe avec `LASTOF_RESULTS` non concluante — voir
[§11.6](#116-comparaison-avec-lastof_results-et-ce-quon-ne-peut-pas-en-conclure). Les bras
`cutpaste` et `arch_cutpaste` de l'ablation existent pour isoler cet effet contre ce contrôle.

### 6.7 Le biais péri-oculaire (classement des visages à lunettes)

Source : `lrad/ensemble.py:collect_eye_region_bias`. Sert uniquement à **classer** les visages OOD
pour la figure `top_ood_glasses.png` ; ce n'est pas un score OOD évalué en AUROC.

Fenêtre oculaire, en fractions de l'image : lignes 42–62 %, colonnes 15–85 %.

$$
\text{score} \;=\; \bar b_{\text{œil}} \times \frac{\bar b_{\text{œil}}}{\bar b_{\text{global}}}
\;=\; \frac{\bar b_{\text{œil}}^{\,2}}{\bar b_{\text{global}}}
$$

**Pourquoi cette forme.** Le premier facteur récompense un biais **fort** dans la région des yeux.
Le second est un facteur de **concentration** : un visage dont la reconstruction est mauvaise
partout a un ratio proche de 1 et ne reçoit aucun bonus. Le produit n'est grand que si le biais
est à la fois fort **et** localisé.

**Exemple, le visage classé n°1 du run** (`top_ood_glasses.json`, index CelebA 176 175) :

$$
\bar b_{\text{œil}} = 0.15272, \quad \bar b_{\text{global}} = 0.04045
\;\Longrightarrow\;
\text{score} = \frac{0.15272^2}{0.04045} = 0.5766
$$

Le biais est **3.78×** plus fort dans la région des yeux que sur l'ensemble du visage.

Le classement complet des dix :

| Rang | Index CelebA | Score | $\bar b_{\text{œil}}$ | $\bar b_{\text{global}}$ | Ratio |
|---|---|---|---|---|---|
| 1 | 176 175 | 0.5766 | 0.15272 | 0.04045 | 3.78 |
| 2 | 192 134 | 0.5186 | 0.12486 | 0.03006 | 4.15 |
| 3 | 49 617 | 0.5049 | 0.16619 | 0.05470 | 3.04 |
| 4 | 118 169 | 0.4646 | 0.14902 | 0.04780 | 3.12 |
| 5 | 138 619 | 0.4486 | 0.11878 | 0.03145 | 3.78 |
| 6 | 143 864 | 0.4328 | 0.23227 | 0.12466 | 1.86 |
| 7 | 1 556 | 0.4322 | 0.10669 | 0.02634 | 4.05 |
| 8 | 196 519 | 0.4214 | 0.13918 | 0.04597 | 3.03 |
| 9 | 49 225 | 0.4183 | 0.16584 | 0.06576 | 2.52 |
| 10 | 198 362 | 0.3952 | 0.09515 | 0.02291 | 4.15 |

Le 10ᵉ est encore à 0.395 — la queue décroît lentement, signe que le mécanisme n'est pas l'affaire
de deux ou trois images chanceuses. Le rang 6 est instructif : c'est le biais oculaire **le plus
fort** du classement (0.232) mais avec le ratio le plus **faible** (1.86) — un visage mal
reconstruit partout, que le facteur de concentration rétrograde correctement.

---

## 7. Fusion

Source : `lrad/fusion.py`. Aucun signal seul ne franchit 0.82. Mais ils se trompent sur des images
**différentes** — donc les combiner aide.

Les **cinq** signaux disponibles dans ce bras, produits en une seule passe par
`collect_fusion_signals` : `locfre_b1`, `locfre_b2`, `locfre_b3`, `unc_epistemic_combined`,
`ens_energy_gender`.

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
3. Elle est **immune aux échelles**. Comparez : `locfre_b1` vit vers 3.1,
   `ens_energy_gender` vers −9.9, `unc_epistemic_combined` vers 0.28. Une moyenne brute serait
   entièrement dominée par l'énergie. Après passage aux rangs, les cinq pèsent pareil.

**Exemple chiffré.** 4 images, 2 signaux.

| Image | $s^{(1)}$ | rang | $r$ | $s^{(2)}$ | rang | $r$ | fusion |
|---|---|---|---|---|---|---|---|
| A | 0.1 | 0 | 0.000 | 900 | 2 | 0.667 | 0.333 |
| B | 0.9 | 3 | 1.000 | 950 | 3 | 1.000 | **1.000** |
| C | 0.4 | 1 | 0.333 | 100 | 0 | 0.000 | 0.167 |
| D | 0.7 | 2 | 0.667 | 500 | 1 | 0.333 | 0.500 |

L'image B, en tête sur les deux signaux, remporte la fusion. L'image A, forte sur $s^{(2)}$ mais
dernière sur $s^{(1)}$, est ramenée au milieu. L'échelle absurdement différente des deux signaux
n'a aucun effet.

**Résultat : AUROC 0.8420** (moyenne ID 0.431, moyenne OOD 0.699), contre 0.8112 pour le meilleur
signal isolé. Le gain est de **+0.031**, obtenu **sans aucune étiquette OOD**. C'est le chiffre à
citer pour un détecteur réellement déployable.

**Pourquoi le gain est bien meilleur qu'au run archivé** (+0.031 ici contre +0.009 là-bas) : la
fusion par rang applique une moyenne uniforme, donc elle est plombée par son plus mauvais signal.
Le run archivé traînait `unc_epistemic_combined` à 0.5387 — presque du hasard — comptant pourtant
pour 1/6 du poids. Ici le plus mauvais des cinq est `locfre_b2` à 0.7406, et l'épistémique à 0.734
est devenue un contributeur utile. Un ensemble de signaux plus homogène rend la moyenne uniforme
beaucoup moins pénalisante.

### 7.2 Fusion supervisée (régression logistique)

$$
\tilde x_j = \frac{x_j - \mu_j}{\sigma_j},
\qquad
\text{score}(x) = \sum_{j=1}^{5} w_j\,\tilde x_j + b
$$

Le score est le **logit** de $P(\text{OOD} \mid x)$. Comme l'AUROC est invariante par
transformation monotone, travailler sur le logit plutôt que sur la probabilité ne change rien au
résultat.

Le protocole de découpe (§2.4) garantit que les positifs de calibration (6 596) sont **disjoints**
des positifs d'évaluation (6 597), et que les négatifs viennent de `train`, jamais de `test_in`.

**Poids appris** (`fused_auroc.json`, triés par $|w|$) :

| Signal | $\mu$ | $\sigma$ | $w$ | AUROC isolée |
|---|---|---|---|---|
| `ens_energy_gender` | −7.6224 | 4.0963 | **+1.296** | 0.8112 |
| `locfre_b1` | 3.6181 | 1.0813 | **+1.269** | 0.7833 |
| `locfre_b2` | 2.8865 | 1.1079 | **−0.521** | 0.7406 |
| `locfre_b3` | 1.9996 | 0.9630 | +0.368 | 0.7710 |
| `unc_epistemic_combined` | 0.2815 | 0.1584 | +0.351 | 0.7336 |

Intercept $b = -0.0370$. **Résultat : AUROC 0.8730** (moyenne ID = −1.415, moyenne OOD = +1.450).

**Trois choses à lire dans ce tableau.**

1. **`locfre_b2` reçoit un poids négatif (−0.521) alors que son AUROC isolée est 0.7406**, très
   au-dessus du hasard. Ce n'est pas une contradiction : la régression exploite l'information
   *conditionnelle*. Les trois `locfre` sont calculés sur la même reconstruction consensus et sont
   fortement corrélés entre eux ; une fois `locfre_b1` et `locfre_b3` dans le modèle, ce que
   `locfre_b2` ajoute est essentiellement du bruit **commun** aux deux autres, et le meilleur usage
   de ce résidu est de le **soustraire**. On retrouve exactement le rôle de `locfre_b2` dans le run
   archivé, où il recevait un poids quasi nul (+0.030) : il est le membre redondant du trio, quel
   que soit le run.

2. **`ens_energy_gender` et `locfre_b1` dominent à égalité** (+1.296 et +1.269), et ce sont aussi
   les deux meilleures AUROC isolées. La régression n'a pas eu à faire de correction spectaculaire
   ici, contrairement au run archivé — signe que les cinq signaux sont mieux conditionnés.

3. **`unc_epistemic_combined` reçoit un poids positif** (+0.351), là où le run archivé lui donnait
   **−0.742**. C'est la conséquence directe de §6.4 : quand la tête genre est saine, l'épistémique
   pointe dans le bon sens et la régression n'a plus besoin d'en inverser le signe.

**Exemple d'application.** Une image dont `locfre_b1` vaut 5.0 :
$\tilde x = (5.0 - 3.6181)/1.0813 = 1.278$, contribution $= 1.269 \times 1.278 = +1.622$ au logit.
On additionne les 5 contributions et l'intercept ; le résultat est le score final.

> ⚠️ La fusion supervisée voit des étiquettes OOD pendant la calibration. C'est une **borne
> supérieure optimiste**. Un détecteur déployé face à un type d'anomalie inconnu n'aurait pas ces
> étiquettes et retomberait sur la fusion par rang (0.8420).

L'écart rang → supervisé n'est ici que de **+0.031** (0.842 → 0.873), contre +0.060 au run
archivé. Interprétation : plus les signaux sont homogènes et correctement orientés, moins la
supervision a de travail correctif à faire. C'est une bonne nouvelle pour le déploiement — la
version *sans étiquettes* capture désormais 96 % de la performance de la version supervisée.

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
- **Sous 0.5 = le score est informatif mais dans le mauvais sens** — l'inverser donne
  $1 - \text{AUROC}$.

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
choisir un) et insensible au **déséquilibre des classes** (18 941 contre 13 193 ici, ou 6 597 pour
les scores fusionnés). Dans ce run, aucune AUROC ne tombe sous 0.5 — c'est la première fois, et
c'est le symptôme direct de l'absence de membre défaillant.

---

## 9. Interprétation de chaque figure

### 9.1 `ensemble/plots/` — les figures d'ensemble

Les figures en grille utilisent 25 échantillons tirés au hasard — **10 ID** (seed 1234) puis
**15 OOD** (seed 1235) —, étiquetés `ID 1…10` / `OOD 1…15` en marge de ligne.

---

**`ensemble_decomposition.png`** — *la figure fondamentale*

Pour chaque image (une ligne) et chaque bloc, trois tuiles : Risque | Biais | Variance.

*Comment lire.* Vérifiez visuellement l'identité : la tuile Risque doit être la somme pixel à pixel
des deux autres. Sur les lignes OOD, cherchez une zone chaude sur la monture des lunettes. Elle est
présente mais **noyée** dans l'erreur des cheveux et du fond — c'est le diagnostic visuel du
plafond à 0.62 des scores pixel.

*Maths.* Voir §5. Cartes calculées par `decomposition_maps`, échelle de couleur en $[0, 3]$.

---

**`decomposition_auroc.png`** — barres d'AUROC par bloc

Trois groupes de barres (Risque / Biais / Variance), une barre par bloc, plus la valeur agrégée.

| | bloc 1 | bloc 2 | bloc 3 | bloc 4 | bloc 5 |
|---|---|---|---|---|---|
| Risque | 0.5689 | 0.5865 | 0.6251 | **0.6482** | 0.6295 |
| Biais | 0.5494 | 0.5651 | 0.6030 | **0.6339** | 0.6256 |
| Variance | 0.5821 | 0.6034 | 0.6537 | **0.6780** | 0.6306 |

*Pourquoi ce pic au bloc 4.* Les blocs superficiels reconstruisent trop bien : ils copient la
texture, lunettes comprises, donc ne détectent rien. Les blocs profonds reconstruisent trop mal :
tout est flou, l'erreur est dominée par le flou et non par l'anomalie. Le bloc 4 est le compromis —
assez sémantique pour ne pas savoir dessiner des lunettes, assez résolu pour localiser où elles
sont. La forme de la courbe est **identique** à celle du run archivé, à 1 à 2 points près : c'est
une propriété de l'architecture, pas de la diversité de l'ensemble.

> Attention à la numérotation. Le code indexe les blocs de **0 à 4** et les figures les étiquettent
> `L0…L4` ; ce rapport les numérote de **1 à 5**. Le pic « bloc 4 » du tableau ci-dessus correspond
> donc à `L3` dans les figures. Les figures par instance et `score_comparison.png`, elles, sont
> produites au bloc `L4` — le cinquième et dernier.

---

**`architecture_effect.png`** — *ici, une figure de contrôle*

Quatre panneaux, une entrée par membre : (a) AUROC de reconstruction vs nombre de paramètres,
(b) vs largeur totale $\sum$ canaux, (c) regroupée par taille de noyau, (d) carte de chaleur
membre × bloc. Les points sont étiquetés `M1…M10` — attention, ces étiquettes sont **1-basées**
alors que les répertoires sont `model_0/`…`model_9/`.

*Ce que ce run montre.* **Les trois premiers panneaux sont dégénérés par construction** : les dix
membres ont le même nombre de paramètres (981 288), la même largeur totale (736) et le même noyau
(3×3). Les dix points se superposent donc en une seule colonne verticale, et il n'y a aucune
corrélation à mesurer.

C'est précisément ce qui rend la figure utile ici : **la dispersion verticale de cette colonne est
la variabilité pure due à la graine**, sans aucune contribution de l'architecture.

| Lecture | Valeur |
|---|---|
| Étendue de l'AUROC sur les 10 membres | 0.6290 → 0.6399 (**1.1 point**) |
| Écart-type inter-membres | 0.0031 |

À comparer avec le run archivé, où dix architectures couvrant un facteur 3 en paramètres
produisaient une étendue de 1.2 point. **Autrement dit : faire varier l'architecture sur cette
plage n'ajoute quasiment rien à la dispersion que la graine produit déjà toute seule.** C'est un
résultat négatif utile, et le bras `arch` de l'ablation est là pour le confirmer ou l'infirmer sur
les scores d'ensemble plutôt que sur les scores par membre.

Le panneau (d) confirme, membre par membre, le pic au bloc `L3` : la structure en profondeur est
la même pour les dix.

---

**`ensemble_score_hists.png`** — histogrammes des scores agrégés

Une paire d'histogrammes ID/OOD (densité normalisée) par terme.

*Comment lire.* Le **recouvrement** des deux distributions est le complément visuel de l'AUROC.
Ici le recouvrement est large (AUROC ≈ 0.62–0.65), donc deux cloches presque superposées, la
cloche OOD légèrement décalée à droite. Comparez avec le panneau droit de `fused_auroc.png`
(AUROC 0.873) pour voir à quoi ressemble une vraie séparation.

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
fonctions quantiles $F^{-1}_{\text{ID}}(q)$ et $F^{-1}_{\text{OOD}}(q)$. C'est un **Q–Q plot**
entre les deux distributions, déplié le long de $q$.

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

Cartes $(x - \bar f_k)^2$ uniquement. La vue la plus propre pour juger si les lunettes émergent,
sans être distrait par le risque et la variance. Échelle de couleur partagée entre toutes les
tuiles → les blocs sont directement comparables entre eux.

---

**`mean_error_maps.png`** — le terme de risque, par bloc

$\frac{1}{M}\sum_m (x - \hat f^m_k)^2$ : la moyenne des cartes d'erreur.

*Distinction cruciale à ne pas confondre avec la figure précédente.*

- `mean_error_maps` = **moyenne des erreurs** = Risque
- `mean_abs_bias` = **erreur de la moyenne** = Biais

L'écart entre les deux figures, pixel par pixel, **est** la variance. Les regarder côte à côte est
la façon la plus directe de voir où les modèles sont d'accord et où ils divergent. **Dans ce run,
les deux figures se ressemblent beaucoup** — c'est la signature visuelle de dix membres identiques,
et l'explication de la faible amplitude du terme de variance.

---

**`min_error_maps.png`** — l'erreur du meilleur membre

$\min_m (x - \hat f^m_k)^2$, pixel par pixel.

*Comment lire.* Répond à « à quel point le **meilleur** membre reconstruit-il ce pixel ». Une
région ne reste claire que si **aucun** modèle de l'ensemble n'y arrive — un signal OOD plus fort
que la moyenne. Note d'affichage : cette tuile a sa **propre** échelle de couleur, car le minimum
vit à une magnitude bien inférieure au biais ; une échelle partagée la rendrait uniformément noire.

---

**`variance_heatmaps_ood.png`** et **`variance_heatmaps_all.png`** — le désaccord

Cartes de variance, sur les OOD seuls puis sur ID + OOD.

*Comment lire.* Cherchez si le désaccord se concentre sur les lunettes. Le résultat du run dit
qu'il s'y concentre partiellement (variance agrégée : AUROC 0.6510, le meilleur des trois termes
pixel), mais le mécanisme reste faible : sur une **occlusion**, tous les membres échouent au même
endroit, donc l'erreur est corrélée. Avec dix membres architecturalement identiques, c'est le cas
extrême — le désaccord utile est ailleurs, dans les prédictions (§6.4), où il vaut 0.734.

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
sur les 13 193 images OOD, restreint à celles portant réellement l'attribut `Eyeglasses`. Le
classement complet est dans `top_ood_glasses.json` et reproduit en §6.7.

*Comment lire.* **C'est une figure de meilleurs cas, pas une évaluation.** Elle montre à quoi
ressemble le pipeline **quand il fonctionne** — facteur 3.78 entre biais oculaire et biais global
pour le n°1. Elle ne dit rien de la performance moyenne, qui est donnée par les AUROC.

*Note d'affichage.* La superposition applique `smooth_cam` : rectification à $\geq 0$,
normalisation par le pic, flou gaussien $\sigma = 1.5$, renormalisation, puis alpha
$= \text{cam}^{0.8}$. Ce sont des réglages **purement cosmétiques**, jamais présents dans le score.

---

**`fused_auroc.png`** — le panneau récapitulatif de la fusion

Produit par `scripts/run_fused.py` (pas par `run_ensemble.py`). Trois panneaux : barres d'AUROC par
signal (les fusions en couleur d'accent, ligne de hasard à 0.5) | courbes ROC de la meilleure
fusion et du meilleur signal isolé | histogrammes ID/OOD du score fusionné supervisé.

*Comment lire.* **C'est la figure à montrer en premier.** Le panneau de gauche donne la hiérarchie
complète des signaux d'un coup d'œil, de `locfre_b2` (0.741) au sommet (`fused_supervised`, 0.873).
Remarquez que, dans ce run, **les cinq barres de signaux sont de hauteurs proches** (0.74 à 0.81) —
c'est visuellement la raison pour laquelle la fusion par rang s'en sort si bien (§7.1).

---

### 9.2 `ensemble/plots/instances_in/` et `instances_ood/` — étude par visage

20 visages ID et 20 visages OOD, tirés au hasard (seeds 1334 et 1335), chacun dans son dossier
`ID_XX/` ou `OOD_XX/`, reconstruits au **bloc L4**.

| Fichier | Contenu |
|---|---|
| `model_01.png` … `model_10.png` | Pour **un** membre : Original \| Reconstruction \| Erreur $(x - \hat f^m)^2$. Échelle fixe $[0, 0.5]$ partagée → les 10 fichiers sont directement comparables. |
| `summary.png` | La vue consensus, 5 tuiles : Original \| Biais $(x-\bar f)^2$ \| Erreur moyenne \| Erreur min \| Superposition du biais lissé. |
| `all_models.png` | Superposition d'erreur de **chaque** membre côte à côte, puis les trois résumés (biais / moyenne / min). |

*Comment les utiliser.* `all_models.png` est l'outil de diagnostic : il rend le **désaccord**
directement visible. Dans ce run, les dix tuiles membres se ressemblent fortement — la variance est
faible et l'échec est systématique, ce qui est exactement le comportement attendu d'un ensemble à
architecture unique face à une occlusion. Ouvrez le même fichier dans un run du bras `arch` pour
voir la différence.

Les dossiers frères **`instances_in_raw/`** et **`instances_ood_raw/`** contiennent les mêmes
cartes en PNG nus — `original.png`, `bias.png`, `bias_overlay.png`, `mean_error.png`,
`min_error.png` — sans titres, axes ni barres de couleur. Ils sont **pixel-identiques** aux tuiles
de `summary.png` (mêmes dérivations via `_instance_summary_maps`), prêts à être placés dans un
article ou une présentation.

### 9.3 `model_i/plots/` — les figures par membre (10 × 11 fichiers)

| Fichier | Contenu et lecture |
|---|---|
| `training_history.png` | Perte et exactitude par époque. Sans split de validation, seules les courbes d'entraînement existent. Vérifiez que la perte descend sans plateau brutal. |
| `batch_accuracy.png` | Exactitude genre et attributs **par batch**, avec des marqueurs de fin d'époque. Beaucoup plus fin que la courbe par époque : révèle l'instabilité intra-époque. |
| `batch_loss.png` | Perte combinée **par batch** sur tout l'entraînement : trace brute en clair + moyenne mobile causale, échelle y logarithmique, frontières d'époque marquées. La vue la plus sensible pour repérer une divergence. |
| `decoder_history.png` | MSE de reconstruction **par bloc et par époque** (une ligne viridis par bloc, total en pointillés gris, échelle log, valeur finale annotée). Version graphique du tableau de §4.2 ; l'étagement des cinq courbes y est immédiat. |
| `per_block_breakdown.png` | Original \| Err Lk \| Recon Lk pour ce membre seul. À comparer avec `mean_recon_breakdown.png` de l'ensemble : un membre isolé reconstruit visiblement moins bien que le consensus. |
| `recons_only.png` | Les 5 reconstructions de ce membre, sans cartes d'erreur. |
| `activations.png` | Moyenne sur les canaux des activations de chaque bloc, superposée en `inferno` sur le visage. **Montre où le modèle met son énergie à chaque profondeur.** Chaque bloc a sa propre échelle. |
| `fusion_overlay.png` | Original \| Err L0…L4 \| Fusionné (max) \| Superposition. La carte fusionnée est le **max par pixel sur les blocs**, échelle fixe $[0, 0.5]$. |
| `roc_ood.png` | Les 2 courbes ROC du classifieur (entropie genre, entropie attributs) sur une même figure. |
| `score_dist_score_entropy_gender.png` | Histogramme ID vs OOD de l'entropie de la tête genre. **Dans ce run, les dix membres ont la cloche OOD décalée à droite** (AUROC 0.665 à 0.812) — aucun n'est inversé. |
| `score_dist_score_entropy_attrs.png` | Idem pour l'entropie des têtes attributs. Beaucoup plus bruité (0.423 à 0.598) — voir §11.3. |

Comme ce bras n'a pas de tête CutPaste, les figures et scores associés sont absents, et
`batch_cutpaste_acc` reste vide dans `history.json`.

---

## 10. Tableaux de résultats

### 10.1 Décomposition pixel — `ensemble/summary.json`

*18 941 ID vs 13 193 OOD, réduction p95, moyenne uniforme des blocs, 10 membres.*

| Terme | bloc 1 | bloc 2 | bloc 3 | bloc 4 | bloc 5 | **agrégé** |
|---|---|---|---|---|---|---|
| Risque | 0.5689 | 0.5865 | 0.6251 | 0.6482 | 0.6295 | **0.6339** |
| Biais | 0.5494 | 0.5651 | 0.6030 | 0.6339 | 0.6256 | **0.6246** |
| Variance | 0.5821 | 0.6034 | 0.6537 | 0.6780 | 0.6306 | **0.6510** |

### 10.2 Score localisé — `ensemble/localized_auroc.json`

*Mêmes splits complets. Fenêtres patch-max 4/8/16 px, référence : 5 120 images de `train`.*

| Terme | b1 | b2 | b3 | b4 | b5 | **z agrégé** | ligne de base p95 |
|---|---|---|---|---|---|---|---|
| Risque | 0.6200 | 0.6638 | 0.6784 | 0.7110 | 0.6925 | **0.6988** | 0.6339 |
| Biais | 0.5657 | 0.6480 | 0.6659 | 0.7046 | 0.6810 | **0.6850** | 0.6246 |
| Variance | 0.5761 | 0.6644 | 0.6793 | 0.6921 | 0.6897 | **0.6939** | 0.6510 |

Le meilleur bloc localisé est `b4` (= `L3`), cohérent avec le pic de §10.1.

### 10.3 Signaux de fusion — `ensemble/fused_auroc.json`

*18 941 ID vs 6 597 OOD (moitié d'évaluation). Blocs locfre 1, 2, 3.*

| Signal | AUROC | moyenne ID | moyenne OOD |
|---|---|---|---|
| **`fused_supervised`** | **0.8730** | −1.415 | +1.450 |
| **`fused_rank`** | **0.8420** | 0.431 | 0.699 |
| `ens_energy_gender` | 0.8112 | −9.885 | −5.325 |
| `locfre_b1` | 0.7833 | 3.106 | 4.161 |
| `locfre_b3` | 0.7710 | 1.578 | 2.490 |
| `locfre_b2` | 0.7406 | 2.442 | 3.368 |
| `unc_epistemic_combined` | 0.7336 | 0.2328 | 0.3547 |

### 10.4 Incertitude prédictive — `ensemble/summary.json`

| Tête | Total | Aléatoire | Épistémique |
|---|---|---|---|
| genre | 0.7864 | 0.7870 | **0.7832** |
| attributs | 0.6450 | 0.5675 | 0.6584 |
| combinée | 0.7421 | 0.7089 | 0.7336 |

---

## 11. Lecture critique et anomalies

### 11.1 Aucun membre ne s'effondre — et c'est un résultat

Le run archivé `LASTOF_RESULTS` avait **deux** membres dont l'AUROC `entropy_gender` tombait sous
0.5 (0.296 et 0.465), c'est-à-dire des modèles *plus confiants* sur les visages à lunettes que sur
les visages propres. Le run à 8 membres qui l'avait précédé en avait un.

**Ici, zéro.** Les dix `entropy_gender` vont de 0.665 à 0.812, les dix exactitudes genre de 0.941 à
0.982. Deux lectures possibles, et il faut être honnête sur le fait qu'on ne peut pas trancher
avec ce seul run :

1. **L'architecture de référence est simplement plus robuste** que certaines des variantes de
   `member_variants`. Les membres défaillants du run archivé étaient les membres 0 et 9 — dont
   les architectures étaient respectivement `[32,64,128,256,256] k=3` (le même qu'ici !) et
   `[32,48,112,224,336] k=5`. Cela affaiblit sérieusement cette explication.
2. **L'absence de CutPaste**. Dans le run archivé, la perte prétexte s'ajoutait à poids 0.5 et les
   pertes supervisées ne voyaient plus que les images intactes — ce qui réduit de fait le signal
   supervisé et peut déstabiliser une tête binaire. Le bras `cutpaste` de l'ablation, qui garde
   l'architecture unique et ajoute le prétexte, est exactement l'expérience qui départage.

C'est la question la plus intéressante que ce run laisse ouverte, et elle est directement testable.

### 11.2 L'hypothèse de départ n'est toujours pas confirmée

Variance (0.6510) > Risque (0.6339) > Biais (0.6246). Le biais, présenté comme *le* score
d'anomalie du projet, arrive dernier des trois.

La différence avec le run archivé, c'est qu'**ici l'explication par les membres défaillants ne
tient plus**. Le résultat est donc structurel : sur une tâche d'**occlusion**, tous les membres
échouent au même endroit ; le biais est grand mais son *ordonnancement* entre images est moins
informatif que celui du désaccord résiduel. Cela vaut aussi bloc par bloc — la variance domine le
biais sur les cinq blocs (§10.1).

Conclusion pratique : la décomposition reste utile comme outil de diagnostic et comme fondement
théorique, mais **le terme à exporter comme score n'est pas le biais**.

### 11.3 Deux têtes d'attributs instables selon la graine

Exactitudes par attribut, sur `test_in`, les dix membres :

| Attribut | min | max | étendue |
|---|---|---|---|
| `Arched_Eyebrows` | 0.727 | 0.828 | 0.101 |
| `Bushy_Eyebrows` | 0.615 | 0.912 | 0.297 |
| `Narrow_Eyes` | **0.364** | 0.896 | **0.532** |
| `Bags_Under_Eyes` | 0.635 | 0.834 | 0.199 |
| `Smiling` | 0.820 | 0.917 | 0.097 |
| `Young` | **0.369** | 0.879 | **0.510** |

Deux têtes décrochent complètement selon la graine : `Young` (membres 1, 3, 8, 10 sous 0.56, avec
un taux de positifs de 79.9 % dans `train` — une constante ferait mieux) et `Narrow_Eyes` sur le
membre 10 (0.364, contre 0.81–0.90 partout ailleurs). `Bushy_Eyebrows` décroche sur le membre 7.

**Cause probable.** La perte BCE est **non pondérée** et les six attributs sont fortement
déséquilibrés. Rien dans le schedule ne surveille ces têtes — pas de validation, pas d'arrêt
anticipé (§2.2) — donc une tête peut se caler sur une solution dégénérée pendant 20 époques sans
que quoi que ce soit ne le signale.

**Impact.** Aucun sur les scores principaux, qui lisent les logits **genre** et l'erreur de
reconstruction. Mais cela explique le bruit de `score_entropy_attrs` (0.423 à 0.598, §4.3) et le
fait que l'épistémique sur la tête attributs (0.6584) reste bien en dessous de l'épistémique genre
(0.7832) — alors que le design du dataset (4 attributs péri-oculaires) prédisait l'inverse.

**Correctif le moins cher** : pondérer la BCE par l'inverse de la fréquence des classes, ou
simplement rétablir un petit split de validation pour détecter les têtes dégénérées.

### 11.4 `unc_epistemic_combined` fonctionne enfin — mais pas pour la bonne raison

Le run archivé identifiait comme « amélioration la moins chère » le fait d'exposer
`unc_epistemic_attrs` à la fusion, parce que la version combinée y était plombée par la composante
genre (0.4680).

**Ce run inverse le diagnostic.** Ici c'est la composante **genre** qui est bonne (0.7832) et la
composante **attributs** qui traîne (0.6584) — conséquence directe de §11.3. La combinée à 0.7336
est donc tirée vers le bas par les attributs, exactement le symétrique de la situation précédente.

Deux conclusions :

- exporter `unc_epistemic_gender` **et** `unc_epistemic_attrs` séparément (au lieu de la seule
  combinée) reste la modification la plus rentable, mais pour une raison différente de celle
  identifiée en juillet — et cela vaut désormais dans les deux sens ;
- **la qualité de ce signal est un pur reflet de la santé des têtes du classifieur**, pas une
  propriété du mécanisme épistémique. C'est une fragilité à documenter : le même code produit
  0.539 sur un run et 0.734 sur un autre.

### 11.5 `locfre_b2` est structurellement redondant

Poids appris −0.521 malgré une AUROC isolée de 0.7406. Le run archivé lui donnait +0.030 pour une
AUROC isolée de 0.7773.

Dans les deux runs, `locfre_b2` est le membre du trio dont la contribution conditionnelle est
nulle ou négative. Les trois `locfre` partent de la **même** reconstruction consensus $\bar f_4$ et
ne diffèrent que par la profondeur du ré-encodage — leur corrélation est mécaniquement forte, et
le bloc intermédiaire est celui qui n'apporte rien que les deux extrêmes n'aient déjà.

**Piste concrète** : ne calculer que `locfre_b1` et `locfre_b3` (économie d'une passe de
ré-encodage complète), ou remplacer `b2` par un bloc réellement différent (`b0` ou `b4`) pour
vérifier si l'échelle intermédiaire est en cause ou si c'est bien la redondance.

### 11.6 Comparaison avec `LASTOF_RESULTS` — et ce qu'on ne peut pas en conclure

> Les fichiers de `LASTOF_RESULTS` (1 Go de poids et de figures) ne sont plus
> conservés dans le dépôt. Les chiffres cités ci-dessous sont ceux que ses
> `ensemble/*.json` rapportaient ; la recette reste reproductible depuis
> `configs/ablation_arch_cutpaste.yaml`.

| Score | `baseline` (ce run) | `LASTOF_RESULTS` | Δ |
|---|---|---|---|
| `fused_supervised` | **0.8730** | 0.8638 | +0.009 |
| `fused_rank` | **0.8420** | 0.8040 | **+0.038** |
| meilleur signal isolé | 0.8112 | 0.7952 | +0.016 |
| `unc_epistemic_combined` | 0.7336 | 0.5387 | **+0.195** |
| `ens_energy_gender` | 0.8112 | 0.7127 | **+0.099** |
| variance pixel p95 | 0.6510 | 0.6425 | +0.009 |
| biais pixel p95 | 0.6246 | 0.6228 | +0.002 |

Le bras de contrôle — dix modèles identiques, sans prétexte — **bat le run le plus élaboré du
dépôt sur tous ces chiffres**. C'est tentant à lire comme « la diversité d'architecture et
CutPaste ne servent à rien ». **Cette conclusion serait prématurée**, pour trois raisons :

1. **Deux facteurs changent à la fois** (diversité d'architecture *et* CutPaste), donc aucun des
   deux n'est isolé. C'est exactement ce que l'ablation corrige avec ses bras `arch` et
   `cutpaste`.
2. **Le jeu de signaux de fusion n'est pas le même** : six signaux là-bas (dont `cutpaste_prob` à
   0.674), cinq ici. Une moyenne uniforme de rangs sur des ensembles de signaux différents n'est
   pas comparable terme à terme.
3. **L'essentiel du Δ vient d'un seul endroit** : la santé de la tête genre. Les deux plus gros
   écarts du tableau (`unc_epistemic_combined` +0.195, `ens_energy_gender` +0.099) sont tous deux
   des signaux lus sur les logits genre, et les deux membres défaillants du run archivé les
   polluaient directement (§11.1). Les termes pixel, eux, sont quasi identiques entre les deux
   runs (+0.002 sur le biais).

Autrement dit : ce run est meilleur **surtout parce que ses dix classifieurs sont sains**, pas
nécessairement parce qu'il est plus simple. Départager les deux explications est précisément l'objet
de `scripts/compare_ablation.py` sur les quatre bras.

### 11.7 Réglages sous-exploités

- **`agg: p95` n'est jamais comparé** à `mean` ou `max` sur ce run. Le choix est raisonné (§6.1)
  mais pas mesuré.
- **Les fenêtres patch-max** (`{4, 8, 16}` px sur une image 128 px) n'ont jamais été balayées. Une
  paire de lunettes fait ≈ 40 × 15 px : la fenêtre 16 est la plus proche, mais 24 ou 32 n'ont pas
  été testées.
- **`ood_cal_frac = 0.5`** est généreux. La moitié des OOD sert à calibrer cinq poids — une
  fraction bien plus faible suffirait, et libérerait des positifs pour l'évaluation.
- **Le nombre de membres n'est pas balayé.** Avec dix membres identiques, la question « combien de
  graines faut-il réellement » est directement mesurable *a posteriori* en sous-échantillonnant
  les reconstructions déjà sauvegardées, sans réentraînement. C'est l'expérience la moins chère
  que ce run rend possible.

---

## 12. Reproduire ce run

```bash
# 1. Entraînement de l'ensemble + décomposition (≈ 8 h sur L40S)
python scripts/run_ensemble.py \
    --config configs/ablation_baseline.yaml \
    --output-dir outputs/celeba_ood/ablation/baseline_<horodatage>

# 2. Score localisé (z-score + patch-max), pas de réentraînement
python scripts/run_localized.py \
    --output-dir outputs/celeba_ood/ablation/baseline_<horodatage>

# 3. Fusion supervisée + panneau récapitulatif
python scripts/run_fused.py \
    --output-dir outputs/celeba_ood/ablation/baseline_<horodatage> \
    --blocks 1 2 3 --supervised

# 4. (optionnel) Régénérer les schémas d'architecture
python scripts/generate_arch_svg.py \
    --config configs/ablation_baseline.yaml --out-dir docs/diagrams
```

Sur le cluster, `scripts/oar_run_ablation.sh` soumet un job OAR par bras et enchaîne les étapes 1
à 3 dans chacun ; le dernier job terminé lance `scripts/compare_ablation.py`.

Ne relancez que les étapes 2 et 3 pour ré-évaluer un ensemble déjà entraîné ; ajoutez `--eval-only`
à l'étape 1 pour recalculer la décomposition et les figures depuis les poids sauvegardés.

### Carte des fichiers de sortie

```
baseline_20260819_133802_6866854/
├── Documentation.md              ce rapport
├── config.resolved.yaml          configuration effective, après résolution
├── logs/                         le log du job OAR 6866854
├── diagrams/                     schémas générés par le run (SVG + PNG)
├── model_0/ … model_9/           un dossier par membre (10)
│   ├── config.resolved.yaml      l'architecture de CE membre (identique aux 9 autres)
│   ├── weights/{model,decoders}.pt
│   ├── history.json              courbes d'entraînement du classifieur
│   ├── decoders_history.json     MSE par bloc et par époque
│   ├── summary.json              exactitude + AUROC de ce membre
│   └── plots/                    11 figures (voir §9.3)
└── ensemble/
    ├── summary.json              décomposition + incertitude + per-model, par bloc et agrégé
    ├── localized_auroc.json      scores z + patch-max
    ├── fused_auroc.json          signaux, calibration, AUROC fusionnées
    ├── top_ood_glasses.json      classement des 10 meilleurs cas
    ├── architectures.svg         schéma des 10 membres, tel qu'instancié
    └── plots/                    16 figures + 4 arborescences d'instances (voir §9.1, §9.2)
```

### Carte du code

| Module | Rôle |
|---|---|
| `lrad/dataset.py` | Découpe CelebA in/OOD, chargeurs, `split_loader` |
| `lrad/model.py` | `FacialCNN` — tronc 5 blocs + 2 têtes (3 avec CutPaste) |
| `lrad/cutpaste.py` | Augmentation CutPaste — **non utilisée par ce bras** |
| `lrad/decoder.py` | `BlockDecoder` — un décodeur par bloc |
| `lrad/train.py` | Les deux boucles d'entraînement (classifieur, décodeurs) |
| `lrad/ensemble.py` | Décomposition Risque/Biais/Variance, incertitude prédictive, biais péri-oculaire |
| `lrad/anomaly_score.py` | Réductions pixel → scalaire (mean / max / p95) |
| `lrad/localized.py` | z-score par pixel + patch-max multi-échelle |
| `lrad/feature_error.py` | Le signal `locfre` |
| `lrad/fusion.py` | Collecte des signaux, fusion par rang et fusion logistique supervisée |
| `lrad/evaluate.py` | Entropies par tête, énergie, AUROC |
| `lrad/plots.py` | Toutes les figures |
| `lrad/arch_diagram.py` | Rendu SVG des architectures (classifieur, décodeurs, ensemble) |
| `scripts/compare_ablation.py` | Fusion des quatre bras : tableaux + figures comparatives |
| `scripts/generate_arch_svg.py` | CLI des schémas de `docs/diagrams/` |
