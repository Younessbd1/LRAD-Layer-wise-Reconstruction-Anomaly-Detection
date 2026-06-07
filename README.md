# LRAD — Détection d'Anomalies par Reconstruction par Couches

> On essaie de détecter des visages avec lunettes en mesurant **à quel point
> un ensemble de modèles arrive à les reconstruire, couche par couche** — et en
> décomposant cette erreur en une partie *biais* (l'anomalie) et une partie
> *variance* (le désaccord entre les modèles).

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.4-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## C'est quoi l'idée

J'ai entraîné un CNN from scratch sur des visages CelebA **sans lunettes**
(`Eyeglasses == 0`). Pour chaque bloc convolutif, un petit décodeur apprend à
reconstruire l'image d'entrée à partir des activations de ce bloc. J'ai
entraîné plusieurs modèles indépendamment — un **ensemble profond** — et pour
chaque image on obtient une reconstruction par modèle.

Ce qui est intéressant, c'est que l'erreur de reconstruction pixel par pixel
se décompose exactement en deux termes :

```text
Risque  =  Biais  +  Variance
```

* **Biais** = erreur de la reconstruction *moyenne de l'ensemble* `(x − f̄)²`.
  C'est l'erreur irréductible — ce qui reste même après avoir agrégé tous les
  modèles. J'utilise ça comme **score d'anomalie** : un visage avec lunettes ne
  peut pas être bien reconstruit à partir de features apprises uniquement sur
  des visages sans lunettes, donc son biais explose.
* **Variance** = désaccord entre les modèles `mean_m (f̂ᵐ − f̄)²`. Sur des
  entrées hors-distribution, les modèles extrapolent différemment — c'est donc
  aussi un signal d'**incertitude épistémique**, mais moins robuste que le biais
  (voir les docs).

Pas de blanchiment, pas de normalisation : le score d'anomalie c'est simplement
`biais = risque − variance = (x − f̄)²`.

Les images CelebA avec `Eyeglasses == 1` ne sont jamais vues à l'entraînement —
elles servent uniquement à évaluer la détection OOD.

---

## Comment ça marche

### 1. Découpage des données

| Partition    | Sélection            | Rôle                                          |
|--------------|----------------------|-----------------------------------------------|
| train / test_in | `Eyeglasses == 0` | visages propres ; entraînement + in-dist      |
| test_ood     | `Eyeglasses == 1`    | OOD à l'éval (jamais vu à l'entraînement)     |

On découpe le pool propre par ratio (défaut `train_ratio=0.90`,
`val_ratio=0.00`). Avec `val_ratio=0`, **il n'y a ni boucle de validation ni
arrêt précoce** — chaque modèle suit intégralement le planning d'époques. C'est
voulu : la diversité d'un ensemble profond doit venir uniquement de
l'initialisation aléatoire et de l'ordre SGD, pas d'un point d'arrêt différent
selon la validation.

### 2. Le classifieur (`FacialCNN`)

Un tronc conv multi-tête, sans poids pré-entraînés. Chaque bloc est
`Conv3×3 → BN → ReLU (→ MaxPool2×2)` ; le dernier bloc omet le pool. Un
average pooling global alimente deux têtes linéaires :

```text
Entrée (B, 3, 64, 64)
  Bloc1  Conv(3→32)   + BN + ReLU + MaxPool   → (32, 32, 32)
  Bloc2  Conv(32→64)  + BN + ReLU + MaxPool   → (64, 16, 16)
  Bloc3  Conv(64→128) + BN + ReLU + MaxPool   → (128, 8, 8)
  Bloc4  Conv(128→256)+ BN + ReLU + MaxPool   → (256, 4, 4)
  Bloc5  Conv(256→256)+ BN + ReLU + MaxPool   → (256, 2, 2)
  Bloc6  Conv(256→256)+ BN + ReLU             → (256, 2, 2)
  AdaptiveAvgPool → (256,)
    head_gender : Linear(256 → 2)   # Homme / Femme        — softmax + CE
    head_attrs  : Linear(256 → 6)   # 6 attributs binaires — sigmoid + BCE
```

Le nombre de blocs est fixé par `model.channels` — on utilise 6 blocs
(`[32, 64, 128, 256, 256, 256]`). Les cibles in-distribution sont :

| Tête    | Attribut(s)                                                                  | Perte               |
|---------|------------------------------------------------------------------------------|---------------------|
| gender  | Homme / Femme                                                                | `CrossEntropyLoss`  |
| attrs   | Young, Smiling, Mouth_Slightly_Open, High_Cheekbones, Pointy_Nose, Oval_Face | `BCEWithLogitsLoss` |

Loss combinée, un seul backward :

```text
loss = CE(gender_logits, gender) + attr_loss_weight · BCE(attr_logits, attrs)
```

J'ai délibérément **exclu** les attributs accessoires (Wearing_Hat, Heavy_Makeup…)
des cibles — le tronc doit apprendre des traits d'identité/expression, pas des
caractéristiques qui généraliseraient partiellement aux lunettes.

> **Pas de Dropout.** La diversité de l'ensemble vient uniquement de
> l'initialisation aléatoire + l'ordre SGD — pas du MC-Dropout.

### 3. Décodeurs par bloc

Le classifieur étant gelé, on entraîne un `BlockDecoder` par bloc convolutif
(MSE) pour remonter les activations vers une reconstruction `(3, 64, 64)` via
une pile `ConvTranspose2d → BN → ReLU` terminée par `Conv 1×1 + Sigmoid`. Ces
reconstructions `f̂` alimentent la décomposition biais/variance et les
visualisations.

### 4. Décomposition biais/variance

Pour le bloc `k`, l'image `x`, le pixel `i`, avec les `M` reconstructions `f̂ᵐ`
et leur moyenne `f̄ = (1/M) Σ f̂ᵐ` :

```text
Risque_k(x)[i]   = (1/M) Σ_m ( x[i] − f̂ᵐ(x)[i] )²
Biais_k(x)[i]    =          ( x[i] − f̄(x)[i] )²
Variance_k(x)[i] = (1/M) Σ_m ( f̂ᵐ(x)[i] − f̄(x)[i] )²
```

`Risque = Biais + Variance` exactement, pixel par pixel — on le vérifie à
l'exécution (résidu max sur le run livré : `1.4e-7`).

Les cartes par pixel sont réduites à un scalaire avec `agg` ∈
`{mean, max, p95}` (défaut `p95` — robuste aux pixels isolés mais sensible à
une occlusion localisée comme des lunettes), puis combinées entre blocs et
scorées par AUROC.

### 5. Baseline confiance du classifieur

Pour comparer, on calcule aussi les scores OOD classiques depuis les têtes :

* `score_msp` — `1 − max softmax(gender)`
* `score_entropy_gender` — entropie du softmax gender
* `score_entropy_attrs` — entropie de Bernoulli moyenne sur les 6 attributs
* `score_entropy_combined` — entropie `gender + attrs`

---

## Résultats

Ensemble de 10 modèles (seeds 42–51), tronc 6 blocs, `agg = p95`, partition
OOD `Eyeglasses`.

| Métrique                                      | Valeur         |
|-----------------------------------------------|----------------|
| Précision gender in-distribution              | ~97.2 – 97.6 % |
| **AUROC anomalie (Biais) — agrégé**           | **0.593**      |
| AUROC anomalie (Biais) — meilleur bloc        | 0.643          |
| AUROC Risque / Variance — agrégé              | 0.591 / 0.575  |
| Résidu max `Risque = Biais + Variance`        | 1.4e-7         |

La tâche est genuinement difficile — les lunettes occultent une région petite
et localisée. Les AUROCs sont modestement au-dessus du hasard, et les baselines
de confiance (MSP/entropie, ~0.48–0.70 selon les seeds) sont plus instables.
Ce qui m'intéressait surtout, c'est la **décomposition** elle-même : isoler le
biais, qui porte vraiment le signal OOD, de la variance, qui reflète juste le
désaccord entre modèles.

*(Les figures peuvent être régénérées avec `run_ensemble.py` — les graphiques
vont dans `outputs/`, qui est gitignored.)*

---

## Structure du projet

```text
lrad/
├── lrad/                       # bibliothèque principale (flat layout)
│   ├── dataset.py              # chargeurs CelebA ; filtre Eyeglasses pour l'OOD
│   ├── model.py                # FacialCNN : tronc conv + têtes gender/attrs
│   ├── decoder.py              # décodeurs de reconstruction par bloc
│   ├── train.py                # entraînement classifieur + décodeurs
│   ├── evaluate.py             # précision + AUROC OOD confiance classifieur
│   ├── anomaly_score.py        # erreur par pixel + réductions pixel→scalaire
│   ├── ensemble.py             # décomposition biais/variance de l'ensemble
│   ├── plots.py                # toutes les figures
│   ├── utils.py                # device, seed, logging
│   └── __init__.py
├── configs/
│   └── celeba_ood.yaml         # config unique — utilisée par les deux runners
├── scripts/
│   ├── run_celeba.py           # orchestrateur mono-modèle (importé par run_ensemble.py)
│   ├── run_ensemble.py         # orchestrateur ensemble + décomposition
│   └── oar_run_ensemble.sh     # wrapper Grid'5000 / OAR (ensemble — train + plots, un seul run)
├── tests/                      # pytest : score anomalie, ensemble, entraînement sans val
├── docs/                       # notes et explications
├── requirements.txt            # dépendances épinglées (CUDA 11.8) pour Grid'5000
├── setup.py
└── README.md
```

---

## Installation

```bash
# Installation éditable avec dépendances souples (CPU ou torch déjà installé)
pip install -e .

# Ou la pile CUDA 11.8 épinglée utilisée sur Grid'5000
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

Python ≥ 3.10. Dépendances principales : torch, torchvision, numpy (<2.0),
scikit-learn, matplotlib, pyyaml, Pillow.

## Démarrage rapide

```bash
# Ensemble + décomposition biais/variance (le pipeline principal)
python scripts/run_ensemble.py --config configs/celeba_ood.yaml

# Taille / répertoire de sortie personnalisés
python scripts/run_ensemble.py --config configs/celeba_ood.yaml \
    --output-dir outputs/celeba_ood/ensemble_run --ensemble-size 5

# Relancer uniquement la décomposition sur des modèles déjà entraînés
python scripts/run_ensemble.py --config configs/celeba_ood.yaml \
    --output-dir outputs/celeba_ood/ensemble_run --eval-only
```

Mono-modèle (classifieur + décodeurs + scores confiance/fusion) :

```bash
python scripts/run_celeba.py --config configs/celeba_ood.yaml
python scripts/run_celeba.py --config configs/celeba_ood.yaml --eval-only
python scripts/run_celeba.py --config configs/celeba_ood.yaml \
    --override training.epochs=50 dataset.batch_size=128
```

Les deux runners acceptent `--override clé=valeur …` (chemins pointés, ex.
`ensemble.size=3`) et `--no-plots`.

## Configuration

`configs/celeba_ood.yaml` est la source unique de vérité pour les deux runners :

| Section      | Clés principales                                                                            |
|--------------|---------------------------------------------------------------------------------------------|
| `experiment` | `name`, `seed`, `output_dir`                                                                |
| `dataset`    | `root`, `download`, `image_size`, `batch_size`, `train_ratio`, `val_ratio`                  |
| `model`      | `channels` (un entier par bloc conv), `n_attrs`, `n_gender`                                 |
| `training`   | `epochs`, `lr`, `weight_decay`, `attr_loss_weight` ; imbriqué `decoders: {epochs, lr, …}`   |
| `evaluation` | `n_viz_in_samples`, `n_viz_ood_samples`, `viz_seed`                                         |
| `ensemble`   | `size`, `base_seed` (le modèle *i* a le seed `base_seed + i`), `agg`                        |

Mettre `training.decoders` à null pour sauter l'entraînement des décodeurs.

## Sorties

Tout sous `outputs/` est **gitignored**. Un run mono-modèle écrit dans
`experiment.output_dir` :

```text
weights/model.pt, weights/decoders.pt
history.json, decoders_history.json, summary.json, config.resolved.yaml
plots/  training_history · batch_accuracy · score_dist_* · roc_ood
        per_block_breakdown · recons_only · activations
        fusion_overlay · fusion_auroc
logs/
```

Un run ensemble écrit un jeu de résultats complet par modèle plus la décomposition :

```text
model_<i>/            résultats mono-modèle complets, un par membre de l'ensemble
ensemble/summary.json AUROCs par modèle + décomposition, résidu d'identité,
                      anomaly_auroc principal (terme Biais)
ensemble/plots/       ensemble_decomposition · decomposition_auroc
                      ensemble_score_hists · mean_recon_breakdown
                      mean_recons_only · mean_abs_bias
                      mean_error_maps · min_error_maps
                      variance_heatmaps_{ood,all}
                      bias_variance_vs_{block,percentile}
logs/ · config.resolved.yaml
```

## Grid'5000 / OAR

```bash
./scripts/oar_run_ensemble.sh '2026-05-28 20:00:00'   # réservation anticipée
./scripts/oar_run_ensemble.sh                          # soumission immédiate
oarstat -u $USER
tail -f outputs/celeba_ood/_oar/oar.<jobid>.stdout
```

Les wrappers ciblent le cluster `gratouille`, réservent un GPU (un ensemble de
10 modèles sur le tronc 6 blocs tourne ~11–13 h) et créent un dossier de sortie
sous `outputs/celeba_ood/`.

## Téléchargement de CelebA

Si le téléchargement Google Drive de torchvision échoue sur un nœud de calcul,
il faut récupérer l'archive une fois sur le frontend et la décompresser sous
`data/celeba/` :

```text
data/celeba/img_align_celeba/        (202 599 fichiers .jpg)
data/celeba/list_attr_celeba.txt
data/celeba/list_eval_partition.txt
data/celeba/identity_CelebA.txt
data/celeba/list_bbox_celeba.txt
data/celeba/list_landmarks_align_celeba.txt
```

## Tests

```bash
pytest
```

Couvre les réductions du score d'anomalie, la décomposition de l'ensemble
(y compris l'identité `Risque = Biais + Variance`), et l'entraînement avec
`val_ratio=0`.

---

## Licence

MIT — voir [LICENSE](LICENSE).
