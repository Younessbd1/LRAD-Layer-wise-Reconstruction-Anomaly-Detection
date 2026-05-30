# LRAD — Détection d'Anomalies par Reconstruction par Couches

> Détection de visages hors-distribution (lunettes) en mesurant **la qualité
> de reconstruction d'un ensemble profond, couche par couche** — et décomposition
> de cette erreur en une partie *biais* (l'anomalie) et une partie *variance*
> (désaccord entre les modèles).

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.4-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Présentation

Un CNN entraîné de zéro sur des visages CelebA **sans lunettes**
(`Eyeglasses == 0`). Par bloc convolutif, un petit décodeur apprend à reconstruire
l'image d'entrée à partir des activations de ce bloc. Plusieurs modèles sont
entraînés indépendamment — un **ensemble profond** — et pour toute image on obtient
une reconstruction par modèle.

L'erreur de reconstruction pixel par pixel, bloc par bloc, se décompose exactement
en deux termes :

```text
Risque  =  Biais  +  Variance
```

* **Biais** = erreur de la reconstruction *moyenne de l'ensemble* `(x − f̄)²`. C'est
  l'erreur irréductible du modèle consensus — ce qui subsiste après l'ensemblage.
  C'est le **score d'anomalie** : un visage portant des lunettes ne peut pas être
  reconstruit à partir de caractéristiques apprises uniquement sur des visages sans
  lunettes, donc son biais explose.
* **Variance** = désaccord entre les modèles entraînés indépendamment
  `mean_m (f̂ᵐ − f̄)²`. Sur des entrées hors-distribution, les modèles extrapolent
  différemment — la variance est donc aussi un signal d'**incertitude épistémique**.

Aucun blanchiment σ, aucune division : l'anomalie est le terme biais brut
`biais = risque − variance = (x − f̄)²`.

Les images CelebA avec `Eyeglasses == 1` (lunettes de soleil incluses) ne sont
jamais vues à l'entraînement et servent uniquement d'ensemble OOD à l'évaluation.

---

## Méthode

### 1. Protocole de distribution

| Partition    | Sélection            | Rôle                                          |
|--------------|----------------------|-----------------------------------------------|
| train / test_in | `Eyeglasses == 0` | visages propres ; entraînement + in-dist tenu |
| test_ood     | `Eyeglasses == 1`    | OOD à l'éval (jamais vu à l'entraînement)     |

Le pool propre est découpé par ratio (défaut `train_ratio=0.90`,
`val_ratio=0.00`). Avec `val_ratio=0`, **il n'y a ni boucle de validation ni
arrêt précoce** — chaque modèle suit intégralement le planning d'époques, ce qui
correspond exactement à ce que veut un ensemble profond (la diversité vient
uniquement de l'initialisation aléatoire + l'ordre de mélange SGD, jamais d'un
point d'arrêt partagé sur la validation).

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

Le nombre de blocs est fixé par `model.channels` — le modèle par défaut est
5 blocs ; la config livrée en utilise 6 (`[32, 64, 128, 256, 256, 256]`).
Les cibles in-distribution sont :

| Tête    | Attribut(s)                                                                  | Perte               |
|---------|------------------------------------------------------------------------------|---------------------|
| gender  | Homme / Femme                                                                | `CrossEntropyLoss`  |
| attrs   | Young, Smiling, Mouth_Slightly_Open, High_Cheekbones, Pointy_Nose, Oval_Face | `BCEWithLogitsLoss` |

Objectif combiné, un seul backward :

```text
loss = CE(gender_logits, gender) + attr_loss_weight · BCE(attr_logits, attrs)
```

Les attributs accessoires (Wearing_Hat, Heavy_Makeup, …) ne sont délibérément
**pas** des cibles — le tronc doit apprendre les traits d'identité/expression,
pas des caractéristiques d'accessoires qui généraliseraient partiellement aux
lunettes de soleil.

> **Pas de Dropout.** La diversité de l'ensemble est censée provenir uniquement
> de l'initialisation aléatoire indépendante + l'ordre SGD (un ensemble profond),
> pas du MC-Dropout.

### 3. Décodeurs par bloc

Le classifieur étant gelé, un `BlockDecoder` par bloc convolutif est entraîné
(MSE) pour suréchantillonner les activations du bloc vers une reconstruction
`(3, 64, 64)` via une pile `ConvTranspose2d → BN → ReLU` terminée par une conv
`1×1` + `Sigmoid`. Ces reconstructions `f̂` alimentent la décomposition
biais/variance et les visualisations par bloc.

### 4. Décomposition biais/variance de l'ensemble profond

Pour le bloc `k`, l'image `x`, le pixel `i` (valeur par pixel = moyenne RGB),
avec les `M` reconstructions `f̂ᵐ` et leur moyenne `f̄ = (1/M) Σ f̂ᵐ` :

```text
Risque_k(x)[i]   = (1/M) Σ_m ( x[i] − f̂ᵐ(x)[i] )²       erreur par modèle
Biais_k(x)[i]    =          ( x[i] − f̄(x)[i] )²          erreur de la recon. moyenne
Variance_k(x)[i] = (1/M) Σ_m ( f̂ᵐ(x)[i] − f̄(x)[i] )²   désaccord entre modèles
```

Ces quantités vérifient `Risque = Biais + Variance` exactement, pixel par pixel
(vérifié à l'exécution — le résidu max sur le run livré est `1.4e-7`).

Les cartes par pixel sont réduites à un scalaire par image avec `agg` ∈
`{mean, max, p95}` (défaut `p95` — robuste aux pixels isolés mais sensible à une
occlusion localisée comme les lunettes), puis combinées entre blocs (uniforme ou
pondéré) et scorées par AUROC sur `test_in` (label 0) vs `test_ood` (label 1).

### 5. Baseline confiance du classifieur

À titre de référence, le runner mono-modèle rapporte aussi les scores OOD
classiques issus des sorties de têtes :

* `score_msp` — `1 − max softmax(gender)`
* `score_entropy_gender` — entropie du softmax gender
* `score_entropy_attrs` — entropie de Bernoulli moyenne sur les 6 têtes d'attributs
* `score_entropy_combined` — entropie `gender + attrs`

---

## Résultats

Ensemble de 10 modèles (seeds 42–51), tronc 6 blocs, `agg = p95`, sur la
partition OOD `Eyeglasses`.

| Métrique                                      | Valeur         |
|-----------------------------------------------|----------------|
| Précision gender in-distribution              | ~97.2 – 97.6 % |
| **AUROC anomalie (Biais) — agrégé**           | **0.593**      |
| AUROC anomalie (Biais) — meilleur bloc        | 0.643          |
| AUROC Risque / Variance — agrégé              | 0.591 / 0.575  |
| Résidu max `Risque = Biais + Variance`        | 1.4e-7         |

La détection OOD sur cette tâche est genuinement difficile — les lunettes
occultent une région petite et localisée — d'où des AUROCs modestement au-dessus
du hasard ; les baselines de confiance du classifieur sont plus faibles et
instables selon les seeds (MSP/entropie AUROC ~0.48–0.70). L'intérêt du projet
est la **décomposition** : isoler le terme biais qui porte réellement le signal OOD
du terme variance qui reflète le désaccord de l'ensemble.

*(Les figures peuvent être régénérées avec `run_ensemble.py` — les graphiques sont
écrits dans `outputs/`, qui est gitignored.)*

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
│   ├── ensemble.py             # décomposition biais/variance de l'ensemble profond
│   ├── plots.py                # toutes les figures
│   ├── utils.py                # device, seed, logging
│   └── __init__.py
├── configs/
│   └── celeba_ood.yaml         # config unique — utilisée par les deux runners
├── scripts/
│   ├── run_celeba.py           # orchestrateur mono-modèle
│   ├── run_ensemble.py         # orchestrateur ensemble profond + décomposition
│   ├── oar_run_celeba.sh       # wrapper Grid'5000 / OAR (mono-modèle)
│   └── oar_run_ensemble.sh     # wrapper Grid'5000 / OAR (ensemble)
├── tests/                      # pytest : score anomalie, ensemble, entraînement sans val
├── docs/                       # notes et explications supplémentaires
├── requirements.txt            # dépendances épinglées (CUDA 11.8) pour Grid'5000
├── setup.py
└── README.md
```

---

## Installation

```bash
# Installation éditable avec dépendances souples (CPU ou torch déjà installé)
pip install -e .

# Ou la pile CUDA 11.8 épinglée utilisée sur Grid'5000 (GPU Pascal/A100)
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

Python ≥ 3.10. Dépendances principales : torch, torchvision, numpy (<2.0),
scikit-learn, matplotlib, pyyaml, Pillow.

## Démarrage rapide

```bash
# Ensemble profond + décomposition biais/variance (le pipeline principal).
# size / base_seed / agg viennent du bloc ensemble: de la config.
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

`configs/celeba_ood.yaml` est la source unique pour les deux runners :

| Section      | Clés principales                                                                            |
|--------------|---------------------------------------------------------------------------------------------|
| `experiment` | `name`, `seed`, `output_dir`                                                                |
| `dataset`    | `root`, `download`, `image_size`, `batch_size`, `train_ratio`, `val_ratio`                  |
| `model`      | `channels` (un entier par bloc conv), `n_attrs`, `n_gender`                                 |
| `training`   | `epochs`, `lr`, `weight_decay`, `attr_loss_weight` ; imbriqué `decoders: {epochs, lr, …}`   |
| `evaluation` | `n_viz_in_samples`, `n_viz_ood_samples`, `viz_seed`                                         |
| `ensemble`   | `size`, `base_seed` (le modèle *i* a le seed `base_seed + i`), `agg`                        |

Mettre `training.decoders` à null pour sauter l'entraînement des décodeurs
(la décomposition de l'ensemble n'a alors rien à reconstruire — à conserver pour
`run_ensemble`).

## Sorties

Tout sous `outputs/` est **gitignored** (les runs sont volumineux — poids des
modèles et graphiques haute résolution). Un run mono-modèle écrit dans
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
récupérer l'archive une fois sur le frontend et la décompresser sous `data/celeba/` :

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
