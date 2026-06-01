# Décomposition Biais / Variance d'un Deep Ensemble

## Table des matières

1. [Contexte et motivation](#1-contexte-et-motivation)
2. [Les trois termes : définitions et formules](#2-les-trois-termes--définitions-et-formules)
3. [Démonstration : Risk = Biais + Variance](#3-démonstration--risk--biais--variance)
4. [Interprétations](#4-interprétations)
5. [Exemple complet avec image RGB — 3 plans](#5-exemple-complet-avec-image-rgb--3-plans)
6. [Pourquoi le Biais est le score d'anomalie](#6-pourquoi-le-biais-est-le-score-danomaalie)
7. [Limite de la Variance comme signal OOD](#7-limite-de-la-variance-comme-signal-ood)

---

## 1. Contexte et motivation

On a entraîné **M modèles indépendants** (deep ensemble) — même architecture,
mêmes données, initialisations aléatoires différentes. Chaque modèle `m`
produit, pour chaque bloc convolutif `k`, une reconstruction `f̂ᵐₖ(x)` de
l'image d'entrée `x`.

Le but : détecter des images **hors-distribution** (OOD) — des visages avec
lunettes, alors qu'on n'a entraîné que sur des visages sans lunettes.

Dès le départ, la question qui s'est posée : **quel signal utiliser pour dire
qu'une image est OOD ?**

- L'erreur d'un seul modèle est trop bruitée pour être fiable.
- La moyenne de M modèles est plus stable, mais comment l'interpréter ?
- Comment décomposer cette erreur de façon à distinguer ce qui vient vraiment
  de l'anomalie vs ce qui vient juste du désaccord entre les modèles ?

C'est là que la **décomposition biais/variance** entre en jeu — elle sépare
exactement ces deux sources d'erreur.

---

## 2. Les trois termes : définitions et formules

Pour un bloc `k`, une image `x` et un pixel `i` (valeur scalaire = moyenne RGB),
avec les M reconstructions `f̂ᵐ` et leur moyenne d'ensemble :

```
f̄(x)[i] = (1/M) Σₘ f̂ᵐ(x)[i]
```

### Risk

```
Risk_k(x)[i] = (1/M) Σₘ ( x[i] − f̂ᵐ(x)[i] )²
```

Erreur quadratique **moyenne sur les M modèles** pour ce pixel. C'est le coût
attendu si on tire un modèle au hasard dans l'ensemble.

### Biais

```
Biais_k(x)[i] = ( x[i] − f̄(x)[i] )²
```

Erreur quadratique du **modèle consensuel** — ce modèle fictif dont les
prédictions seraient exactement la moyenne de l'ensemble. Il mesure l'erreur
**irréductible** : celle qui persiste même après avoir agrégé tous les modèles.

### Variance

```
Variance_k(x)[i] = (1/M) Σₘ ( f̂ᵐ(x)[i] − f̄(x)[i] )²
```

Désaccord moyen des modèles autour de leur propre moyenne. C'est l'**incertitude
épistémique** — ce sur quoi les modèles ne s'accordent pas.

### Identité exacte

Ces trois termes vérifient l'identité algébrique suivante, **pixel par pixel** :

```
Risk = Biais + Variance
```

---

## 3. Démonstration : Risk = Biais + Variance

### Notations

Pour alléger, on pose pour un pixel `i` fixé :

```
a    = x[i]               (valeur réelle)
bₘ   = f̂ᵐ(x)[i]          (reconstruction du modèle m)
b̄    = (1/M) Σₘ bₘ        (moyenne de l'ensemble = f̄(x)[i])
```

### Développement

On part du Risk et on décompose chaque erreur en deux parties :

```
a − bₘ = (a − b̄) + (b̄ − bₘ)
          ───────   ─────────
           erreur    écart du
           du        modèle m
           consensus à la moyenne
```

On développe le carré :

```
Risk = (1/M) Σₘ (a − bₘ)²
     = (1/M) Σₘ [ (a − b̄) + (b̄ − bₘ) ]²
     = (1/M) Σₘ [ (a − b̄)²  +  2(a − b̄)(b̄ − bₘ)  +  (b̄ − bₘ)² ]
```

On distribue la somme sur les trois termes :

```
     = (a − b̄)²
     + 2(a − b̄) · (1/M) Σₘ (b̄ − bₘ)
     + (1/M) Σₘ (b̄ − bₘ)²
```

### Annulation du terme croisé

Le terme du milieu contient :

```
(1/M) Σₘ (b̄ − bₘ) = b̄ − (1/M) Σₘ bₘ = b̄ − b̄ = 0
```

La moyenne des écarts à la moyenne est toujours nulle — le terme croisé
disparaît.

### Résultat

```
Risk = (a − b̄)²  +  (1/M) Σₘ (b̄ − bₘ)²
       ─────────      ──────────────────────
         Biais                Variance

⟹  Risk = Biais + Variance     ∎
```

C'est une identité **exacte**, pas une approximation. Elle tient pixel par
pixel, à l'erreur flottante près.

---

## 4. Interprétations

### Risk — "Quel modèle si je tire au sort ?"

Le Risk mesure l'erreur attendue d'un modèle **aléatoirement sélectionné**
dans l'ensemble — ce qu'on paierait en déployant un seul modèle choisi au hasard.

- **Élevé sur ID** si les modèles sont mauvais individuellement mais se
  compensent en moyenne.
- **Élevé sur OOD** dans tous les cas.
- **Limite** : il mélange deux signaux (erreur systématique + désaccord), ce
  qui le rend moins diagnostique.

### Biais — "Même en s'accordant, l'ensemble se trompe-t-il ?"

Le Biais mesure l'erreur du **modèle consensuel** `f̄`. La question sous-jacente :
*même si on prend le meilleur estimateur possible (la moyenne), est-ce que la
reconstruction est correcte ?*

À noter : "biais" ici ne désigne pas un offset constant architectural. C'est
une mesure **par image** :

- Sur un visage normal (ID) : `f̄` reconstruit bien → Biais ≈ 0.
- Sur un visage avec lunettes (OOD) : `f̄` ne sait pas reconstruire les lunettes
  → Biais >> 0.

C'est l'erreur **irréductible** — ce qu'aucune diversification de modèles ne
peut corriger, parce que le problème est systématique pour cette entrée.

| Angle | Lecture |
|---|---|
| Statistique | Erreur de l'estimateur optimal (la moyenne) |
| Reconstruction | Ce que l'ensemble ne peut pas reconstruire, même en consensus |
| OOD | Signature d'une entrée hors de la distribution d'entraînement |
| Décision | Si Biais est élevé → l'image est probablement OOD |

### Variance — "Les modèles sont-ils d'accord entre eux ?"

La Variance mesure le **désaccord entre modèles** — leur dispersion autour de
leur propre moyenne. C'est le signal d'**incertitude épistémique** : les modèles
extrapolent différemment sur des entrées inconnues.

- Sur ID : les modèles ont appris les mêmes patterns → faible désaccord → Variance ≈ 0.
- Sur OOD : les modèles extrapolent différemment → désaccord → Variance plus élevée.

| Angle | Lecture |
|---|---|
| Statistique | Dispersion des prédictions individuelles autour de la moyenne |
| Incertitude | Signal d'incertitude épistémique |
| Localisation | La carte de variance montre *où* les modèles hésitent |
| Diagnostic | Utile pour comprendre les régions incertaines, pas pour la décision finale |

---

## 5. Exemple complet avec image RGB — 3 plans

### Configuration

- **M = 3 modèles**, **1 pixel** à la position `(h, w)` d'un bloc `k`.
- L'image est en RGB : la valeur scalaire d'un pixel est la **moyenne sur les
  3 canaux**.

### Valeur réelle du pixel

```
x[i] = (R = 0.9,  G = 0.6,  B = 0.3)
```

### Reconstructions des 3 modèles

| Modèle | R    | G    | B    |
|--------|------|------|------|
| f̂¹    | 0.70 | 0.50 | 0.20 |
| f̂²    | 0.80 | 0.60 | 0.30 |
| f̂³    | 0.90 | 0.70 | 0.40 |

### Étape 1 — Moyenne de l'ensemble f̄ (par canal)

```
f̄_R = (0.70 + 0.80 + 0.90) / 3 = 2.40 / 3 = 0.800
f̄_G = (0.50 + 0.60 + 0.70) / 3 = 1.80 / 3 = 0.600
f̄_B = (0.20 + 0.30 + 0.40) / 3 = 0.90 / 3 = 0.300
```

### Étape 2 — Calcul du Risk

Erreur quadratique de chaque modèle, moyennée sur les 3 canaux RGB :

```
se₁ = (1/3) × [(0.9 − 0.70)² + (0.6 − 0.50)² + (0.3 − 0.20)²]
    = (1/3) × [  0.0400      +    0.0100      +    0.0100     ]
    = (1/3) × 0.0600
    = 0.0200

se₂ = (1/3) × [(0.9 − 0.80)² + (0.6 − 0.60)² + (0.3 − 0.30)²]
    = (1/3) × [  0.0100      +    0.0000      +    0.0000     ]
    = (1/3) × 0.0100
    = 0.003333

se₃ = (1/3) × [(0.9 − 0.90)² + (0.6 − 0.70)² + (0.3 − 0.40)²]
    = (1/3) × [  0.0000      +    0.0100      +    0.0100     ]
    = (1/3) × 0.0200
    = 0.006667
```

```
Risk = (1/3) × [se₁ + se₂ + se₃]
     = (1/3) × [0.0200 + 0.003333 + 0.006667]
     = (1/3) × 0.0300
     = 0.0100
```

### Étape 3 — Calcul du Biais

Erreur quadratique du modèle consensuel, moyennée sur les 3 canaux :

```
Biais = (1/3) × [(0.9 − 0.800)² + (0.6 − 0.600)² + (0.3 − 0.300)²]
      = (1/3) × [  (0.100)²    +      0          +       0        ]
      = (1/3) × 0.0100
      = 0.003333
```

Seul le canal **Rouge** contribue au biais — les canaux Vert et Bleu sont
parfaitement reconstruits en moyenne.

### Étape 4 — Calcul de la Variance

Désaccord de chaque modèle par rapport à f̄, moyenné sur les 3 canaux :

```
var₁ = (1/3) × [(0.70 − 0.800)² + (0.50 − 0.600)² + (0.20 − 0.300)²]
     = (1/3) × [    0.0100     +     0.0100      +     0.0100     ]
     = (1/3) × 0.0300
     = 0.0100

var₂ = (1/3) × [(0.80 − 0.800)² + (0.60 − 0.600)² + (0.30 − 0.300)²]
     = (1/3) × [    0.0000     +      0.0000     +      0.0000    ]
     = 0.0000

var₃ = (1/3) × [(0.90 − 0.800)² + (0.70 − 0.600)² + (0.40 − 0.300)²]
     = (1/3) × [    0.0100     +     0.0100      +     0.0100     ]
     = (1/3) × 0.0300
     = 0.0100
```

```
Variance = (1/3) × [var₁ + var₂ + var₃]
         = (1/3) × [0.0100 + 0.0000 + 0.0100]
         = (1/3) × 0.0200
         = 0.006667
```

### Étape 5 — Vérification de l'identité

```
Biais + Variance = 0.003333 + 0.006667 = 0.010000 = Risk  ✓
```

### Récapitulatif

| Terme    | Valeur   | Interprétation pour ce pixel                          |
|----------|----------|-------------------------------------------------------|
| Risk     | 0.0100   | Erreur moyenne si on prend un modèle au hasard        |
| Biais    | 0.003333 | Erreur du consensus — canal R légèrement sous-estimé  |
| Variance | 0.006667 | Désaccord entre modèles — modèles 1 et 3 divergent    |

La **variance domine** ici (les modèles divergent entre 0.70 et 0.90 sur le
canal R). Le **biais est faible** (la moyenne 0.80 n'est qu'à 0.10 de 0.90).
Ce pixel serait classé comme peu anomal.

---

## 6. Pourquoi le Biais est le score d'anomalie

### Ce que chaque terme capture sur ID vs OOD

| Situation | Biais | Variance | Diagnostic |
|---|---|---|---|
| ID normal | ≈ 0 | ≈ 0 | Image normale |
| OOD, modèles divergent | **Élevé** | Élevé | OOD détecté par les deux |
| OOD, modèles se trompent pareil | **Élevé** | ≈ 0 | Seul le biais détecte |

### Pourquoi pas Risk ?

```
Risk = Biais + Variance
```

Le Risk mélange les deux signaux. Sur des données ID, des modèles
individuellement imparfaits mais dont la moyenne est bonne donnent un Risk
élevé mais un Biais faible — le Risk est donc un signal plus bruité.

### Pourquoi pas Variance ?

La variance est nulle quand tous les modèles se trompent de la même façon. Si
les M modèles ont appris la même représentation incorrecte d'un pattern OOD,
ils vont tous produire la même mauvaise reconstruction — la variance reste basse,
l'anomalie est manquée.

### Pourquoi le Biais ?

Le Biais mesure si le modèle consensuel `f̄` échoue à reconstruire l'entrée.
Même si tous les modèles se trompent de façon identique, `f̄` sera incorrect,
et `(x − f̄)²` sera grand.

```
Cas extrême : tous les f̂ᵐ sont identiques
→ f̄ = f̂ᵐ pour tout m
→ Variance = 0
→ Risk = Biais
→ le Biais capture encore l'anomalie
```

Le Biais est donc le signal le plus **robuste et direct** — il dit simplement
*"l'ensemble ne sait pas reconstruire cette image"*.

---

## 7. Limite de la Variance comme signal OOD

### Le problème d'effondrement d'ensemble

Quand les M modèles ont la même architecture, les mêmes données et une loss
similaire, ils peuvent apprendre les mêmes biais inductifs. Face à un input OOD,
ils extrapolent tous de la même façon — le désaccord reste faible :

```
f̂¹(lunettes) ≈ f̂²(lunettes) ≈ f̂³(lunettes)   (tous reconstruisent un visage sans lunettes)
→ Variance ≈ 0
→ mais (x − f̄)² >> 0   (les lunettes ne sont toujours pas reconstruites)
→ Biais détecte, Variance rate
```

C'est précisément ce qu'on a observé empiriquement — les AUROC confirment que
Biais > Risk > Variance.

### Quand la Variance reste utile malgré tout

- **Visualisation** : la carte de variance localise *où* les modèles hésitent,
  même si elle ne sert pas à la décision finale.
- **Diagnostic** : une variance élevée sur une région ID peut signaler une zone
  structurellement difficile (occultation, flou).
- **Validation** : comparer les AUROC des trois termes prouve empiriquement
  le bon choix du signal — sans ça, choisir le Biais resterait arbitraire.

### Résumé

```
Variance → "les modèles sont incertains ici"  (signal d'incertitude)
Biais    → "même le consensus se trompe ici"  (signal d'erreur)

Pour la détection OOD : Biais >> Variance en robustesse
Pour l'interprétabilité : les deux sont complémentaires
```
