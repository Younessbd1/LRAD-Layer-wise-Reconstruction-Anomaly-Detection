# Anchored Ensembling et AUROC Variance

## Contexte

Dans le run `celeba_ood/ensemble_20260620_065006_6601493`, j'utilise un ensemble de 8 modèles entraînés avec `anchor_lambda = 0.9`. Ce document explique la méthode d'anchor et ce que signifie l'"AUROC variance" que je calcule sur les blocs du CNN.

---

## Méthode d'Anchor (Pearce et al., 2020)

### Principe

À l'initialisation de chaque membre de l'ensemble, je prends une photo des poids aléatoires initiaux et je la garde comme **ancre** θ₀. Pendant l'entraînement du décodeur, j'ajoute une pénalité à la loss de reconstruction :

```
L_total = L_reconstruction + 0.5 * λ * Σⱼ (θⱼ − θ₀ⱼ)²
```

Avec `λ = 0.9` dans ce run. Le gradient de cette pénalité est `λ * (θ − θ₀)` — il tire chaque paramètre vers son point de départ aléatoire, comme un weight decay ordinaire mais **vers l'ancre plutôt que vers zéro**.

### Pourquoi les ancres diffèrent entre membres

Chaque membre de l'ensemble est initialisé avec une seed différente (42 → 49). Donc les θ₀ sont distincts d'un membre à l'autre, même si l'architecture est identique. C'est là que se crée la diversité.

### Comportement in-distribution vs OOD

**In-distribution** : le terme de reconstruction MSE est fort. Il domine et force tous les décodeurs à bien reconstruire l'image → les membres convergent vers des solutions proches → faible désaccord entre eux.

**OOD** : le terme MSE devient faible (l'image est hors du manifold appris). La force principale qui agit sur chaque décodeur est alors l'attraction vers **son propre θ₀**, qui est différent pour chaque membre → les membres divergent → fort désaccord entre eux.

### Exemple concret

Prenons un poids `w` dans le décodeur du bloc 3, initialisé à :
- membre 0 : θ₀ = +0.3
- membre 5 : θ₀ = −0.2

Sur une image CelebA (in-distribution), le MSE est fort → `w` converge vers ~0.05 pour les deux membres.

Sur une image OOD, le MSE est presque nul → seule l'ancre agit :
- le membre 0 est tiré vers +0.3
- le membre 5 est tiré vers −0.2

Les deux décodeurs reconstruisent alors des images **systématiquement différentes** → la variance de reconstruction est élevée → signal OOD.

Sans anchor (`anchor_lambda = 0`), les deux membres glisseraient vers le même minimum local sur données OOD → pas de signal de désaccord entre eux.

---

## Qu'est-ce que l'"AUROC Variance" ?

C'est deux concepts combinés.

### 1. La Variance (score par image)

Pour chaque image `x` et chaque bloc `k`, mes 8 décodeurs produisent 8 reconstructions. La variance est la dispersion entre elles :

```
Var_k(x) = moyenne_i [ ||reconstruction_i(x) − reconstruction_moyenne(x)||² ]
```

C'est un scalaire par image : **élevé si les membres sont en désaccord, faible s'ils s'accordent**.

### 2. L'AUROC

J'ai deux populations d'images :
- images **in-distribution** (CelebA test) → je veux que `Var_k` soit faible
- images **OOD** → je veux que `Var_k` soit élevée

L'AUROC mesure la capacité de ce score à séparer les deux populations : quelle est la probabilité qu'une image OOD tirée au hasard ait une variance **plus élevée** qu'une image in-distribution tirée au hasard ?

- AUROC = 1.0 → séparation parfaite
- AUROC = 0.5 → aléatoire, le score ne distingue rien
- AUROC < 0.5 → le score est inversé (OOD a moins de variance qu'in-dist)

---

## Résultats sur ce run

### Variance par bloc (AUROC)

| Bloc | AUROC Variance | Commentaire |
|------|---------------|-------------|
| 0    | 0.497         | Quasi-aléatoire — les features bas-niveau ne divergent pas |
| 1    | 0.547         | Meilleur signal de désaccord |
| 2    | 0.513         | Signal faible |
| 3    | 0.530         | Signal modeste |
| 4    | 0.537         | Signal notable |
| 5    | 0.533         | Signal notable |

**Variance agrégée** : AUROC = **0.535**

### Comparaison avec l'entropie épistémique (tête gender)

| Méthode | AUROC |
|---------|-------|
| Variance par bloc (agrégée) | 0.535 |
| Entropie épistémique — gender | **0.717** |

La variance inter-membres des décodeurs reste un signal OOD faible dans ce run. L'entropie épistémique calculée directement sur la tête de classification gender est nettement plus discriminante. Cela suggère que la diversité générée par l'anchor se manifeste davantage dans les prédictions finales que dans les reconstructions pixel-level.
