# Clarifications — Décomposition Biais / Variance

---

## 1. Différence entre Risk et erreur de reconstruction

L'**erreur de reconstruction** d'un seul modèle `m` pour un pixel `i` est :

```text
err_m(x)[i] = ( x[i] − f̂ᵐ(x)[i] )²
```

C'est le nombre qu'on obtient en lançant un seul modèle sur une image. Chaque
modèle donne une valeur différente — cette erreur est donc aléatoire selon
lequel des M modèles on choisit.

Le **Risk** est la moyenne de ces erreurs sur tous les M modèles :

```text
Risk(x)[i] = (1/M) Σₘ ( x[i] − f̂ᵐ(x)[i] )²
```

Le Risk, c'est la moyenne des erreurs individuelles — ce qu'on espère obtenir si
on déploie un seul modèle tiré au hasard dans l'ensemble.

Analogie concrète : si 3 archers tirent chacun une fois sur une cible, l'erreur
de reconstruction c'est la distance d'un archer particulier ; le Risk, c'est
la distance moyenne des 3 tirs.

---

## 2. Pourquoi le Risk est le « coût attendu si on tire un modèle au hasard »

Tirer un modèle au hasard dans un ensemble de M modèles, c'est tirer `m`
uniformément dans `{1, 2, …, M}` avec probabilité `1/M` chacun. L'espérance
du coût est :

```text
E_m[ (x[i] − f̂ᵐ(x)[i])² ]
  = Σₘ P(m) · (x[i] − f̂ᵐ(x)[i])²
  = Σₘ (1/M) · (x[i] − f̂ᵐ(x)[i])²
  = (1/M) Σₘ ( x[i] − f̂ᵐ(x)[i] )²
  = Risk(x)[i]
```

C'est exactement la formule du Risk. Pas d'approximation : Risk = espérance
empirique de l'erreur au carré sur l'ensemble.

### Exemple chiffré

Supposons M = 3 modèles sur un pixel dont la vraie valeur est `x[i] = 0.8` :

| Modèle | Prédiction | Erreur au carré |
|--------|-----------|-----------------|
| m = 1  | 0.60      | (0.8 − 0.60)² = 0.0400 |
| m = 2  | 0.80      | (0.8 − 0.80)² = 0.0000 |
| m = 3  | 1.00      | (0.8 − 1.00)² = 0.0400 |

```text
Risk = (0.0400 + 0.0000 + 0.0400) / 3 = 0.0267
```

Si on déploie un seul modèle tiré au hasard, son erreur attendue est `0.0267`.
Le modèle 2 ferait `0.0000`, les modèles 1 et 3 feraient `0.0400` — le Risk
résume tout ça en une seule valeur.

---

## 3. Le modèle consensuel et le modèle fictif

### Le modèle consensuel = modèle fictif = f̄

```text
f̄(x)[i] = (1/M) Σₘ f̂ᵐ(x)[i]
```

Ce « modèle » n'existe pas en mémoire comme réseau de neurones entraîné. C'est
une **abstraction** : si on pouvait créer un seul réseau dont les prédictions
seraient exactement la moyenne des M réseaux, ce serait le modèle consensuel.
On l'appelle « fictif » parce qu'il n'est jamais entraîné — il émerge
mécaniquement comme la moyenne des sorties.

### Exemple

Même pixel `x[i] = 0.8`, M = 3 modèles, prédictions : 0.60, 0.80, 1.00.

```text
f̄(x)[i] = (0.60 + 0.80 + 1.00) / 3 = 0.800
```

Le modèle fictif prédit exactement `0.800`. Le Biais de ce pixel est :

```text
Biais = (0.8 − 0.800)² = 0.0000
```

Les modèles individuels se trompaient (erreurs de ±0.2), mais leur moyenne est
parfaite. → Biais nul, la diversité des modèles a compensé leurs erreurs.

### Contre-exemple : cas OOD (lunettes)

Sur un pixel qui appartient à une monture de lunettes, les 3 modèles
reconstruisent tous un morceau de peau — ils ne savent pas reconstruire les
lunettes. Prédictions : 0.75, 0.78, 0.72 ; vraie valeur : `x[i] = 0.20`
(lunette sombre).

```text
f̄(x)[i] = (0.75 + 0.78 + 0.72) / 3 = 0.750
Biais    = (0.20 − 0.750)² = 0.3025
```

Le modèle consensuel se trompe massivement. Ce n'est pas de la diversité qui
manque — c'est que l'ensemble entier ne connaît tout simplement pas les lunettes.

---

## 4. Biais = erreur irréductible : mais est-ce vraiment nul sur ID ?

### Ce que dit la formule

```text
Biais(x)[i] = ( x[i] − f̄(x)[i] )²
```

Cette erreur est « irréductible » dans le sens suivant : peu importe combien de
modèles on ajoute à l'ensemble, la moyenne `f̄` tend vers `E[f̂(x)[i]]` —
l'espérance sur les initialisations aléatoires. Si cette espérance ne vaut pas
`x[i]`, le Biais reste positif même avec M → ∞.

### Sur ID, le Biais n'est pas exactement zéro

Quand on dit « Biais ≈ 0 sur ID », c'est une simplification pédagogique.
En pratique, même sur des images normales, le Biais est petit mais non nul pour
plusieurs raisons :

**a) Bords et contours fins**

Un ConvTranspose2D ne peut pas reconstruire parfaitement les transitions nettes
entre zones de couleur. La résolution spatiale perdue lors de l'encodage (Max
Pooling) n'est pas entièrement récupérée. Ces erreurs de bord sont systématiques
— elles persistent même en moyennant infiniment de modèles.

**b) Artefacts du ConvTranspose2D (checkerboard artifacts)**

Le `ConvTranspose2D` produit des artefacts en damier sur certains pixels à cause
du recouvrement irrégulier des noyaux. Ces artefacts sont architecturalement
imposés — ils dépendent du stride et de la taille du kernel, pas de
l'initialisation. Ils contribuent donc au Biais, pas à la Variance.

```text
x[i]  = 0.52  (pixel de bord, image ID normale)
f̄[i]  = 0.58  (artefact de recouvrement du ConvTranspose2D)
Biais = (0.52 − 0.58)² = 0.0036   ← petit mais non nul
```

**c) Capacité limitée de l'architecture**

L'encodeur compresse l'image via des couches convolutives — des détails fins
(textures de cheveux, pores de peau) ne peuvent pas être stockés dans l'espace
latent. La décompression ne peut pas les recréer. Cette perte est fondamentale :
même le modèle consensuel `f̄` ne peut pas reconstruire ce qui n'est plus dans
la représentation.

### Où est ce terme dans la formule ?

Il n'y a pas de terme séparé — tout ça est contenu dans `( x[i] − f̄(x)[i] )²`.
C'est pour ça qu'on dit « irréductible » : ce n'est pas une erreur aléatoire
entre modèles (qui disparaîtrait en moyennant) mais une erreur systématique de
ce que l'ensemble peut représenter.

Sur ID : ces erreurs irréductibles sont faibles → Biais faible.
Sur OOD : une erreur systématique supplémentaire s'y ajoute (la zone inconnue)
→ Biais fort.

---

## 5. Risk élevé sur ID si les modèles sont mauvais individuellement mais se compensent

C'est un cas où les modèles ont des erreurs **opposées** qui s'annulent en
moyenne — et ça m'a posé des questions au début.

### Exemple numérique

Pixel `x[i] = 0.50`, 4 modèles :

| Modèle | Prédiction | Erreur (x − f̂ᵐ) |
|--------|-----------|-------------------|
| m = 1  | 0.20      | +0.30 |
| m = 2  | 0.80      | −0.30 |
| m = 3  | 0.15      | +0.35 |
| m = 4  | 0.85      | −0.35 |

Chaque modèle se trompe énormément. Pourtant :

```text
f̄ = (0.20 + 0.80 + 0.15 + 0.85) / 4 = 2.00 / 4 = 0.50

Biais    = (0.50 − 0.50)² = 0.0000   ← parfait !
Variance = (1/4) × [(0.20−0.50)² + (0.80−0.50)² + (0.15−0.50)² + (0.85−0.50)²]
         = (1/4) × [0.0900 + 0.0900 + 0.1225 + 0.1225]
         = (1/4) × 0.4250
         = 0.1063

Risk = Biais + Variance = 0.0000 + 0.1063 = 0.1063
```

Le Risk est très élevé (0.1063) même si c'est une image ID normale ! Ça arrive
parce que chaque modèle se trompe beaucoup, mais leurs erreurs sont exactement
opposées.

Si on déployait un seul modèle au hasard, on aurait en moyenne une erreur de
`0.1063`. Mais si on prend la moyenne de l'ensemble, l'erreur est 0.

→ Voilà pourquoi le Risk est un mauvais signal de détection OOD sur ID : il peut
être élevé même sur des images normales, à cause de la dispersion des modèles,
pas d'une vraie anomalie.

---

## 6. Problème d'effondrement d'ensemble

### Pourquoi ça arrive même avec des seeds différentes

Changer la seed change :
- L'ordre des batches pendant l'entraînement
- L'initialisation des poids

Mais ne change **pas** :
- L'architecture (mêmes couches, même structure)
- Les données d'entraînement (même distribution, mêmes patterns appris)
- La loss (même fonction objectif)
- Les biais inductifs (un ConvNet favorise les textures locales, les
  invariances de translation, etc.)

Résultat : les M modèles ont des poids différents, mais ils ont convergé vers
des **solutions similaires dans l'espace des fonctions**. Sur des inputs ID, ils
donnent des réponses légèrement différentes (d'où la Variance > 0 mais faible).
Sur des inputs OOD qui sortent de leur distribution commune, ils extrapolent tous
vers la même mauvaise réponse — celle dictée par leur biais inductif commun.

### Exemple concret

L'ensemble est entraîné uniquement sur des visages sans lunettes. Chaque modèle
apprend que la zone autour des yeux = peau + sourcils. C'est encodé dans les
filtres des 4 premières couches.

Quand une image avec lunettes arrive :

```text
Modèle 1 (seed=42) :  "je vois quelque chose bizarre autour des yeux → je reconstruis de la peau"
Modèle 2 (seed=7)  :  "je vois quelque chose bizarre autour des yeux → je reconstruis de la peau"
Modèle 3 (seed=123):  "je vois quelque chose bizarre autour des yeux → je reconstruis de la peau"

f̄ ≈ peau   (consensus fort)
Variance ≈ 0 (ils sont d'accord)
Biais >> 0  (l'image réelle montre des lunettes, pas de la peau)
```

Ce n'est pas une question de seed — c'est que les 3 modèles ont appris le même
concept "zone des yeux = peau". Des seeds différentes leur ont juste donné des
poids légèrement différents pour arriver au même résultat fonctionnel.

### Ce qui augmenterait vraiment la diversité

- Architectures hétérogènes (ConvNet + ViT + ResNet)
- Données d'entraînement différentes (sous-ensembles disjoints)
- Loss functions différentes (MSE + perceptual + adversarial)

Avec nos M modèles de même architecture, même données, seeds différentes : la
Variance reste un signal faible pour les OODs « systématiques » comme les
lunettes. C'est précisément pourquoi le Biais est le meilleur signal.

---

## 7. Pourquoi on estime toujours avec la moyenne

### La moyenne résume quoi ?

La moyenne d'un ensemble de valeurs `{v₁, v₂, …, vₙ}` est :

```text
v̄ = (1/n) Σᵢ vᵢ
```

Elle résume la **tendance centrale** de la distribution. C'est la valeur qui
minimise l'erreur quadratique totale :

```text
v̄ = argmin_c Σᵢ (vᵢ − c)²
```

Autrement dit : si on doit choisir un seul nombre pour représenter toute la
distribution, la moyenne est le choix optimal au sens des moindres carrés.

### Pourquoi l'utiliser partout

**Cas 1 — Moyenne sur les M modèles pour obtenir f̄**

On a M prédictions `{f̂¹(x)[i], f̂²(x)[i], …, f̂ᴹ(x)[i]}`. On cherche la
meilleure prédiction unique issue de l'ensemble. La moyenne est optimale au sens
MSE : si les erreurs des modèles sont indépendantes et de biais nul, la moyenne
réduit le bruit d'un facteur M.

```text
Variance de f̄ = Variance de f̂ᵐ / M   (si les erreurs sont indépendantes)
```

Plus M est grand, plus f̄ est proche de la vraie valeur.

**Cas 2 — Moyenne spatiale des pixels pour obtenir un scalaire**

On a une carte d'erreur `(H, W)`. On veut un seul nombre par image pour
comparer des images entre elles. La moyenne spatiale résume l'intensité globale
de l'anomalie sur toute l'image.

```text
Exemple image 2×2, carte d'erreur :
  [ 0.0  0.0 ]   ← zone normale
  [ 0.0  0.8 ]   ← pixel anomal (lunette)

Moyenne = (0.0 + 0.0 + 0.0 + 0.8) / 4 = 0.200
```

La valeur `0.200` capture qu'il y a un problème quelque part, même si trois
pixels sur quatre sont normaux.

**Cas 3 — Mean Abs Bias sous les tuiles**

Dans `mean_abs_bias.png`, le nombre affiché est `(1/HW) Σᵢⱼ |x − f̄|`. C'est
la distance L1 moyenne sur tous les pixels. Cette valeur résume en un chiffre
l'intensité globale de l'erreur de reconstruction pour cette image à ce bloc.

### La moyenne vs d'autres résumés

| Résumé | Ce qu'il capture | Usage dans ce projet |
|--------|-----------------|----------------------|
| Moyenne | Erreur globale, sensible à toute l'image | `mean_abs_bias`, Risk, Biais, Variance |
| Maximum | Pixel le plus anomal (un seul suffit) | `agg="max"` dans `_reduce_over_pixels` |
| Percentile 95 | Compromis — ignore les pixels isolés, sensible aux zones anomales | `agg="p95"` (défaut dans `aggregate_anomaly_score`) |

On utilise le p95 comme défaut précisément parce que la moyenne est trop diluée
sur 64×64 pixels : une petite paire de lunettes n'affecte qu'une fraction des
pixels, et la moyenne dilue son signal. Le p95 garantit que si 5% des pixels
sont très anormaux, le score global le reflète.

### La moyenne n'est pas la distribution

La moyenne est un **résumé**, pas la distribution complète. Deux images peuvent
avoir la même moyenne d'erreur mais des distributions très différentes :
- Image A : toute l'image légèrement floue → 64×64 pixels à erreur 0.010
- Image B : paire de lunettes nette → quelques pixels à erreur 0.640, le reste à 0.000

Même moyenne = 0.010, mais l'image B est OOD. C'est pour ça qu'on compare aussi
le maximum et le p95 — et pourquoi les heatmaps restent complémentaires aux
scalaires.
