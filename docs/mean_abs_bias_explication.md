# Comment sont calculés les nombres sous chaque tuile de `mean_abs_bias.png`

Ce document explique pas à pas comment on obtient le **petit nombre blanc
affiché en bas de chaque tuile** du plot `mean_abs_bias.png`, avec un exemple
numérique complet.

> **En une phrase :** le nombre sous une tuile, c'est la **moyenne spatiale de
> la carte d'erreur absolue** `|x − f̄_k|` (moyennée sur les canaux RGB), où
> `f̄_k` est la reconstruction *moyenne de l'ensemble* au bloc convolutif `k`.

C'est la vue **L1 (valeur absolue)** du terme de *biais* de la décomposition
biais/variance. Unités : pixels normalisés dans `[0, 1]`.

---

## 1. Ce que montre chaque tuile

Pour chaque image (une ligne `ID i` ou `OOD i`) on affiche :

```text
Original | |x − f̄| L0 | |x − f̄| L1 | ... | |x − f̄| Ln
```

- **Colonne 0** : l'image originale `x`.
- **Colonnes suivantes** : une heatmap de l'erreur `|x − f̄_k|` pour chaque
  bloc convolutif `k`.
- **Le nombre sous chaque carte** : la moyenne de cette carte d'erreur sur tous
  les pixels.

Toutes les tuiles d'erreur partagent la **même échelle de couleurs** et une
seule barre de couleur. En pratique, les lignes **ID** (normales) restent
sombres partout, et les lignes **OOD** s'allument sur la zone des lunettes.

---

## 2. La chaîne de calcul (avec références au code)

### Étape A — reconstruction par modèle

Chacun des `M = 10` modèles reconstruit l'image au bloc `k` :

```text
f_{m,k}(x)        # forme (B, 3, H, W),  m = 0 .. M-1
```

Code : `block_reconstructions(...)` appelé dans
[`sample_decomposition`](../lrad/ensemble.py#L115) (`lrad/ensemble.py`).

### Étape B — reconstruction moyenne de l'ensemble `f̄_k`

On moyenne sur les `M` modèles
([`decomposition_maps`, `lrad/ensemble.py:85`](../lrad/ensemble.py#L85)) :

```python
recons      = torch.stack([recons_per_model[m][k] for m in range(M)], dim=0)  # (M, B, 3, H, W)
mean_recon  = recons.mean(dim=0)                                              # (B, 3, H, W)  == f̄_k
```

Cette `mean_recon` est ce qui est passé au plot sous le nom `mean_recons`
([`scripts/run_ensemble.py:294`](../scripts/run_ensemble.py#L294)).

### Étape C — carte d'erreur absolue (moyenne sur RGB)

Dans le plot ([`plot_mean_abs_bias`, `lrad/plots.py:606`](../lrad/plots.py#L606))
on calcule, pour chaque image `r` et chaque bloc `j` :

```python
e = _abs_error(images_np[r], recons_np[j][r])   # (H, W)
```

avec ([`lrad/plots.py:127`](../lrad/plots.py#L127)) :

```python
def _abs_error(orig, recon):
    return np.abs(orig - recon).mean(axis=-1)    # |x - f̄| moyenné sur les 3 canaux RGB
```

Les images sont d'abord clampées dans `[0, 1]` puis réordonnées en `(H, W, 3)`
par `_to_image_grid` ([`lrad/plots.py:122`](../lrad/plots.py#L122)).

### Étape D — le nombre affiché = moyenne spatiale

Le nombre écrit sous la tuile est la moyenne de la carte `(H, W)` sur **tous
les pixels** ([`lrad/plots.py:638-642`](../lrad/plots.py#L638-L642)) :

```python
ax_e.text(0.5, 0.04, f"{float(e.mean()):.3f}", ...)
```

### Formule compacte

Pour l'image `x`, le bloc `k`, l'ensemble de `M` modèles, image `H × W`,
3 canaux :

```text
f̄_k = (1/M) · Σ_{m=1..M} f_{m,k}(x)

nombre(k) = (1 / (H · W)) · Σ_{i,j} [ (1/3) · Σ_{c∈{R,G,B}} | x_{i,j,c} − f̄_{k,i,j,c} | ]
```

Soit la **distance L1 moyenne par pixel** entre l'image et la reconstruction
moyenne de l'ensemble.

---

## 3. Exemple numérique complet

Pour rester lisible : une image minuscule `H = W = 2` (4 pixels), 3 canaux RGB,
et un ensemble de `M = 3` modèles, pour un seul bloc `k`.
(Dans le plot réel : `H = W = 64`, `M = 10`.)

### Données : image originale `x`

| pixel | R    | G    | B    |
|-------|------|------|------|
| (0,0) | 0.80 | 0.80 | 0.80 |
| (0,1) | 0.50 | 0.50 | 0.50 |
| (1,0) | 0.20 | 0.20 | 0.20 |
| (1,1) | 0.60 | 0.30 | 0.30 |

### Reconstructions des 3 modèles (bloc `k`)

| pixel | modèle 1 (R,G,B)     | modèle 2 (R,G,B)     | modèle 3 (R,G,B)     |
|-------|----------------------|----------------------|----------------------|
| (0,0) | 0.70, 0.70, 0.70     | 0.74, 0.74, 0.74     | 0.78, 0.78, 0.78     |
| (0,1) | 0.55, 0.55, 0.55     | 0.50, 0.50, 0.50     | 0.45, 0.45, 0.45     |
| (1,0) | 0.25, 0.25, 0.25     | 0.20, 0.20, 0.20     | 0.30, 0.30, 0.30     |
| (1,1) | 0.50, 0.30, 0.30     | 0.56, 0.36, 0.36     | 0.62, 0.30, 0.30     |

### Étape B — moyenne d'ensemble `f̄_k`

| pixel | R    | G    | B    | détail                              |
|-------|------|------|------|-------------------------------------|
| (0,0) | 0.74 | 0.74 | 0.74 | (0.70+0.74+0.78)/3 = 0.74           |
| (0,1) | 0.50 | 0.50 | 0.50 | (0.55+0.50+0.45)/3 = 0.50           |
| (1,0) | 0.25 | 0.25 | 0.25 | (0.25+0.20+0.30)/3 = 0.25           |
| (1,1) | 0.56 | 0.32 | 0.32 | R: (0.50+0.56+0.62)/3=0.56 ; G,B: (0.30+0.36+0.30)/3=0.32 |

### Étape C — erreur absolue, puis moyenne sur RGB → carte `(H, W)`

| pixel | \|R\|              | \|G\| | \|B\| | moyenne RGB |
|-------|--------------------|-------|-------|-------------|
| (0,0) | \|0.80−0.74\|=0.06 | 0.06  | 0.06  | **0.060**   |
| (0,1) | \|0.50−0.50\|=0.00 | 0.00  | 0.00  | **0.000**   |
| (1,0) | \|0.20−0.25\|=0.05 | 0.05  | 0.05  | **0.050**   |
| (1,1) | \|0.60−0.56\|=0.04 | \|0.30−0.32\|=0.02 | 0.02 | (0.04+0.02+0.02)/3 = **0.0267** |

### Étape D — moyenne spatiale = nombre affiché sous la tuile

```text
nombre(k) = (0.060 + 0.000 + 0.050 + 0.0267) / 4
          = 0.13667 / 4
          = 0.03417
          ≈ 0.034     # affiché en .3f sous la tuile
```

**Résultat : `0.034`** — c'est le chiffre blanc qui apparaîtrait sous cette tuile.

---

## 4. Interprétation

- **Petit nombre** (tuile sombre) → la reconstruction moyenne colle bien à
  l'image : peu de biais. C'est ce qu'on observe sur les lignes **ID**.
- **Grand nombre** (tuile lumineuse) → forte erreur de reconstruction = biais
  élevé, signe d'anomalie. C'est ce qu'on voit sur les lignes **OOD** au niveau
  de la zone des lunettes.
- En comparant les colonnes (L0, L1, …) on voit à **quel bloc convolutif** le
  biais OOD devient le plus discriminant.

> Lien avec la décomposition biais/variance : ici on affiche `|x − f̄_k|` (norme
> **L1**). Le terme de *biais* « officiel » de la décomposition
> (`risk = bias + variance`) utilise la version **L2** au carré
> `(x − f̄_k)²` — voir [`lrad/ensemble.py:89`](../lrad/ensemble.py#L89).
> Cette tuile est donc la même quantité en valeur absolue plutôt qu'au carré.

---

## 5. Reproduire le calcul (extrait minimal)

```python
import numpy as np

x = np.array([[[0.80,0.80,0.80],[0.50,0.50,0.50]],
              [[0.20,0.20,0.20],[0.60,0.30,0.30]]])          # (H, W, 3)

recons = np.array([
    [[[0.70]*3,[0.55]*3],[[0.25]*3,[0.50,0.30,0.30]]],       # modèle 1
    [[[0.74]*3,[0.50]*3],[[0.20]*3,[0.56,0.36,0.36]]],       # modèle 2
    [[[0.78]*3,[0.45]*3],[[0.30]*3,[0.62,0.30,0.30]]],       # modèle 3
])                                                            # (M, H, W, 3)

mean_recon = recons.mean(axis=0)                              # f̄_k  (étape B)
err_map    = np.abs(x - mean_recon).mean(axis=-1)            # |x-f̄| sur RGB (étape C)
number     = float(err_map.mean())                           # moyenne spatiale (étape D)

print(round(number, 3))   # -> 0.034
```
