# MESURES — formulation mathématique de toutes les mesures OOD du projet

*Dernière mise à jour : 18 juillet 2026 — couvre les runs `ensemble_20260707_152254_6754917` (baseline, 10 modèles 64 px), `ensemble_20260716_170930_6774881` (validation sur splits complets) et `gridsearch_6781780` (recherche d'hyperparamètres CutPaste).*

Chaque section donne : la **définition formelle**, l'**interprétation**, et un
**exemple numérique** entièrement calculé. Les équations sont en LaTeX
(rendues par GitHub et l'aperçu Markdown de VSCode).

## Sommaire

1. [Notation](#1-notation)
2. [AUROC](#2-auroc)
3. [Scores du classifieur : MSP et entropies](#3-scores-du-classifieur)
4. [Décomposition Risque = Biais + Variance](#4-décomposition-risque--biais--variance)
5. [Incertitude prédictive](#5-incertitude-prédictive)
6. [Énergie](#6-énergie)
7. [Score localisé : z-score par pixel + patch-max](#7-score-localisé)
8. [locfre — erreur de features localisée](#8-locfre)
9. [Fusion par rang](#9-fusion-par-rang)
10. [Fusion supervisée](#10-fusion-supervisée)
11. [CutPaste](#11-cutpaste)
12. [Grid search multi-métriques](#12-grid-search-multi-métriques)
13. [Historique des résultats](#13-historique-des-résultats)
14. [Carte du code](#14-carte-du-code)

---

## 1. Notation

| Symbole | Signification |
| --- | --- |
| $x \in [0,1]^{3\times H\times W}$ | image d'entrée (visage aligné CelebA) |
| $M$ | taille de l'ensemble ($M{=}10$ au run de référence) |
| $\hat f^{\,m}_k(x)$ | reconstruction de $x$ par les décodeurs du membre $m$ au bloc $k$ |
| $\bar f_k(x) = \frac{1}{M}\sum_{m=1}^{M} \hat f^{\,m}_k(x)$ | reconstruction **consensus** de l'ensemble |
| $a^m_j(x)$ | activations du bloc $j$ du tronc du membre $m$ |
| $p^m(c\mid x)$ | probabilité softmax de la classe $c$ par le membre $m$ |
| $y \in \{0,1\}$ | étiquette OOD : $0$ = in-distribution (sans lunettes), $1$ = OOD (lunettes) |
| $s(x)$ | un *score* d'anomalie — par convention, **plus grand sur les OOD** |

Toute mesure du projet est un score $s : x \mapsto \mathbb{R}$, et sa qualité
est jugée par l'AUROC (§2).

---

## 2. AUROC

### 2.1 Définition probabiliste

$$
\mathrm{AUROC}(s) \;=\; \mathbb{P}\big(s(X_{\mathrm{ood}}) > s(X_{\mathrm{in}})\big)
\;+\; \tfrac{1}{2}\,\mathbb{P}\big(s(X_{\mathrm{ood}}) = s(X_{\mathrm{in}})\big)
$$

où $X_{\mathrm{in}}$ et $X_{\mathrm{ood}}$ sont tirés uniformément dans les
splits de test respectifs.

**Interprétation.** C'est la probabilité qu'en tirant *une* image OOD et
*une* image normale au hasard, le score classe l'OOD au-dessus. $0.5$ =
hasard pur ; $1.0$ = séparation parfaite ; $<0.5$ = le score est informatif
mais **inversé**. Il n'y a aucun seuil de décision : l'AUROC juge le
classement tout entier.

### 2.2 Estimateur empirique (statistique de Wilcoxon–Mann–Whitney)

Avec $n$ images in-dist de scores $\{s_i^{\mathrm{in}}\}$ et $n'$ images OOD
de scores $\{s_j^{\mathrm{ood}}\}$ :

$$
\widehat{\mathrm{AUROC}}
= \frac{1}{n\,n'} \sum_{i=1}^{n}\sum_{j=1}^{n'}
\left[ \mathbf{1}\!\left\{ s_j^{\mathrm{ood}} > s_i^{\mathrm{in}} \right\}
+ \tfrac12\, \mathbf{1}\!\left\{ s_j^{\mathrm{ood}} = s_i^{\mathrm{in}} \right\} \right]
$$

**Interprétation.** On compte les paires bien ordonnées parmi les $n\,n'$
paires possibles (une égalité vaut ½). C'est exactement la statistique $U$
de Mann–Whitney normalisée : $\mathrm{AUROC} = U/(n\,n')$.

**Exemple.** $n{=}3$ normales $s^{\mathrm{in}} = (0.2,\; 0.5,\; 0.9)$ et
$n'{=}2$ OOD $s^{\mathrm{ood}} = (0.6,\; 0.8)$. Les $6$ comparaisons :

$$
\underbrace{0.2<0.6}_{\checkmark}\quad
\underbrace{0.2<0.8}_{\checkmark}\quad
\underbrace{0.5<0.6}_{\checkmark}\quad
\underbrace{0.5<0.8}_{\checkmark}\quad
\underbrace{0.9>0.6}_{\times}\quad
\underbrace{0.9>0.8}_{\times}
\qquad\Rightarrow\qquad
\widehat{\mathrm{AUROC}} = \frac{4}{6} \approx 0.667
$$

L'image normale à $0.9$ (p. ex. des cheveux mal reconstruits) coûte à elle
seule 2 paires : c'est le mécanisme exact qui limitait le score p95 global
à $0.638$.

### 2.3 Courbe ROC et équivalence

Pour un seuil $t$, on déclare « OOD » si $s(x) > t$ :

$$
\mathrm{TPR}(t) = \mathbb{P}\big(s(X_{\mathrm{ood}}) > t\big),
\qquad
\mathrm{FPR}(t) = \mathbb{P}\big(s(X_{\mathrm{in}}) > t\big)
$$

La courbe ROC est le lieu $\{(\mathrm{FPR}(t),\,\mathrm{TPR}(t)) : t \in \mathbb{R}\}$, et

$$
\mathrm{AUROC} \;=\; \int_0^1 \mathrm{TPR}\;\mathrm{d}\,\mathrm{FPR}
$$

**Interprétation.** L'aire sous la courbe des compromis « détection vs
fausses alarmes ». La diagonale $\mathrm{TPR}=\mathrm{FPR}$ est le hasard.
C'est le panneau central de `plots/fused_auroc.png`.

### 2.4 Propriété clé : invariance monotone

Pour toute fonction strictement croissante $g$ :

$$
\mathrm{AUROC}(g \circ s) = \mathrm{AUROC}(s)
$$

**Interprétation.** Seul **l'ordre** des scores compte, jamais leur échelle.
C'est ce qui autorise à remplacer un score par son rang (fusion §9) sans
rien perdre.

---

## 3. Scores du classifieur

Intuition commune : sur une image OOD, le classifieur **hésite** — sa
distribution prédictive s'aplatit.

### 3.1 MSP (Maximum Softmax Probability)

Avec les logits $z \in \mathbb{R}^C$ de la tête gender et
$p(c\mid x) = \mathrm{softmax}(z)_c = e^{z_c} / \sum_{c'} e^{z_{c'}}$ :

$$
s_{\mathrm{MSP}}(x) \;=\; 1 - \max_{c}\, p(c \mid x)
$$

**Exemple.** $p = (0.98,\, 0.02) \Rightarrow s = 0.02$ (confiant, in-dist) ;
$p = (0.55,\, 0.45) \Rightarrow s = 0.45$ (hésitant, suspect).

### 3.2 Entropie de Shannon (tête gender)

$$
\mathcal{H}(p) \;=\; -\sum_{c=1}^{C} p_c \ln p_c
\qquad\in\; [0,\; \ln C]
$$

**Exemple.** $\mathcal{H}(0.9, 0.1) = -0.9\ln 0.9 - 0.1\ln 0.1 = 0.095 +
0.230 = 0.325$ nats ; le maximum $\mathcal{H}(0.5,0.5) = \ln 2 \approx
0.693$ est atteint à l'hésitation totale.

**Interprétation.** Comme le MSP mais sensible à *toute* la distribution,
pas seulement au maximum.

### 3.3 Entropie de Bernoulli des attributs — et sa forme stable

Pour un attribut de probabilité $p = \sigma(z)$ (sigmoïde du logit $z$) :

$$
\mathcal{H}_b(p) = -p\ln p - (1-p)\ln(1-p)
$$

Problème : en float32, $1 - 10^{-12}$ s'arrondit à $1.0$, donc un attribut
saturé donne $\ln 0 = -\infty$ et un score NaN. On utilise l'identité
$\ln p = -\mathrm{softplus}(-z)$ et $\ln(1-p) = -\mathrm{softplus}(z)$, d'où
la forme **exacte et stable pour tout $z$** :

$$
\mathcal{H}_b\big(\sigma(z)\big)
= \sigma(z)\,\mathrm{softplus}(-z) + \big(1-\sigma(z)\big)\,\mathrm{softplus}(z),
\qquad \mathrm{softplus}(z) = \ln(1+e^{z})
$$

**Résultats mesurés** (run de référence) : entropie gender $0.45$–$0.79$
selon le membre ; MSP $0.44$–$0.76$ ; entropie attrs $\approx 0.50$ (bruit
pur — les 6 attributs restent prédictibles avec des lunettes), ce qui
**diluait** le score combiné historique.

---

## 4. Décomposition Risque = Biais + Variance

### 4.1 Définitions

Au bloc $k$, au pixel $i$, l'erreur est la L2 au carré **sommée sur RGB**
(un pixel vit donc dans $[0,3]$) :

$$
\mathrm{Risque}_k(x)[i] = \frac{1}{M}\sum_{m=1}^{M} \big( x[i] - \hat f^{\,m}_k(x)[i] \big)^2
$$

$$
\mathrm{Biais}_k(x)[i] = \big( x[i] - \bar f_k(x)[i] \big)^2
\qquad\qquad
\mathrm{Variance}_k(x)[i] = \frac{1}{M}\sum_{m=1}^{M} \big( \hat f^{\,m}_k(x)[i] - \bar f_k(x)[i] \big)^2
$$

### 4.2 L'identité exacte et sa preuve

$$
\boxed{\;\mathrm{Risque}_k(x)[i] \;=\; \mathrm{Biais}_k(x)[i] + \mathrm{Variance}_k(x)[i]\;}
$$

*Preuve.* En posant $e_m = x[i] - \hat f^{\,m}_k(x)[i]$ et
$\bar e = \frac1M \sum_m e_m = x[i] - \bar f_k(x)[i]$ :

$$
\frac1M \sum_m e_m^2
= \frac1M \sum_m \big( (e_m - \bar e) + \bar e \big)^2
= \underbrace{\bar e^{\,2}}_{\mathrm{Biais}}
+ \underbrace{\frac1M \sum_m (e_m - \bar e)^2}_{\mathrm{Variance}}
+ \underbrace{\frac{2\bar e}{M} \sum_m (e_m - \bar e)}_{=\,0}
\qquad\blacksquare
$$

C'est la décomposition biais–variance classique, appliquée à l'ensemble
comme approximation empirique de $\mathbb{E}_{\mathcal{D}}$ (l'espérance
sur les entraînements). Le code la vérifie pixel par pixel : résidu
$\approx 2\times 10^{-7}$ (bruit float32).

**Interprétation.** Le **Biais** est l'erreur *du consensus* — ce
qu'*aucun* membre ne sait reconstruire, donc l'anomalie (les lunettes,
absentes du train). La **Variance** est le *désaccord* entre membres —
l'incertitude épistémique. Le **Risque** est le coût d'un membre moyen.

**Exemple.** Un pixel, $M{=}3$, $x = 0.8$,
$\hat f = (0.5,\, 0.6,\, 0.7)$, donc $\bar f = 0.6$ :

$$
\mathrm{Risque} = \tfrac{0.09 + 0.04 + 0.01}{3} = 0.0467,\qquad
\mathrm{Biais} = (0.8-0.6)^2 = 0.04,\qquad
\mathrm{Variance} = \tfrac{0.01 + 0 + 0.01}{3} = 0.0067
$$

$$
0.04 + 0.0067 = 0.0467\;\checkmark
$$

### 4.3 Réduction pixel → image

Une carte $A(x)\in\mathbb{R}^{H\times W}$ devient un scalaire par :

$$
s_{\mathrm{mean}}(x) = \frac{1}{HW}\sum_i A(x)[i],
\qquad
s_{\max}(x) = \max_i A(x)[i],
\qquad
s_{p95}(x) = Q_{0.95}\big(A(x)\big)
$$

où $Q_{0.95}$ est le quantile empirique à 95 %. Le score **historique** du
projet était $s_{p95}(\mathrm{Biais})$ moyenné sur les blocs : AUROC $0.638$.

**Pourquoi il plafonne.** Des lunettes couvrent $\lesssim 500$ pixels sur
$64^2 = 4096$, soit $\lesssim 12\,\%$. Or $Q_{0.95}$ ne « voit » que les
5 % de pixels les plus hauts — un budget que cheveux et fond épuisent sur
*chaque* visage normal. Mesuré :
$\mathbb{E}[s_{p95} \mid \mathrm{in}] \approx 0.95$ vs
$\mathbb{E}[s_{p95} \mid \mathrm{ood}] \approx 1.10$ — un écart de 15 %
seulement, très recouvrant.

---

## 5. Incertitude prédictive

La même décomposition, appliquée aux **probabilités** des têtes. Pour une
tête de prédictions $p^m$ :

$$
\mathrm{U}_{\mathrm{tot}}(x) = \mathcal{H}\!\Big( \frac1M \sum_{m} p^m(\cdot\mid x) \Big),
\qquad
\mathrm{U}_{\mathrm{al}}(x) = \frac1M \sum_{m} \mathcal{H}\big( p^m(\cdot\mid x) \big)
$$

$$
\boxed{\;\mathrm{U}_{\mathrm{ep}}(x) = \mathrm{U}_{\mathrm{tot}}(x) - \mathrm{U}_{\mathrm{al}}(x) \;\ge\; 0\;}
$$

La positivité vient de la concavité de $\mathcal{H}$ (inégalité de Jensen).
$\mathrm{U}_{\mathrm{ep}}$ est exactement l'**information mutuelle**
$I(y\,;\,m \mid x)$ entre la prédiction et l'identité du membre : elle est
non nulle *si et seulement si* les membres prédisent différemment.

**Interprétation.**

- $\mathrm{U}_{\mathrm{al}}$ (aléatoire) : l'ambiguïté *intrinsèque* de
  l'image, sur laquelle tous les membres s'accordent.
- $\mathrm{U}_{\mathrm{ep}}$ (épistémique) : le *désaccord* — le manque de
  connaissance. Sur une image OOD, chaque membre extrapole à sa façon
  $\Rightarrow$ désaccord $\Rightarrow$ signal OOD.

**Exemple ($M{=}2$).**

*Cas 1 — d'accord mais hésitants* : $p^1 = p^2 = (0.6, 0.4)$.

$$
\mathrm{U}_{\mathrm{tot}} = \mathcal{H}(0.6,0.4) = 0.673,\quad
\mathrm{U}_{\mathrm{al}} = 0.673
\;\Rightarrow\; \mathrm{U}_{\mathrm{ep}} = 0
$$

*Cas 2 — sûrs mais contradictoires* : $p^1 = (0.9, 0.1)$, $p^2 = (0.1, 0.9)$.

$$
\bar p = (0.5, 0.5) \;\Rightarrow\; \mathrm{U}_{\mathrm{tot}} = \ln 2 = 0.693,\quad
\mathrm{U}_{\mathrm{al}} = \mathcal{H}(0.9,0.1) = 0.325
\;\Rightarrow\; \mathrm{U}_{\mathrm{ep}} = 0.368
$$

Même hésitation moyenne apparente, mais seule la contradiction (cas 2)
produit de l'épistémique. Le score `unc_epistemic_combined` (gender +
attrs) atteint $0.740$ — le meilleur signal individuel du run de référence.

---

## 6. Énergie

Pour les logits $z = (z_0, z_1)$ de la tête gender, moyennée sur l'ensemble :

$$
E(x) \;=\; -\frac1M \sum_{m=1}^{M} \ln\!\Big( e^{z^m_0} + e^{z^m_1} \Big)
\;=\; -\frac1M \sum_m \mathrm{logsumexp}\big(z^m\big)
$$

**Interprétation.** Un réseau entraîné pousse le logit de la vraie classe
très haut sur ses données $\Rightarrow$ $\mathrm{logsumexp}$ grand
$\Rightarrow$ énergie très négative. Sur une image OOD les logits restent
petits $\Rightarrow$ l'énergie **monte**. Contrairement au MSP, l'énergie
n'est pas normalisée par le softmax : elle garde l'information d'*amplitude*
des logits, que la normalisation détruit.

**Exemple.** In-dist confiant $z = (8, -2)$ :
$E = -\ln(e^{8} + e^{-2}) \approx -8.0$. OOD mou $z = (1.2,\, 0.8)$ :
$E = -\ln(e^{1.2} + e^{0.8}) \approx -1.70$. On a bien $-1.70 > -8.0$. ✓

Mesuré : $0.72$–$0.73$ sur l'ensemble de référence — et jusqu'à **$0.80$**
après entraînement CutPaste (§12), le pretext amplifiant l'effondrement des
logits sur les vraies occlusions.

---

## 7. Score localisé

Deux corrections successives du $s_{p95}$ global
([lrad/localized.py](../lrad/localized.py)).

### 7.1 z-score par pixel

Sur $N_{\mathrm{ref}}$ images in-dist de référence (val, ou slice du
train), on estime **par position** $i$ :

$$
\mu(i) = \frac{1}{N_{\mathrm{ref}}} \sum_{r} \mathrm{Biais}(x_r)[i],
\qquad
\sigma(i) = \sqrt{ \frac{1}{N_{\mathrm{ref}}} \sum_r \mathrm{Biais}(x_r)[i]^2 - \mu(i)^2 }
$$

$$
z(x)[i] \;=\; \frac{ \mathrm{Biais}(x)[i] - \mu(i) }{ \max\big(\sigma(i),\, \sigma_{\min}\big) },
\qquad \sigma_{\min} = 10^{-3}
$$

**Interprétation.** CelebA étant aligné, chaque position de pixel a une
sémantique stable. $\mu(i)$ neutralise les zones *structurellement* mal
reconstruites (cheveux, fond) ; $\sigma(i)$ **amplifie** les zones
d'ordinaire faciles (le visage aligné) : un petit écart y devient un $z$
énorme. Le plancher $\sigma_{\min}$ empêche le bruit float d'exploser aux
positions quasi déterministes. C'est l'effet « région des yeux » obtenu
*sans jamais coder de région*.

**Exemple.** Pixel de fond : $\mathrm{Biais} = 0.9$, $\mu = 0.85$,
$\sigma = 0.3 \Rightarrow z = 0.17$ (banal). Pixel d'œil :
$\mathrm{Biais} = 0.25$, $\mu = 0.05$, $\sigma = 0.02 \Rightarrow z = 10$.
L'erreur brute du fond était $3.6\times$ plus grande — mais c'est l'œil qui
domine après normalisation.

### 7.2 Patch-max multi-échelle

Sur la carte $z$, pour des fenêtres carrées $W_k(u)$ de côté
$k \in \mathcal{K} = \{4, 8, 16\}$ centrées en $u$ (stride $k/2$,
chevauchantes) :

$$
s_{\mathrm{loc}}(x) \;=\; \max_{k \in \mathcal{K}}\; \max_{u}\;
\frac{1}{k^2} \sum_{i \in W_k(u)} z(x)[i]
$$

**Interprétation.** La moyenne intra-fenêtre écrase le bruit d'un pixel
isolé ; le max spatial capte une anomalie localisée **où qu'elle soit** ;
le max sur les échelles rend le score robuste à la taille inconnue de
l'anomalie. Test unitaire clé : une tache couvrant 3.5 % des pixels est
invisible pour $Q_{0.95}$ (car $3.5\% < 5\%$) mais sature une fenêtre du
patch-max.

**Résultat** : biais $0.638 \rightarrow 0.686$ (splits complets). Le
plafond restant vient du flou des décodeurs $\Rightarrow$ §8.

---

## 8. locfre

*Localized Feature Reconstruction Error* — le meilleur signal individuel :
$0.779$ ([lrad/feature_error.py](../lrad/feature_error.py)).

### 8.1 Construction

Soit $k_d$ le bloc le plus profond et
$\bar f(x) = \frac1M \sum_m \hat f^{\,m}_{k_d}(x)$ le consensus. En notant
$\hat c(v) = v / \lVert v \rVert_2$ la normalisation L2 du vecteur de
canaux, la carte d'erreur au bloc $j$, à la position spatiale $u$ :

$$
\ell_j(x)[u] \;=\; \frac{1}{M} \sum_{m=1}^{M}
\Big\lVert\, \hat c\big(a^m_j(x)[u]\big) \;-\; \hat c\big(a^m_j(\bar f(x))[u]\big) \Big\rVert_2^2
$$

puis z-score par position (comme §7.1) et patch-max (fenêtres $\{2,4,8\}$
adaptées à la résolution du bloc). Blocs par défaut : $j \in \{1, 3\}$
(texture fine $16{\times}16$ / sémantique $4{\times}4$ — les plus forts et
complémentaires).

### 8.2 Relation avec le cosinus et bornes

Pour des vecteurs unitaires, $\lVert \hat u - \hat v \rVert^2 =
2\,(1 - \cos\theta)$, donc :

$$
\ell_j(x)[u] \;\in\; [0,\, 4],
\qquad
\ell_j = 0 \iff \text{mêmes directions},\quad
\ell_j = 4 \iff \text{directions opposées}
$$

**Interprétation — pourquoi ça bat l'espace pixel.** Les décodeurs, entraînés
sur des visages sans lunettes, produisent un $\bar f(x)$ **sans lunettes**
même quand $x$ en porte : la reconstruction *efface l'anomalie*. L'erreur
pixel compare des couleurs et se noie dans le flou du décodeur ; l'erreur
de *features normalisées* ignore le contraste et le flou (invariance
d'échelle de $\hat c$) mais détecte un objet **sémantiquement absent** : à
la position des lunettes, $a_j(x)$ pointe vers « monture/verre/reflet » et
$a_j(\bar f(x))$ vers « œil/sourcil/peau » — deux directions quasi
orthogonales, $\cos\theta \approx 0 \Rightarrow \ell \approx 2$, alors
qu'un visage nu donne $\cos\theta \approx 1 \Rightarrow \ell \approx 0$.

---

## 9. Fusion par rang

Soit $S$ signaux $s_1, \dots, s_S$ évalués sur les $N = n + n'$ images du
pool de test. Le rang $r_s(x) \in \{0, \dots, N-1\}$ est la position de
$x$ dans le tri croissant du signal $s$. La fusion (poids $w_s \ge 0$,
uniformes par défaut) :

$$
F(x) \;=\; \frac{1}{\sum_s w_s} \sum_{s=1}^{S} w_s\, \frac{r_s(x)}{N-1}
$$

**Interprétation.** Par l'invariance monotone (§2.4), remplacer $s$ par son
rang ne change pas son AUROC individuelle — mais rend les signaux
**commensurables** ($r/(N{-}1) \in [0,1]$ pour tous), là où une moyenne des
scores bruts serait dominée par l'échelle du plus grand. Aucune étiquette
ni calibration n'est utilisée : la fusion reste totalement non supervisée.
Les erreurs des signaux étant faites sur des images *différentes*, la
moyenne des rangs les compense.

**Exemple ($N{=}4$ : in $=\{A,B\}$, ood $=\{C,D\}$, $S{=}2$).**

$$
\begin{array}{c|cc|cc|c}
 & s_1 & r_1/3 & s_2 & r_2/3 & F \\ \hline
A\ (\mathrm{in})  & 1.2 & 0    & -8.1 & 1/3 & 0.17 \\
B\ (\mathrm{in})  & 3.0 & 2/3  & -9.0 & 0   & 0.33 \\
C\ (\mathrm{ood}) & 2.1 & 1/3  & -5.0 & 1   & 0.67 \\
D\ (\mathrm{ood}) & 4.5 & 1    & -6.2 & 2/3 & 0.83 \\
\end{array}
$$

$s_1$ se trompe ($B$ au-dessus de $C$), $s_2$ aussi (ordre interne),
mais $F$ ordonne parfaitement $\{A,B\} < \{C,D\}$ :
$\mathrm{AUROC}(F) = 1$ alors qu'aucun signal seul n'y arrive.

**Recette validée** : $\{\ell_1,\, \ell_3,\, \mathrm{U}_{\mathrm{ep}},\, E\}$
($+\ \mathrm{cutpaste\_prob}$ si disponible, §11) $\Rightarrow$ **0.803**
sur splits complets.

---

## 10. Fusion supervisée

### 10.1 Modèle

Chaque signal est z-normalisé avec les statistiques de **calibration**
$(\mu_s, \sigma_s)$, puis combiné linéairement :

$$
F_{\sup}(x) \;=\; \sum_{s=1}^{S} w_s\, \frac{s(x) - \mu_s}{\sigma_s} \;+\; b
$$

Les poids minimisent la log-vraisemblance négative de la régression
logistique sur le jeu de calibration $\{(x_i, y_i)\}$ :

$$
(w^\star, b^\star) \;=\; \arg\min_{w,b}\;
\sum_i \Big[ -y_i \ln \sigma\big(F_{\sup}(x_i)\big) - (1-y_i) \ln\big(1 - \sigma(F_{\sup}(x_i))\big) \Big]
$$

avec $\sigma(t) = 1/(1+e^{-t})$. $F_{\sup}(x)$ est le logit de
$\mathbb{P}(\mathrm{OOD} \mid x)$ estimé.

### 10.2 Protocole anti-fuite

$$
\underbrace{\text{pool OOD } (13\,193)}_{\text{split seed }42}
\;\longrightarrow\;
\begin{cases}
\text{moitié A } (6\,596) & \to \text{calibration } (y{=}1) \\
\text{moitié B } (6\,597) & \to \text{évaluation seulement}
\end{cases}
$$

$$
\text{train in-dist} \to 6\,596 \text{ négatifs de calibration } (y{=}0),
\qquad
\text{test\_in } (18\,941) \to \text{évaluation seulement}
$$

L'AUROC finale est mesurée sur `test_in` vs **moitié B** — des images que
la régression n'a jamais vues. Split déterministe par permutation seedée
(`split_loader`, [lrad/dataset.py](../lrad/dataset.py)).

### 10.3 Poids appris (run 6774881) et lecture

$$
w \approx
\begin{cases}
+6.26 & \mathrm{U}_{\mathrm{ep}}^{\mathrm{comb}} \\
+2.40 & \mathrm{MSP}^{\mathrm{ens}} \\
+0.95 & \ell_3 \\
+0.48 & E \\
-4.44 & \mathrm{U}_{\mathrm{ep}}^{\mathrm{gender}} \\
-2.46 & \mathrm{U}_{\mathrm{tot}}^{\mathrm{comb}}
\end{cases}
$$

**Interprétation.** Les poids **négatifs** sont l'avantage décisif sur la
moyenne des rangs : $\mathrm{U}_{\mathrm{ep}}^{\mathrm{comb}}$ et
$\mathrm{U}_{\mathrm{ep}}^{\mathrm{gender}}$ sont très corrélés, la
régression garde l'un ($+6.26$) et **soustrait** l'autre ($-4.44$) pour ne
retenir que l'information non redondante — une décorrélation qu'aucune
moyenne pondérée positive ne peut réaliser. Gain : $0.810$ vs $0.803$.
Coût : des exemples OOD étiquetés, et des poids spécialisés « lunettes ».

---

## 11. CutPaste

Tout ce qui précède est *post-hoc*. CutPaste modifie **l'entraînement**
pour apprendre le concept « quelque chose recouvre le visage », sans donnée
OOD réelle ([lrad/cutpaste.py](../lrad/cutpaste.py)).

### 11.1 Augmentation

Chaque image du batch est altérée avec probabilité $p$. Un rectangle est
découpé dans une image **donneuse** (l'image suivante du batch — donc de la
vraie texture faciale) et collé à une position aléatoire. Forme *patch* :
aire relative $\alpha \sim \mathcal{U}(\alpha_{\min}, \alpha_{\max})$ et
aspect $\rho \sim \mathcal{U}(\rho_{\min}, \rho_{\max})$ donnent

$$
h = \sqrt{\alpha H W \rho},
\qquad
w = \sqrt{\alpha H W / \rho}
$$

Forme *scar* : sliver de $2$–$8$ px $\times$ $10$–$45\,\%$ du côté, dans
une orientation aléatoire ; mélange contrôlé par $p_{\mathrm{scar}}$.

**Exemple.** $64{\times}64$, $\alpha = 0.10$, $\rho = 2$ :
$h = \sqrt{0.1 \cdot 4096 \cdot 2} \approx 29$,
$w = \sqrt{0.1 \cdot 4096 / 2} \approx 14$ — un rectangle $29 \times 14$,
l'ordre de grandeur d'une paire de lunettes.

### 11.2 Perte d'entraînement

Une 3ᵉ tête binaire prédit *intact* ($\tilde y = 0$) vs *altéré*
($\tilde y = 1$). Avec $\mathcal{I}$ l'ensemble des images restées
intactes du batch :

$$
\mathcal{L} \;=\;
\underbrace{\mathcal{L}_{\mathrm{CE}}^{\mathrm{gender}}(\mathcal{I})}_{\text{supervisé}}
+ \lambda_a\, \underbrace{\mathcal{L}_{\mathrm{BCE}}^{\mathrm{attrs}}(\mathcal{I})}_{\text{supervisé}}
+ w_{cp}\, \underbrace{\mathcal{L}_{\mathrm{CE}}^{cp}(\text{batch entier})}_{\text{pretext}}
$$

**Interprétation.** Les pertes supervisées sont restreintes à $\mathcal{I}$
car un patch peut occulter les sourcils dont dépendent les labels ; la
perte pretext couvre tout le batch. $w_{cp}$ règle l'équilibre — le grid
search (§12) montre que $w_{cp} = 0.5$ bat $2.0$ : trop de pretext dégrade
les autres signaux.

### 11.3 Signal à l'évaluation

$$
s_{cp}(x) \;=\; \frac{1}{M} \sum_{m=1}^{M} \mathbb{P}_m\big(\text{altéré} \mid x\big)
$$

**Interprétation.** Entraîné uniquement sur des collages synthétiques, le
concept *généralise* aux occlusions réelles : de vraies lunettes font
monter $\mathbb{P}(\text{altéré})$. Mesuré : jusqu'à $0.746$ pour un seul
petit modèle. Le signal entre automatiquement dans les deux fusions
(§9–§10) quand les membres portent la tête.

---

## 12. Grid search multi-métriques

Comment choisir les hyperparamètres CutPaste **sans** entraîner 30
ensembles complets ([scripts/run_gridsearch.py](../scripts/run_gridsearch.py)).

### 12.1 Espace de recherche et élagage

$$
\Theta = \underbrace{\{0,\, 0.5,\, 1\}}_{p_{\mathrm{scar}}}
\times \underbrace{\{(0.02, 0.08),\, (0.05, 0.15)\}}_{(\alpha_{\min}, \alpha_{\max})}
\times \underbrace{\{0.3,\, 0.5\}}_{p}
\times \underbrace{\{0.5,\, 1,\, 2\}}_{w_{cp}}
\qquad |\Theta| = 36
$$

Élagage : si $p_{\mathrm{scar}} = 1$ (scars uniquement), $\alpha$ n'est
jamais lu — les configs qui ne diffèrent que par lui sont identiques ; on
n'en garde qu'une : $|\Theta'| = 30$.

### 12.2 Procédure par config $\theta$ (≈ 18 min / 2080 Ti)

1. **Graine commune** : toutes les configs partagent seed, init et ordre
   des batchs — la comparaison isole les boutons CutPaste.
2. Classifieur : 6 epochs (vs 20), un checkpoint/epoch ; chaque checkpoint
   est réévalué pour la courbe $\mathrm{AUROC}_{cp}(\text{epoch})$
   (`gridsearch_epochs.png`) — le budget d'epochs se lit gratuitement, sans
   dimension de grille supplémentaire.
3. Décodeurs : 8 epochs (vs 25) — nécessaires aux signaux de reconstruction.
4. Cinq AUROC sur splits plafonnés (~5 000 images/côté) : $s_{cp}$,
   $s_{p95}(\mathrm{Biais})$, $\ell_3$, $E$, et leur fusion par rang
   ($\mathrm{U}_{\mathrm{ep}}$ exclue : avec $M{=}1$ elle est identiquement
   nulle).
5. **Sélection** :

$$
\theta^\star = \arg\max_{\theta \in \Theta'}\;
\mathrm{AUROC}\Big( F_\theta \Big),
\qquad F_\theta = \text{fusion par rang de } \{s_{cp},\, s_{p95},\, \ell_3,\, E\}
$$

**Interprétation.** On optimise le *score final du projet* (la fusion), pas
une métrique isolée — c'est ce qui protège les autres signaux : une
sélection sur $s_{cp}$ seul aurait pu choisir une config qui détruit
l'énergie sans qu'on le voie.

**Hypothèse des schedules courts** : le *classement* des configs à 6+8
epochs prédit celui à 20+25. On ne lit jamais les valeurs absolues d'un
schedule court comme finales — seulement l'ordre.

### 12.3 Résultats (job 6781780)

$$
\theta^\star : \quad p_{\mathrm{scar}} = 0,\quad \alpha \in (0.05,\, 0.15),\quad p = 0.5,\quad w_{cp} = 0.5
\qquad \Rightarrow \qquad \mathrm{AUROC}(F_{\theta^\star}) = 0.827
$$

| Enseignement | Preuve dans la grille |
| --- | --- |
| Gros patchs sans scar $\gg$ scars | rangs 1-2-3-5 tous à $p_{\mathrm{scar}}{=}0$ ; scar-only dernier (0.659) |
| $w_{cp} = 2$ nuit | les configs $w_{cp}{=}2$ peuplent le bas du tableau |
| $s_{p95}$ insensible à CutPaste ($\approx 0.64$ partout) | c'est le 128 px qui devra l'aider, pas l'augmentation |
| L'énergie dépend fortement du pretext | $E$ varie de $0.41$ à $0.80$ selon $\theta$ |
| Le supervisé ne souffre pas | gender\_acc $\ge 0.978$ partout |

Et $0.827$ avec **un seul** modèle court à 64 px dépasse déjà les $0.810$
de l'ensemble complet de 10 — d'où l'objectif $0.85$–$0.90$ du run final.

---

## 13. Historique des résultats

Splits complets sauf mention contraire :

| Étape | Score | AUROC |
| --- | --- | --- |
| Baseline historique | $s_{p95}(\mathrm{Biais})$ (§4) | 0.638 |
| + incertitude | $\mathrm{U}_{\mathrm{ep}}^{\mathrm{comb}}$ (§5) | 0.740 |
| + localisation | z-score + patch-max (§7) | 0.686 |
| + espace features | $\ell_3$ (§8) | 0.779 |
| + fusion | rang, 4 signaux (§9) | **0.803** |
| + supervision | logistique calibrée (§10) | **0.810** |
| + CutPaste (grid, 1 modèle court, sous-échantillon) | $F_{\theta^\star}$ (§12) | **0.827** |
| ensemble $8 \times 128$ px + CutPaste (à lancer) | fusion complète | objectif 0.85–0.90 |

---

## 14. Carte du code

| Mesure | Module | Script d'évaluation |
| --- | --- | --- |
| AUROC / ROC (§2) | `lrad/evaluate.py` (`_auroc_entry`, `ood_auroc`) | tous |
| MSP, entropies (§3) | `lrad/evaluate.py` (`collect_predictions`) | `run_celeba.py` |
| Risque/Biais/Variance (§4) | `lrad/ensemble.py` (`decomposition_maps`) | `run_ensemble.py` |
| Incertitudes (§5) | `lrad/ensemble.py` (`_uncertainty_scores`) | `run_ensemble.py` |
| z-score + patch-max (§7) | `lrad/localized.py` | `run_localized.py` |
| locfre (§8) | `lrad/feature_error.py` | `run_fused.py` |
| énergie (§6), $s_{cp}$ (§11) | `lrad/fusion.py` (`collect_fusion_signals`) | `run_fused.py` |
| fusion rang (§9) / supervisée (§10) | `lrad/fusion.py` | `run_fused.py --supervised` |
| CutPaste (§11) | `lrad/cutpaste.py` + `lrad/train.py` | (entraînement) |
| grid search (§12) | — | `run_gridsearch.py` |

Jobs OAR : `oar_run_gridsearch.sh` (24 h), `oar_run_cutpaste128.sh` (48 h,
gruss), `oar_run_fused.sh` (4 h, éval seule sur checkpoints existants).
