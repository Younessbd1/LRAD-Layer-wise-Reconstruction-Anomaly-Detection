# Guide pratique - Ton rôle et workflow Git/GitHub

## 1. Ton rôle concret

### Ce qui est prêt (le framework)
- Architecture complète : classifieurs (CNN, MLP) + décodeurs + pipeline LRAD
- Extensions UQ : MC Dropout, Deep Ensembles
- Évaluation : AUROC, heatmaps, distributions de scores, calibration
- Configs YAML : MNIST (MLP/CNN), CIFAR-10, MVTec AD
- Tests unitaires : 35 tests couvrant tous les composants
- Visualisation : grilles de heatmaps, overlay, comparaisons

### Ce que TU fais (la recherche)

#### Étape immédiate - Valider sur MNIST (1 jour)
```bash
python scripts/run_experiment.py --config configs/mnist_mlp.yaml
python scripts/run_experiment.py --config configs/mnist_cnn.yaml
python scripts/run_uq_experiment.py --config configs/mnist_cnn_uq.yaml
```
-> Vérifie que les heatmaps sont cohérentes, que l'AUROC est bon, que les anomalies (digits 4-9 + Fashion-MNIST) sont bien détectées.

#### Étape suivante - MVTec AD (1-2 semaines)
1. Télécharge MVTec AD sur ton serveur DCE Metz
2. Modifie `data_root` dans `configs/mvtec_cnn.yaml`
3. Lance pour chaque catégorie (bottle, cable, etc.)
4. Compare tes AUROC avec ceux de PatchCore (99.1%)

#### Étape recherche - Comparer UQ vs déterministe (2-3 semaines)
C'est LE cœur de ton stage. Tu as 3 expériences à mener :
- **LRAD déterministe** (baseline) : `run_experiment.py`
- **LRAD + MC Dropout** : `run_uq_experiment.py` avec `method: mc_dropout`
- **LRAD + Ensemble** : `run_uq_experiment.py` avec `method: ensemble`

Question de recherche : "Est-ce que l'incertitude épistémique améliore la détection/localisation d'anomalies par rapport à l'erreur de reconstruction seule ?"

#### Extensions possibles (pour aller plus loin)
- Ajouter des skip connections (U-Net) dans les décodeurs
- Remplacer le classifieur par un ResNet-18 pré-entraîné (comme PatchCore)
- Implémenter Evidential Deep Learning (Dirichlet) dans les décodeurs
- Comparer directement avec PatchCore sur MVTec AD

---

## 2. Faut-il ré-entraîner ?

**OUI, toujours.** Le framework est le code, pas les poids. Voici ce qui se passe quand tu lances :

```
run_experiment.py
├── Phase 1 : Entraîne le classifieur sur TON dataset -> sauvegarde classifier.pt
├── Phase 2 : Gèle le classifieur, entraîne les décodeurs -> sauvegarde decoder_*.pt
└── Évaluation : Génère heatmaps + AUROC -> sauvegarde dans outputs/
```

Chaque dataset/config produit ses propres poids. Tu ne réutilises jamais les poids d'un autre run. L'entraînement sur MNIST prend ~2 min sur CPU, ~30s sur GPU. Sur MVTec (224×224, RGB), compte ~10-15 min par catégorie sur GPU.

---

## 3. Setup Git/GitHub - étape par étape

### A. Créer le repo GitHub

1. Va sur https://github.com/new
2. Nom du repo : `lrad` (ou `lrad-anomaly-detection`)
3. Description : "Layer-wise Reconstruction Anomaly Detection with Uncertainty Quantification"
4. **NE COCHE PAS** "Initialize with README" (on a déjà le nôtre)
5. Clique "Create repository"

### B. Premier push depuis ton PC local

```bash
# 1. Extrais le projet
tar xzf lrad-project.tar.gz
cd lrad

# 2. Initialise Git
git init
git add .
git commit -m "Initial commit: LRAD framework with UQ extensions"

# 3. Connecte à GitHub (remplace par ton URL)
git remote add origin https://github.com/TON-USERNAME/lrad.git
git branch -M main
git push -u origin main
```

### C. Cloner sur ton serveur DCE Metz

```bash
# Sur le serveur DCE Metz
ssh ton-login@dce-metz.univ-lorraine.fr

# Clone
git clone https://github.com/TON-USERNAME/lrad.git
cd lrad
pip install -e .

# Lance les expériences
python scripts/run_experiment.py --config configs/mnist_cnn.yaml
```

### D. Workflow quotidien push/pull

```bash
# ══════════════════════════════════════
#  Sur DCE Metz (après avoir fait des modifs)
# ══════════════════════════════════════

# Vois ce qui a changé
git status
git diff

# Ajoute + commit
git add -A
git commit -m "feat: add skip connections to CNN decoder"

# Pousse vers GitHub
git push

# ══════════════════════════════════════
#  Sur ton PC local (pour récupérer)
# ══════════════════════════════════════

git pull

# ══════════════════════════════════════
#  Si tu travailles sur les deux machines
# ══════════════════════════════════════

# TOUJOURS pull avant de commencer à travailler
git pull

# ... travaille ...

# TOUJOURS commit + push en fin de session
git add -A
git commit -m "description de ce que tu as fait"
git push
```

### E. Bonnes pratiques de commit

```bash
# Bon : messages descriptifs
git commit -m "feat: MVTec bottle AUROC 0.94 with 4-layer CNN"
git commit -m "fix: decoder output_padding for 224x224 images"
git commit -m "exp: compare MC Dropout T=10,20,50 on MNIST"
git commit -m "docs: add MVTec results table to README"

# Mauvais : messages vagues
git commit -m "update"
git commit -m "fix stuff"
git commit -m "changes"
```

### F. .gitignore (déjà configuré)

Les fichiers suivants ne sont PAS envoyés sur GitHub :
- `data/` (trop gros, ~5 GB pour MVTec)
- `outputs/` (résultats locaux)
- `*.pt` (poids des modèles)
- `__pycache__/`

Si tu veux partager des résultats importants, crée un dossier `results/` et ajoute-le manuellement :
```bash
mkdir results
cp outputs/mvtec_bottle/summary.json results/
cp outputs/mvtec_bottle/roc_curves.png results/
git add results/
git commit -m "results: MVTec bottle final results"
```

### G. Si tu as des problèmes d'authentification GitHub

GitHub ne supporte plus les mots de passe pour push. Tu as deux options :

**Option 1 : Token personnel (plus simple)**
1. GitHub -> Settings -> Developer settings -> Personal access tokens -> Generate
2. Copie le token
3. Quand git demande un mot de passe, colle le token
4. Pour ne pas le retaper à chaque fois :
   ```bash
   git config --global credential.helper store
   ```

**Option 2 : Clé SSH (plus propre, recommandé pour DCE)**
```bash
# Sur DCE Metz
ssh-keygen -t ed25519 -C "ton-email@example.com"
cat ~/.ssh/id_ed25519.pub
# Copie le contenu -> GitHub -> Settings -> SSH Keys -> New SSH Key

# Utilise l'URL SSH au lieu de HTTPS
git remote set-url origin git@github.com:TON-USERNAME/lrad.git
```

---

## 4. Structure de travail recommandée

```
Semaine 1  │  Setup + MNIST validation
           │  git commit -m "validated MNIST MLP/CNN protocols"
           │
Semaine 2  │  MVTec AD integration + premiers résultats
           │  git commit -m "MVTec bottle AUROC=XX.X"
           │
Semaine 3  │  UQ experiments (MC Dropout vs Ensemble)
           │  git commit -m "UQ comparison: MC vs Ensemble on MVTec"
           │
Semaine 4  │  Analyse + visualisations pour le rapport
           │  git commit -m "final figures for report"
```

Chaque soir : `git add -A && git commit -m "..." && git push`
Chaque matin : `git pull`

C'est tout. Pas plus compliqué que ça.
