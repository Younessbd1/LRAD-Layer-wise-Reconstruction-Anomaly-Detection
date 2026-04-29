"""Génère le diagramme d'architecture LRAD — fond blanc, épuré."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

DPI = 200
FW, FH = 19.2, 10.8   # → 3840 × 2160 px

fig, ax = plt.subplots(figsize=(FW, FH), dpi=DPI)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_facecolor('#FAFBFC')
fig.patch.set_facecolor('#FAFBFC')
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')

# ── Palette ──────────────────────────────────────────────────────────────────
DATA   = '#1D4ED8'; DATA_L  = '#EFF6FF'; DATA_B  = '#93C5FD'
CLF    = '#6D28D9'; CLF_L   = '#F5F3FF'; CLF_B   = '#C4B5FD'
DEC    = '#0F766E'; DEC_L   = '#ECFEFF'; DEC_B   = '#67E8F9'
ERR    = '#B91C1C'; ERR_L   = '#FEF2F2'; ERR_B   = '#FCA5A5'
FUS    = '#1E3A8A'; FUS_L   = '#EFF6FF'; FUS_B   = '#93C5FD'
UQ_C   = '#C2410C'; UQ_L    = '#FFF7ED'; UQ_B    = '#FED7AA'
TRN    = '#166534'; TRN_L   = '#F0FDF4'; TRN_B   = '#86EFAC'
EVL    = '#9F1239'; EVL_L   = '#FFF1F2'; EVL_B   = '#FDA4AF'
OUT    = '#5B21B6'; OUT_L   = '#F5F3FF'; OUT_B   = '#C4B5FD'
ARROW  = '#64748B'
SEC    = '#F1F5F9'; SEC_B   = '#CBD5E1'
TXT_W  = '#FFFFFF'; TXT_D   = '#0F172A'; TXT_G   = '#64748B'

# ── Helpers ───────────────────────────────────────────────────────────────────
def blk(x, y, w, h, fc, ec, lw=1.4, rad=0.12, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=z, clip_on=False, alpha=alpha,
    ))

def t(x, y, s, fs=9, fw='bold', col=TXT_D, ha='center', va='center', z=5, **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight=fw,
            color=col, zorder=z, **kw)

def sub(x, y, s, fs=7, col=TXT_G, ha='center', va='center', z=5, **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight='normal',
            color=col, zorder=z, **kw)

def arr(x1, y1, x2, y2, col=ARROW, lw=1.6, hw=0.11, hl=0.14, z=3,
        cs='arc3,rad=0.0'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=f"->,head_width={hw},head_length={hl}",
                    color=col, lw=lw,
                    connectionstyle=cs,
                ), zorder=z)

def darr(x1, y1, x2, y2, col=UQ_B, lw=1.1):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.07,head_length=0.10",
                    color=col, lw=lw, linestyle='dashed',
                ), zorder=3)

def sec_bg(x, y, w, h, label='', z=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.20",
        facecolor=SEC, edgecolor=SEC_B, linewidth=0.8,
        zorder=z, clip_on=False, alpha=0.55,
    ))
    if label:
        ax.text(x + 0.22, y + h - 0.15, label,
                ha='left', va='top', fontsize=8, fontweight='bold',
                color='#94A3B8', zorder=z + 1)

def header_strip(x, y, w, h_strip, color, label, fs=9, rad=0.12, z=3):
    """Full-width colored header inside a box."""
    ax.add_patch(FancyBboxPatch(
        (x, y + h_strip - 0.001), w, 0.001,  # invisible: clip top corners
        boxstyle="square,pad=0", facecolor='none', edgecolor='none', zorder=z-1,
    ))
    # Draw a rectangle that shares top radius with parent
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h_strip,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        facecolor=color, edgecolor=color, linewidth=0,
        zorder=z, clip_on=False,
    ))
    t(x + w/2, y + h_strip/2, label, fs=fs, col=TXT_W, z=z+1)

# ════════════════════════════════════════════════════════════════════════════
#  TITRE
# ════════════════════════════════════════════════════════════════════════════
blk(0, FH - 0.70, FW, 0.70, '#0F172A', '#0F172A', lw=0, rad=0, z=2)
t(FW/2, FH - 0.30, "LRAD  —  Layer-wise Reconstruction Anomaly Detection",
  fs=15.5, col='#F8FAFC', fw='bold')
t(FW/2, FH - 0.55, "Détection d'anomalies one-class par reconstruction multi-échelle d'activations",
  fs=8.5, col='#94A3B8', fw='normal')

# ════════════════════════════════════════════════════════════════════════════
#  SECTION A — DONNÉES
# ════════════════════════════════════════════════════════════════════════════
sec_bg(0.3, 8.88, FW - 0.6, 1.10, label='①  DONNÉES')

BY = 9.02; BH = 0.82; HS = 0.30  # box y, height, header strip height

# Real-IAD ──────────────────────────────────────────────────────────────────
RX = 0.65; RW = 3.50
blk(RX, BY, RW, BH, DATA_L, DATA_B, lw=1.2)
header_strip(RX, BY + BH - HS, RW, HS, DATA, "Real-IAD", fs=9)
t(RX + RW/2, BY + 0.34, "30 catégories industrielles", fs=8, col=DATA, fw='semibold')
sub(RX + RW/2, BY + 0.14, "5 vues / pièce  ·  OK (normal)  ·  NG (défaut) + masques pixel")

# CelebA ─────────────────────────────────────────────────────────────────────
CX = 4.45; CW = 3.50
blk(CX, BY, CW, BH, DATA_L, DATA_B, lw=1.2)
header_strip(CX, BY + BH - HS, CW, HS, DATA, "CelebA", fs=9)
t(CX + CW/2, BY + 0.34, "202 599 images de visages RGB", fs=8, col=DATA, fw='semibold')
sub(CX + CW/2, BY + 0.14, "normal = attribut 0  ·  anomalie = attribut 1  (ex. Lunettes)")

# Arrows → DataLoader ─────────────────────────────────────────────────────
DLX = 8.30; DLW = 3.80
arr(RX + RW, BY + BH/2, DLX, BY + BH/2, col=DATA, lw=1.5)
arr(CX + CW, BY + BH/2, DLX, BY + BH/2, col=DATA, lw=1.5)

# DataLoader ──────────────────────────────────────────────────────────────
blk(DLX, BY, DLW, BH, DATA_L, DATA_B, lw=1.2)
header_strip(DLX, BY + BH - HS, DLW, HS, DATA, "DataLoader", fs=9)
t(DLX + DLW/2, BY + 0.34, "train  ·  val  ·  test_normal  ·  test_anomaly", fs=8, col=DATA, fw='semibold')
sub(DLX + DLW/2, BY + 0.14, "train_ratio=0.8  ·  val_ratio=0.1  ·  seed=42")

# Evaluation ──────────────────────────────────────────────────────────────
EVX = 16.45; EVW = 2.55
blk(EVX, BY, EVW, BH, EVL_L, EVL_B, lw=1.2)
header_strip(EVX, BY + BH - HS, EVW, HS, EVL, "Évaluation", fs=9)
t(EVX + EVW/2, BY + 0.34, "AUROC  ·  PR-AUC", fs=8, col=EVL, fw='semibold')
sub(EVX + EVW/2, BY + 0.14, "Courbe ROC  ·  Heatmaps  ·  histogrammes")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION B — ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════════════════════
sec_bg(0.3, 7.62, FW - 0.6, 1.14, label='②  ENTRAÎNEMENT  (deux phases)')

TY = 7.76; TH = 0.82

# Phase 1 ─────────────────────────────────────────────────────────────────
blk(0.65, TY, 5.60, TH, TRN_L, TRN_B, lw=1.2)
header_strip(0.65, TY + TH - HS, 5.60, HS, TRN, "Phase 1 — Classifieur (données normales)", fs=8.5)
t(0.65 + 2.80, TY + 0.34, "CrossEntropyLoss  +  Adam  →  geler tous les poids", fs=8, col=TRN, fw='semibold')
sub(0.65 + 2.80, TY + 0.14, "Supervisé  ·  rotation pretext (CelebA)  ·  early stopping")

# Phase 2 ─────────────────────────────────────────────────────────────────
blk(6.50, TY, 5.60, TH, TRN_L, TRN_B, lw=1.2)
header_strip(6.50, TY + TH - HS, 5.60, HS, TRN, "Phase 2 — Décodeurs (activations gelées)", fs=8.5)
t(6.50 + 2.80, TY + 0.34, "MSELoss  +  un Adam par décodeur", fs=8, col=TRN, fw='semibold')
sub(6.50 + 2.80, TY + 0.14, "Chaque décodeur entraîné indépendamment  ·  normal seulement")

# UQ Phase (training note) ─────────────────────────────────────────────────
blk(12.35, TY, 4.10, TH, UQ_L, UQ_B, lw=1.2)
header_strip(12.35, TY + TH - HS, 4.10, HS, UQ_C, "UQ — Entraînement (LRADModelUQ)", fs=8.5)
t(12.35 + 2.05, TY + 0.34, "MC Dropout : standard + dropout actif", fs=8, col=UQ_C, fw='semibold')
sub(12.35 + 2.05, TY + 0.14, "Ensemble : M membres indépendants (seeds distincts)")

# Arrows: DataLoader → training
arr(DLX + DLW/2, BY, DLX + DLW/2, TY + TH, col=DATA, lw=1.4,
    cs='arc3,rad=-0.18')
arr(DLX + DLW/2, BY, 6.50 + 2.80, TY + TH, col=DATA, lw=1.4,
    cs='arc3,rad=0.15')

# Training → pipeline (down arrows)
arr(0.65 + 2.80, TY, 0.65 + 2.80, TY - 0.05, col=TRN, lw=1.0)
arr(6.50 + 2.80, TY, 6.50 + 2.80, TY - 0.05, col=TRN, lw=1.0)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION C — PIPELINE D'INFÉRENCE
# ════════════════════════════════════════════════════════════════════════════
sec_bg(0.3, 2.55, FW - 0.6, 4.96, label="③  PIPELINE D'INFÉRENCE  (LRADModel)")

# ── Image d'entrée ───────────────────────────────────────────────────────
IX = 0.50; IY = 4.30; IW = 1.40; IH = 1.50
blk(IX, IY, IW, IH, '#F8FAFC', '#CBD5E1', lw=1.2, rad=0.14)
t(IX + IW/2, IY + IH/2 + 0.20, "Image", fs=9, col='#334155')
t(IX + IW/2, IY + IH/2, "entrée", fs=9, col='#334155')
sub(IX + IW/2, IY + 0.22, "(B, C, H, W)")

# Arrow: image → classifier
arr(IX + IW, IY + IH/2, 2.10, IY + IH/2, col=ARROW)

# ── CLASSIFIEUR GELÉ ─────────────────────────────────────────────────────
CFL_X = 2.10; CFL_Y = 2.72; CFL_W = 3.25; CFL_H = 4.66
blk(CFL_X, CFL_Y, CFL_W, CFL_H, CLF_L, CLF_B, lw=1.4, rad=0.16, z=2)
header_strip(CFL_X, CFL_Y + CFL_H - 0.38, CFL_W, 0.38, CLF,
             "Classifieur (gelé)", fs=9, rad=0.16)
sub(CFL_X + CFL_W/2, CFL_Y + CFL_H - 0.52,
    "CNNClassifier  ou  MLPClassifier", fs=7.2, col='#7C3AED')

# 4 CNN blocks inside classifier
BLK_H = 0.58; BLK_GAP = 0.12
# layout bottom-up: pool+fc at bottom, then blocks 3,2,1,0 above
pool_y = CFL_Y + 0.14
blk(CFL_X + 0.18, pool_y, CFL_W - 0.36, 0.48, '#EDE9FE', '#8B5CF6', lw=0.9, rad=0.10, z=3)
t(CFL_X + CFL_W/2, pool_y + 0.32, "Avg Pool  →  FC  →  Logits", fs=7.5, col='#5B21B6', z=4)
sub(CFL_X + CFL_W/2, pool_y + 0.13, "classification", col='#7C3AED', z=4)

clf_blk_bot = pool_y + 0.48 + BLK_GAP  # bottom of block 3
clf_blk_ys = []  # y of bottom edge for blocks 3,2,1,0
for i in range(4):
    by = clf_blk_bot + i * (BLK_H + BLK_GAP)
    clf_blk_ys.append(by)
    blk(CFL_X + 0.18, by, CFL_W - 0.36, BLK_H, '#EDE9FE', '#8B5CF6',
        lw=0.9, rad=0.10, z=3)
    t(CFL_X + CFL_W/2, by + BLK_H - 0.20, f"Block {3-i}", fs=8, col='#5B21B6', z=4)
    sub(CFL_X + CFL_W/2, by + 0.20, "Conv 3×3  ·  BN  ·  ReLU", col='#7C3AED', z=4)

# act mid-y: block 3-i at clf_blk_ys[i]
act_mids = [by + BLK_H/2 for by in clf_blk_ys]
# Block 0 is at top (clf_blk_ys[3]), its act_mid is highest y

# ── DÉCODEURS ────────────────────────────────────────────────────────────
DEC_X = 5.65; DEC_W = 2.55
sec_bg(DEC_X - 0.12, CFL_Y, DEC_W + 0.24, CFL_H, label='', z=1)
t(DEC_X + DEC_W/2, CFL_Y + CFL_H - 0.18, "DÉCODEURS", fs=8, col='#94A3B8', fw='bold', z=4)
sub(DEC_X + DEC_W/2, CFL_Y + CFL_H - 0.36, "reconstruction inverse", col='#94A3B8', z=4)

for i in range(4):
    dy = clf_blk_ys[i]
    blk(DEC_X, dy, DEC_W, BLK_H, DEC_L, DEC_B, lw=1.1, rad=0.10, z=3)
    t(DEC_X + DEC_W/2, dy + BLK_H - 0.20, f"Décodeur {3-i}", fs=8, col='#0F766E', z=4)
    sub(DEC_X + DEC_W/2, dy + 0.20, "ConvTranspose  →  (B, C, H, W)", col='#0E7490', z=4)
    # Arrow from classifier block to decoder
    arr(CFL_X + CFL_W, act_mids[i], DEC_X, dy + BLK_H/2,
        col='#8B5CF6', lw=1.2, hw=0.07, hl=0.10)
    sub(CFL_X + CFL_W + 0.15, act_mids[i] + 0.10, f"act_{3-i}", fs=6.2, col='#8B5CF6',
        ha='left', z=5)

# ── CARTES D'ERREUR ───────────────────────────────────────────────────────
ERR_X = 8.55; ERR_W = 2.45
sec_bg(ERR_X - 0.12, CFL_Y, ERR_W + 0.24, CFL_H, label='', z=1)
t(ERR_X + ERR_W/2, CFL_Y + CFL_H - 0.18, "CARTES D'ERREUR", fs=8, col='#94A3B8', fw='bold', z=4)
sub(ERR_X + ERR_W/2, CFL_Y + CFL_H - 0.36, "|recon − x|²  par couche", col='#94A3B8', z=4)

for i in range(4):
    dy = clf_blk_ys[i]
    blk(ERR_X, dy, ERR_W, BLK_H, ERR_L, ERR_B, lw=1.1, rad=0.10, z=3)
    t(ERR_X + ERR_W/2, dy + BLK_H - 0.20, f"Carte {3-i}", fs=8, col='#991B1B', z=4)
    sub(ERR_X + ERR_W/2, dy + 0.20, "(B, 1, H, W)  →  upsampled", col='#B91C1C', z=4)
    arr(DEC_X + DEC_W, dy + BLK_H/2, ERR_X, dy + BLK_H/2,
        col=DEC, lw=1.2, hw=0.07, hl=0.10)

# ── FUSION ────────────────────────────────────────────────────────────────
FX = 11.25; FY = 3.78; FW_ = 2.05; FH_ = 2.82
# arrows from error maps to fusion
for i in range(4):
    arr(ERR_X + ERR_W, clf_blk_ys[i] + BLK_H/2, FX, FY + FH_/2,
        col=ERR, lw=1.0, hw=0.06, hl=0.09)

blk(FX, FY, FW_, FH_, FUS_L, FUS_B, lw=1.4, rad=0.14, z=3)
header_strip(FX, FY + FH_ - 0.38, FW_, 0.38, FUS, "Fusion", fs=9, rad=0.14, z=4)
sub(FX + FW_/2, FY + FH_ - 0.52, "multi-échelle", col='#1E40AF', z=4)

fmodes = [("mean", "moyenne"), ("max", "max pixel"), ("weighted", "pondéré")]
for j, (mode, desc) in enumerate(fmodes):
    my = FY + FH_ - 0.80 - j * 0.68
    blk(FX + 0.15, my, FW_ - 0.30, 0.50, '#DBEAFE', '#3B82F6', lw=0.8, rad=0.08, z=4)
    t(FX + FW_/2, my + 0.33, f'"{mode}"', fs=8, col='#1E40AF', z=5)
    sub(FX + FW_/2, my + 0.14, desc, col='#3B82F6', z=5)

# ── CARTE FUSIONNÉE + SCORE ────────────────────────────────────────────────
OX = 13.55; OH = 1.38
OY = FY + FH_/2 - OH/2 + 0.35
arr(FX + FW_, FY + FH_/2, OX, OY + OH/2, col=FUS, lw=1.6)

blk(OX, OY, 2.2, OH, OUT_L, OUT_B, lw=1.4, rad=0.14, z=3)
header_strip(OX, OY + OH - 0.34, 2.2, 0.34, OUT, "Carte d'anomalie", fs=9, rad=0.14, z=4)
t(OX + 1.1, OY + 0.58, "Heatmap fusionnée", fs=8, col=OUT, fw='semibold', z=4)
sub(OX + 1.1, OY + 0.32, "(B, 1, H, W)", z=4)

# Score
SC_Y = OY - 0.82
blk(OX, SC_Y, 2.2, 0.62, UQ_L, UQ_B, lw=1.4, rad=0.12, z=3)
t(OX + 1.1, SC_Y + 0.40, "Score d'anomalie", fs=8.5, col='#92400E', z=4)
sub(OX + 1.1, SC_Y + 0.18, "max pixel par image  (B,)", z=4)
arr(OX + 1.1, OY, OX + 1.1, SC_Y + 0.62, col=OUT, lw=1.2)

# Arrow: anomaly output → evaluation
arr(OX + 2.2, OY + OH/2, EVX, BY + BH/2, col=EVL, lw=1.4, cs='arc3,rad=-0.25')

# ════════════════════════════════════════════════════════════════════════════
#  SECTION D — EXTENSION UQ
# ════════════════════════════════════════════════════════════════════════════
sec_bg(0.3, 0.35, FW - 0.6, 2.10, label='④  EXTENSION UQ  (LRADModelUQ)')

UY = 0.52; UH = 1.72

# MC Dropout ──────────────────────────────────────────────────────────────
UX1 = 0.60; UW1 = 5.50
blk(UX1, UY, UW1, UH, UQ_L, UQ_B, lw=1.3, rad=0.14)
header_strip(UX1, UY + UH - 0.36, UW1, 0.36, UQ_C, "MC Dropout", fs=9, rad=0.14)
t(UX1 + UW1/2, UY + UH - 0.62, "T = 20 passes stochastiques", fs=8.5, col='#92400E', fw='semibold', z=4)
t(UX1 + UW1/2, UY + 0.85, "Dropout actif à l'inférence", fs=8, col=TXT_D, fw='normal', z=4)
sub(UX1 + UW1/2, UY + 0.58, "Chaque passe donne une reconstruction différente", z=4)
sub(UX1 + UW1/2, UY + 0.34, "→  moyenne  +  variance pixel  =  incertitude épistémique", z=4)

# Deep Ensemble ──────────────────────────────────────────────────────────
UX2 = 6.40; UW2 = 5.50
blk(UX2, UY, UW2, UH, UQ_L, UQ_B, lw=1.3, rad=0.14)
header_strip(UX2, UY + UH - 0.36, UW2, 0.36, UQ_C, "Deep Ensemble", fs=9, rad=0.14)
t(UX2 + UW2/2, UY + UH - 0.62, "M = 5 décodeurs indépendants", fs=8.5, col='#92400E', fw='semibold', z=4)
t(UX2 + UW2/2, UY + 0.85, "Initialisations et graines différentes", fs=8, col=TXT_D, fw='normal', z=4)
sub(UX2 + UW2/2, UY + 0.58, "Chaque membre prédit une reconstruction distincte", z=4)
sub(UX2 + UW2/2, UY + 0.34, "→  désaccord entre membres  =  incertitude", z=4)

# Score combiné ──────────────────────────────────────────────────────────
UX3 = 12.20; UW3 = 6.70
blk(UX3, UY, UW3, UH, '#F5F3FF', CLF_B, lw=1.3, rad=0.14)
header_strip(UX3, UY + UH - 0.36, UW3, 0.36, CLF, "Score Combiné", fs=9, rad=0.14)
t(UX3 + UW3/2, UY + UH - 0.65,
  "combiné  =  erreur  ×  ( 1  +  incertitude normalisée )",
  fs=9.5, col='#5B21B6', fw='bold', style='italic', z=4)
t(UX3 + UW3/2, UY + 0.82, "Haute erreur + haute incertitude  →  zone très suspecte", fs=8, col=TXT_D, fw='normal', z=4)
sub(UX3 + UW3/2, UY + 0.55, "Haute erreur + faible incertitude  →  anomalie confirmée", z=4)
sub(UX3 + UW3/2, UY + 0.30, "3 scores distincts : erreur  ·  incertitude  ·  combiné  →  3 AUROC", z=4)

# Dashed arrows: pipeline → UQ
darr(DEC_X + DEC_W/2, CFL_Y, UX1 + UW1/2, UY + UH, col=UQ_B)
darr(DEC_X + DEC_W/2, CFL_Y, UX2 + UW2/2, UY + UH, col=UQ_B)

# ════════════════════════════════════════════════════════════════════════════
#  LÉGENDE
# ════════════════════════════════════════════════════════════════════════════
LX = 16.38; LY = 3.00; LW = 2.62; LH = 4.45
blk(LX, LY, LW, LH, 'white', SEC_B, lw=1.0, rad=0.16)
t(LX + LW/2, LY + LH - 0.26, "LÉGENDE", fs=8.5, col='#334155')

legend = [
    (DATA,  'Données'),
    (TRN,   'Entraînement'),
    (CLF,   'Classifieur'),
    (DEC,   'Décodeur'),
    (ERR,   'Carte d\'erreur'),
    (FUS,   'Fusion'),
    (OUT,   'Sortie'),
    (UQ_C,  'UQ'),
    (EVL,   'Évaluation'),
]
for k, (col, name) in enumerate(legend):
    ley = LY + LH - 0.60 - k * 0.40
    ax.add_patch(FancyBboxPatch(
        (LX + 0.22, ley - 0.09), 0.32, 0.28,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=col, edgecolor='none', zorder=6,
    ))
    ax.text(LX + 0.70, ley + 0.05, name, ha='left', va='center',
            fontsize=8, color=TXT_D, zorder=7)

# ════════════════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════════════════
OUT_PATH = '/home/ybahaddo/lrad/docs/architecture_4k.png'
fig.savefig(OUT_PATH, dpi=DPI, facecolor='#FAFBFC')
print(f"Saved → {OUT_PATH}  ({int(FW*DPI)} × {int(FH*DPI)} px)")
plt.close(fig)
