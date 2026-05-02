"""Diagramme LRAD — Entraînement + Pipeline uniquement, fond blanc, 4K."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

DPI = 200
FW, FH = 19.2, 10.8

fig, ax = plt.subplots(figsize=(FW, FH), dpi=DPI)
ax.set_facecolor('#FAFBFC')
fig.patch.set_facecolor('#FAFBFC')
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')

# ── Palette ──────────────────────────────────────────────────────────────────
CLF   = '#6D28D9'; CLF_L = '#F5F3FF'; CLF_B = '#C4B5FD'
DEC   = '#0F766E'; DEC_L = '#ECFEFF'; DEC_B = '#67E8F9'
ERR   = '#B91C1C'; ERR_L = '#FEF2F2'; ERR_B = '#FCA5A5'
FUS   = '#1E3A8A'; FUS_L = '#EFF6FF'; FUS_B = '#93C5FD'
TRN   = '#166534'; TRN_L = '#F0FDF4'; TRN_B = '#86EFAC'
OUT   = '#5B21B6'; OUT_L = '#F5F3FF'; OUT_B = '#C4B5FD'
ARR   = '#64748B'
SEC   = '#F1F5F9'; SEC_B = '#CBD5E1'
WT    = '#FFFFFF'; DK    = '#0F172A'; GY    = '#64748B'

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(x, y, w, h, fc, ec, lw=1.4, rad=0.12, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=z, clip_on=False,
    ))

def hdr(x, y, w, hs, color, label, fs=10, rad=0.12, z=3):
    """Header strip at the TOP of a box."""
    ax.add_patch(FancyBboxPatch(
        (x, y + hs*0.4), w, hs*0.6,          # invisible filler to hide bottom rounding
        boxstyle="square,pad=0",
        facecolor=color, edgecolor='none', linewidth=0, zorder=z, clip_on=False,
    ))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, hs,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        facecolor=color, edgecolor=color, linewidth=0,
        zorder=z, clip_on=False,
    ))
    ax.text(x + w/2, y + hs/2, label,
            ha='center', va='center', fontsize=fs, fontweight='bold',
            color=WT, zorder=z+1)

def t(x, y, s, fs=9, fw='bold', col=DK, ha='center', va='center', z=5, **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight=fw, color=col, zorder=z, **kw)

def s(x, y, txt, fs=7.5, col=GY, ha='center', va='center', z=5, **kw):
    ax.text(x, y, txt, ha=ha, va=va, fontsize=fs, fontweight='normal', color=col, zorder=z, **kw)

def arr(x1, y1, x2, y2, col=ARR, lw=1.7, hw=0.11, hl=0.14, z=3, cs='arc3,rad=0.0'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=f"->,head_width={hw},head_length={hl}",
                    color=col, lw=lw, connectionstyle=cs,
                ), zorder=z)

def sec(x, y, w, h, label='', z=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.22",
        facecolor=SEC, edgecolor=SEC_B, linewidth=0.9,
        zorder=z, clip_on=False, alpha=0.5,
    ))
    if label:
        ax.text(x + 0.25, y + h - 0.18, label,
                ha='left', va='top', fontsize=9, fontweight='bold',
                color='#94A3B8', zorder=z+1)

# ════════════════════════════════════════════════════════════════════════════
#  TITRE
# ════════════════════════════════════════════════════════════════════════════
ax.add_patch(FancyBboxPatch((0, FH-0.72), FW, 0.72,
    boxstyle="square,pad=0", facecolor='#0F172A', edgecolor='none', zorder=2))
t(FW/2, FH-0.30, "LRAD  —  Entraînement & Pipeline d'Inférence",
  fs=16, col='#F8FAFC', fw='bold')
t(FW/2, FH-0.57, "Détection d'anomalies one-class par reconstruction multi-échelle d'activations",
  fs=9, col='#94A3B8', fw='normal')

# ════════════════════════════════════════════════════════════════════════════
#  ①  ENTRAÎNEMENT  (y = 7.65 → 9.98)
# ════════════════════════════════════════════════════════════════════════════
sec(0.28, 7.65, FW - 0.56, 2.25, label='①  ENTRAÎNEMENT')

TY = 7.80; TH = 1.95; HS = 0.40   # box y, height, header height

# ─── Phase 1 ─────────────────────────────────────────────────────────────
P1X = 0.55; P1W = 8.55
box(P1X, TY, P1W, TH, TRN_L, TRN_B)
hdr(P1X, TY + TH - HS, P1W, HS, TRN, "Phase 1 — Entraînement du Classifieur", fs=10.5)

SW = (P1W - 0.50) / 3 - 0.15   # sub-step width
SY = TY + 0.14; SH = TH - HS - 0.28

steps1 = [
    ("Données normales", "exemples OK uniquement",          TRN),
    ("CrossEntropyLoss", "classification supervisée",        TRN),
    ("Adam  →  Freeze",  "geler tous les poids\naprès entraînement", '#DC2626'),
]
for i, (title, desc, col) in enumerate(steps1):
    sx = P1X + 0.25 + i * (SW + 0.22)
    box(sx, SY, SW, SH, WT, TRN_B, lw=1.0, rad=0.10, z=3)
    t(sx + SW/2, SY + SH - 0.22, title, fs=9, col=col, z=4)
    for j, line in enumerate(desc.split('\n')):
        s(sx + SW/2, SY + SH/2 - 0.06 + (0.5-j)*0.24, line, fs=7.8, z=4)
    if i < 2:
        arr(sx + SW, SY + SH/2, sx + SW + 0.22, SY + SH/2, col=TRN, lw=1.5, hw=0.09, hl=0.12)

# ─── Phase 2 ─────────────────────────────────────────────────────────────
P2X = P1X + P1W + 0.28; P2W = FW - P2X - 0.45
box(P2X, TY, P2W, TH, TRN_L, TRN_B)
hdr(P2X, TY + TH - HS, P2W, HS, TRN, "Phase 2 — Entraînement des Décodeurs", fs=10.5)

SW2 = (P2W - 0.50) / 3 - 0.15
steps2 = [
    ("Activations gelées",  "classifieur.eval()\nrequires_grad=False",   CLF),
    ("MSELoss(recon_k, x)", "décodeur k reconstruit x\ndepuis act_k",      DEC),
    ("Adam indépendant",    "un optimiseur\npar décodeur",                TRN),
]
for i, (title, desc, col) in enumerate(steps2):
    sx = P2X + 0.25 + i * (SW2 + 0.22)
    box(sx, SY, SW2, SH, WT, TRN_B, lw=1.0, rad=0.10, z=3)
    t(sx + SW2/2, SY + SH - 0.22, title, fs=9, col=col, z=4)
    for j, line in enumerate(desc.split('\n')):
        s(sx + SW2/2, SY + SH/2 - 0.06 + (0.5-j)*0.24, line, fs=7.8, z=4)
    if i < 2:
        arr(sx + SW2, SY + SH/2, sx + SW2 + 0.22, SY + SH/2, col=TRN, lw=1.5, hw=0.09, hl=0.12)

# Arrows Phase → Pipeline
arr(P1X + P1W/2, TY, P1X + P1W/2, TY - 0.07, col=TRN, lw=1.2)
arr(P2X + P2W/2, TY, P2X + P2W/2, TY - 0.07, col=TRN, lw=1.2)

# ════════════════════════════════════════════════════════════════════════════
#  ②  PIPELINE D'INFÉRENCE  (y = 0.18 → 7.56)
# ════════════════════════════════════════════════════════════════════════════
sec(0.28, 0.18, FW - 0.56, 7.38, label="②  PIPELINE D'INFÉRENCE")

# ── Positions ────────────────────────────────────────────────────────────
IN_X  = 0.48;  IN_W  = 1.20
CLX   = 1.92;  CLW   = 3.60;  CLY = 0.32;  CLH = 7.03
DEC_X = 5.80;  DEC_W = 3.00
ERR_X = 9.10;  ERR_W = 2.70
FUX   = 12.12; FUW   = 2.05;  FUY = 1.52;  FUH = 4.68
OTX   = 14.50; OTW   = 2.50;  OTY = 4.30;  OTH = 1.62
SCX   = 14.50; SCW   = 2.50;  SCY = 3.22;  SCH = 0.82

BH = 1.10; BG = 0.22   # block height and gap

# ── Classifieur ──────────────────────────────────────────────────────────
box(CLX, CLY, CLW, CLH, CLF_L, CLF_B, lw=1.8, rad=0.18, z=2)
hdr(CLX, CLY + CLH - 0.46, CLW, 0.46, CLF, "Classifieur (gelé)", fs=11.5, rad=0.18, z=3)
s(CLX + CLW/2, CLY + CLH - 0.64, "CNNClassifier  ou  MLPClassifier", fs=8.5, col='#7C3AED', z=4)

# Pool + FC (bottom of classifier)
pool_y = CLY + 0.16
pool_h = 0.68
box(CLX+0.22, pool_y, CLW-0.44, pool_h, '#EDE9FE', '#8B5CF6', lw=1.0, rad=0.10, z=3)
t(CLX+CLW/2, pool_y+pool_h-0.24, "Avg Pool  →  FC  →  Logits", fs=9.5, col='#5B21B6', z=4)
s(CLX+CLW/2, pool_y+0.22, "sortie classification", fs=8, col='#7C3AED', z=4)

# 4 CNN blocks  (block 3 just above pool, block 0 near top)
bots = []   # bottom y of each inner block: index 0 = Block 3, index 3 = Block 0
base = pool_y + pool_h + BG
for i in range(4):
    bots.append(base + i*(BH+BG))

for i, by in enumerate(bots):
    label_num = 3 - i           # i=0 → Block 3 ... i=3 → Block 0
    box(CLX+0.22, by, CLW-0.44, BH, '#EDE9FE', '#8B5CF6', lw=1.0, rad=0.10, z=3)
    t(CLX+CLW/2, by+BH-0.28, f"Block {label_num}", fs=10.5, col='#5B21B6', z=4)
    s(CLX+CLW/2, by+BH/2, "Conv 3×3  ·  BN  ·  ReLU", fs=8.5, col='#7C3AED', z=4)
    s(CLX+CLW/2, by+0.22, "stride 2  →  downsampling ×2", fs=7.8, col='#8B5CF6', z=4)

act_mids = [by + BH/2 for by in bots]   # [Block 3 mid, ..., Block 0 mid]

# ── Image d'entrée ───────────────────────────────────────────────────────
# Centred vertically on Block 0 (bots[3])
IN_H = 1.30
in_y = bots[3] + BH/2 - IN_H/2
box(IN_X, in_y, IN_W, IN_H, '#F8FAFC', '#94A3B8', lw=1.4, rad=0.14, z=3)
t(IN_X+IN_W/2, in_y+IN_H/2+0.22, "Image", fs=10, col='#334155', z=4)
t(IN_X+IN_W/2, in_y+IN_H/2-0.02, "entrée", fs=10, col='#334155', z=4)
s(IN_X+IN_W/2, in_y+0.22, "(B, C, H, W)", fs=8, z=4)
arr(IN_X+IN_W, in_y+IN_H/2, CLX, bots[3]+BH/2, col=ARR, lw=1.8)

# ── Décodeurs ────────────────────────────────────────────────────────────
sec(DEC_X-0.12, CLY, DEC_W+0.24, CLH, z=1)
t(DEC_X+DEC_W/2, CLY+CLH-0.23, "DÉCODEURS", fs=10, col='#64748B', fw='bold', z=4)
s(DEC_X+DEC_W/2, CLY+CLH-0.46, "reconstruction inverse", fs=8, col='#94A3B8', z=4)

for i, by in enumerate(bots):
    label_num = 3 - i
    box(DEC_X, by, DEC_W, BH, DEC_L, DEC_B, lw=1.3, rad=0.12, z=3)
    t(DEC_X+DEC_W/2, by+BH-0.28, f"Décodeur {label_num}", fs=10.5, col='#0F766E', z=4)
    s(DEC_X+DEC_W/2, by+BH/2, "ConvTranspose2d  ×N  +  Sigmoid", fs=8.5, col='#0E7490', z=4)
    s(DEC_X+DEC_W/2, by+0.22, f"recon_{label_num}  →  (B, C, H, W)", fs=8, col='#115E59', z=4)
    arr(CLX+CLW, act_mids[i], DEC_X, by+BH/2, col='#8B5CF6', lw=1.5, hw=0.09, hl=0.12)
    mid_x = CLX + CLW + (DEC_X - CLX - CLW)/2
    t(mid_x, act_mids[i]+0.14, f"act_{label_num}", fs=8, col='#8B5CF6', fw='bold', z=5)

# ── Cartes d'erreur ───────────────────────────────────────────────────────
sec(ERR_X-0.12, CLY, ERR_W+0.24, CLH, z=1)
t(ERR_X+ERR_W/2, CLY+CLH-0.23, "CARTES D'ERREUR", fs=10, col='#64748B', fw='bold', z=4)
t(ERR_X+ERR_W/2, CLY+CLH-0.46, "| recon − x |²", fs=9, col='#94A3B8', fw='bold', z=4)

for i, by in enumerate(bots):
    label_num = 3 - i
    box(ERR_X, by, ERR_W, BH, ERR_L, ERR_B, lw=1.3, rad=0.12, z=3)
    t(ERR_X+ERR_W/2, by+BH-0.28, f"Carte {label_num}", fs=10.5, col='#991B1B', z=4)
    s(ERR_X+ERR_W/2, by+BH/2, "erreur pixel par pixel", fs=8.5, col='#B91C1C', z=4)
    s(ERR_X+ERR_W/2, by+0.22, "(B, 1, H, W)  →  upsampled  H×W", fs=8, col='#DC2626', z=4)
    arr(DEC_X+DEC_W, by+BH/2, ERR_X, by+BH/2, col=DEC, lw=1.5, hw=0.08, hl=0.11)

# ── Fusion ────────────────────────────────────────────────────────────────
for i, by in enumerate(bots):
    arr(ERR_X+ERR_W, by+BH/2, FUX, FUY+FUH/2, col=ERR, lw=1.1, hw=0.07, hl=0.10)

box(FUX, FUY, FUW, FUH, FUS_L, FUS_B, lw=1.8, rad=0.18, z=3)
hdr(FUX, FUY+FUH-0.46, FUW, 0.46, FUS, "Fusion", fs=12, rad=0.18, z=4)
s(FUX+FUW/2, FUY+FUH-0.66, "agrégation multi-échelle", fs=8.5, col='#1E40AF', z=4)

modes = [
    ('"mean"',     "moyenne des\n4 cartes"),
    ('"max"',      "maximum\npixel à pixel"),
    ('"weighted"', "pondéré\n1 / MSE"),
]
slot_h = (FUH - 0.75) / 3 - 0.10
for j, (mode, desc) in enumerate(modes):
    my = FUY + FUH - 0.82 - j*(slot_h+0.12)
    box(FUX+0.16, my, FUW-0.32, slot_h, '#DBEAFE', '#3B82F6', lw=0.9, rad=0.10, z=4)
    t(FUX+FUW/2, my+slot_h-0.26, mode, fs=10, col='#1E40AF', z=5)
    for k, line in enumerate(desc.split('\n')):
        s(FUX+FUW/2, my+slot_h/2-0.12 + (0.5-k)*0.22, line, fs=8, col='#3B82F6', z=5)

# ── Carte d'anomalie + Score ──────────────────────────────────────────────
arr(FUX+FUW, FUY+FUH/2, OTX, OTY+OTH/2, col=FUS, lw=2.0, hw=0.12, hl=0.16)

box(OTX, OTY, OTW, OTH, OUT_L, OUT_B, lw=1.8, rad=0.16, z=3)
hdr(OTX, OTY+OTH-0.42, OTW, 0.42, OUT, "Carte d'anomalie", fs=10.5, rad=0.16, z=4)
t(OTX+OTW/2, OTY+0.85, "Heatmap fusionnée", fs=9.5, col=OUT, fw='semibold', z=4)
s(OTX+OTW/2, OTY+0.52, "(B, 1, H, W)", fs=8.5, z=4)
s(OTX+OTW/2, OTY+0.28, "pixels chauds  =  zones anormales", fs=8, z=4)

arr(OTX+OTW/2, OTY, OTX+OTW/2, SCY+SCH, col=OUT, lw=1.6)

box(SCX, SCY, SCW, SCH, '#FFFBEB', '#FCD34D', lw=1.8, rad=0.14, z=3)
t(SCX+SCW/2, SCY+SCH-0.28, "Score d'anomalie", fs=10, col='#92400E', z=4)
s(SCX+SCW/2, SCY+0.24, "max pixel  →  scalaire par image  (B,)", fs=8.5, z=4)

# ── AUROC (petit badge à droite) ─────────────────────────────────────────
AUX = 17.35; AUY = SCY+0.03; AUW = 1.65; AUH = SCH-0.06
box(AUX, AUY, AUW, AUH, '#FFF1F2', '#FDA4AF', lw=1.3, rad=0.12, z=3)
t(AUX+AUW/2, AUY+AUH-0.26, "AUROC", fs=10, col='#9F1239', z=4)
s(AUX+AUW/2, AUY+0.22, "évaluation", fs=8, z=4)
arr(SCX+SCW, SCY+SCH/2, AUX, AUY+AUH/2, col='#F43F5E', lw=1.5)

# ── Légende ────────────────────────────────────────────────────────────────
LX = 17.18; LY = 4.48; LW = 1.82; LH = 2.68
box(LX, LY, LW, LH, WT, SEC_B, lw=1.0, rad=0.16)
t(LX+LW/2, LY+LH-0.26, "LÉGENDE", fs=9.5, col='#334155')
for k, (col, name) in enumerate([
    (TRN, 'Entraînement'),
    (CLF, 'Classifieur'),
    (DEC, 'Décodeur'),
    (ERR, "Erreur"),
    (FUS, 'Fusion'),
    (OUT, 'Sortie'),
]):
    ey = LY + LH - 0.55 - k*0.36
    ax.add_patch(FancyBboxPatch(
        (LX+0.20, ey-0.09), 0.28, 0.26,
        boxstyle="round,pad=0,rounding_size=0.04",
        facecolor=col, edgecolor='none', zorder=6,
    ))
    ax.text(LX+0.62, ey+0.04, name, ha='left', va='center',
            fontsize=9, color=DK, zorder=7)

# ── Save ────────────────────────────────────────────────────────────────────
OUT_FILE = '/home/ybahaddo/lrad/docs/architecture_4k.png'
fig.savefig(OUT_FILE, dpi=DPI, facecolor='#FAFBFC')
print(f"Saved → {OUT_FILE}  ({int(FW*DPI)} × {int(FH*DPI)} px)")
plt.close(fig)
