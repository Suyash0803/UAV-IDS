"""
Real-World Visual Architecture Diagram for UAV IDS
====================================================
Generates a publication-quality, visually rich architecture image
showing the full Hybrid BiLSTM-Autoencoder IDS pipeline with
real-world iconography and styled components.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "presentation_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
BG          = "#0D1B2A"      # dark navy background
UAV_COL     = "#00E5FF"      # cyan  – UAV / input
PREPROC_COL = "#FFD600"      # amber – preprocessing
BILSTM_COL  = "#1565C0"      # deep blue – BiLSTM path
AE_COL      = "#6A1B9A"      # purple – Autoencoder path
FUSION_COL  = "#B71C1C"      # red   – Ensemble fusion
OUTPUT_COL  = "#2E7D32"      # green – Final output
FEATURE_COL = "#37474F"      # grey  – feature boxes
TEXT_LIGHT  = "#FFFFFF"
TEXT_DARK   = "#0D1B2A"
GRID_COL    = "#1E3A5F"

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    10,
    "figure.dpi":   180,
    "savefig.dpi":  220,
    "savefig.bbox": "tight",
})


# ══════════════════════════════════════════════════════════════════════════════
# Helper drawing functions
# ══════════════════════════════════════════════════════════════════════════════

def draw_box(ax, x, y, w, h, color, label, sublabel="",
             text_color=TEXT_LIGHT, radius=0.03, alpha=0.92, fontsize=10):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0.01,rounding_size={radius}",
                         linewidth=1.8, edgecolor=color,
                         facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(box)
    # subtle inner glow
    glow = FancyBboxPatch((x - w/2 + 0.005, y - h/2 + 0.005), w - 0.01, h - 0.01,
                          boxstyle=f"round,pad=0.008,rounding_size={radius*0.8}",
                          linewidth=0, facecolor="white", alpha=0.07, zorder=4)
    ax.add_patch(glow)
    ax.text(x, y + (0.015 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=5)
    if sublabel:
        ax.text(x, y - 0.028, sublabel,
                ha="center", va="center", fontsize=fontsize - 2.5,
                color=text_color, alpha=0.85, zorder=5, style="italic")


def arrow(ax, x1, y1, x2, y2, color="#FFFFFF", lw=1.8, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=6)


def draw_uav(ax, cx, cy, size=0.07):
    """Draw a simple drone top-view icon."""
    # body
    body = Circle((cx, cy), size * 0.35, color=UAV_COL, zorder=8, alpha=0.95)
    ax.add_patch(body)
    # 4 arms
    for angle in [45, 135, 225, 315]:
        rad = np.radians(angle)
        ex  = cx + np.cos(rad) * size * 0.65
        ey  = cy + np.sin(rad) * size * 0.65
        ax.plot([cx, ex], [cy, ey], color=UAV_COL, lw=2.5, zorder=7)
        # rotor circle
        rotor = Circle((ex, ey), size * 0.22,
                       fill=False, edgecolor=UAV_COL, lw=1.8, zorder=7, alpha=0.8)
        ax.add_patch(rotor)
    # centre dot
    dot = Circle((cx, cy), size * 0.10, color=TEXT_DARK, zorder=9)
    ax.add_patch(dot)
    ax.text(cx, cy - size * 1.1, "UAV Platform",
            ha="center", va="top", fontsize=8, color=UAV_COL, fontweight="bold", zorder=9)


def draw_signal_waves(ax, cx, cy, n=3, color=UAV_COL):
    for i in range(1, n + 1):
        r = 0.04 * i
        arc = Wedge((cx, cy), r, -60, 60,
                    fill=False, edgecolor=color, lw=1.2, alpha=0.5 - i * 0.1, zorder=7)
        ax.add_patch(arc)


def draw_neural_layer(ax, cx, cy, n_nodes=4, color=BILSTM_COL,
                      node_r=0.012, v_gap=0.045):
    """Draw a column of neural-net nodes."""
    total_h = (n_nodes - 1) * v_gap
    for i in range(n_nodes):
        ny = cy + total_h / 2 - i * v_gap
        c  = Circle((cx, ny), node_r, color=color, zorder=8, alpha=0.9)
        ax.add_patch(c)
    return cy + total_h / 2, cy - total_h / 2   # top, bottom y


def connect_layers(ax, cx1, top1, bot1, cx2, top2, bot2, color, n=4, alpha=0.25):
    """Draw all connections between two neuron columns."""
    v1 = np.linspace(top1, bot1, n)
    v2 = np.linspace(top2, bot2, n)
    for y1 in v1:
        for y2 in v2:
            ax.plot([cx1, cx2], [y1, y2], color=color, lw=0.5, alpha=alpha, zorder=6)


def draw_waveform(ax, x, y, width=0.18, height=0.04, color=PREPROC_COL):
    t   = np.linspace(0, 4 * np.pi, 200)
    sig = np.sin(t) * height / 2
    ax.plot(x + np.linspace(0, width, 200), y + sig,
            color=color, lw=1.4, alpha=0.85, zorder=7)


def draw_heatmap_icon(ax, x, y, size=0.06, cmap="Blues"):
    data = np.random.rand(4, 4)
    sub  = ax.inset_axes([x, y, size * 1.5, size * 1.5], transform=ax.transData)
    sub.imshow(data, cmap=cmap, aspect="auto")
    sub.axis("off")


# ══════════════════════════════════════════════════════════════════════════════
# Main figure
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(22, 15))
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── Background grid ────────────────────────────────────────────────────────────
for x in np.arange(0, 1.01, 0.05):
    ax.axvline(x, color=GRID_COL, lw=0.3, alpha=0.4)
for y in np.arange(0, 1.01, 0.05):
    ax.axhline(y, color=GRID_COL, lw=0.3, alpha=0.4)

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.965,
        "UAV Intrusion Detection System – Real-World Architecture",
        ha="center", va="center", fontsize=17, fontweight="bold",
        color=TEXT_LIGHT, zorder=10,
        path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
ax.text(0.5, 0.938,
        "Hybrid Bidirectional LSTM  ⊕  Autoencoder Ensemble  |  OR-Fusion Decision Engine",
        ha="center", va="center", fontsize=11, color=UAV_COL, alpha=0.9, zorder=10)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 – UAV + Telemetry Input
# ══════════════════════════════════════════════════════════════════════════════
draw_uav(ax, 0.12, 0.845, size=0.055)
draw_signal_waves(ax, 0.12 + 0.055 * 0.65, 0.845, n=3)

# Telemetry input box
draw_box(ax, 0.38, 0.845, 0.36, 0.065, UAV_COL,
         "UAV Telemetry Input",
         "9 Features  ×  10 Timesteps (Sliding Window)",
         text_color=TEXT_DARK, fontsize=11)

# Feature pills
features = ["GPS (lat, lon, alt)", "Velocity", "Pitch / Roll / Yaw",
            "Battery %", "Command ID"]
fx_start = 0.60
for i, feat in enumerate(features):
    fx = fx_start + (i % 3) * 0.125
    fy = 0.875 - (i // 3) * 0.038
    draw_box(ax, fx, fy, 0.11, 0.028, FEATURE_COL, feat,
             fontsize=7.5, radius=0.015, alpha=0.85)

arrow(ax, 0.12 + 0.055, 0.845, 0.195, 0.845, UAV_COL, lw=2)
arrow(ax, 0.56, 0.845, 0.595, 0.845, UAV_COL, lw=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 – Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
draw_box(ax, 0.38, 0.765, 0.36, 0.058, PREPROC_COL,
         "Data Preprocessing",
         "Min-Max Normalisation  →  Sliding Window T=10",
         text_color=TEXT_DARK, fontsize=11)

# Waveform decoration
draw_waveform(ax, 0.62, 0.762, width=0.12, height=0.032, color=TEXT_DARK)

arrow(ax, 0.38, 0.812, 0.38, 0.794, PREPROC_COL, lw=2)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 – Split label
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.38, 0.720, "↙  Path A : Temporal Classification",
        ha="center", va="center", fontsize=9.5, color=BILSTM_COL,
        fontweight="bold", zorder=9)
ax.text(0.38, 0.700, "Path B : Anomaly Detection  ↘",
        ha="center", va="center", fontsize=9.5, color=AE_COL,
        fontweight="bold", zorder=9)
arrow(ax, 0.38, 0.736, 0.38, 0.724, TEXT_LIGHT, lw=1.5)

# ── Diverging arrows ──────────────────────────────────────────────────────────
arrow(ax, 0.38, 0.695, 0.215, 0.650, BILSTM_COL, lw=2)
arrow(ax, 0.38, 0.695, 0.560, 0.650, AE_COL,     lw=2)

# ══════════════════════════════════════════════════════════════════════════════
# PATH A – BiLSTM Classifier  (left column)
# ══════════════════════════════════════════════════════════════════════════════
# BiLSTM-1
draw_box(ax, 0.195, 0.615, 0.26, 0.060, BILSTM_COL,
         "Bidirectional LSTM – Layer 1",
         "64 hidden units  |  Forward + Backward pass", fontsize=9.5)

# Neural-net node visualisation for BILSTM-1
t1, b1 = draw_neural_layer(ax, 0.07, 0.615, n_nodes=5, color=BILSTM_COL)
t2, b2 = draw_neural_layer(ax, 0.10, 0.615, n_nodes=5, color="#42A5F5")
connect_layers(ax, 0.07, t1, b1, 0.10, t2, b2, BILSTM_COL)

# BiLSTM-2
draw_box(ax, 0.195, 0.540, 0.26, 0.060, BILSTM_COL,
         "Bidirectional LSTM – Layer 2",
         "64 hidden units  |  Dropout 0.3", fontsize=9.5)
arrow(ax, 0.195, 0.585, 0.195, 0.570, BILSTM_COL, lw=2)

# Fully Connected
draw_box(ax, 0.195, 0.468, 0.26, 0.055, "#1976D2",
         "Fully Connected Layer",
         "64 → 2 neurons  |  Sigmoid activation", fontsize=9.5)
arrow(ax, 0.195, 0.510, 0.195, 0.496, BILSTM_COL, lw=2)

# BiLSTM Output
draw_box(ax, 0.195, 0.398, 0.26, 0.055, "#0288D1",
         "BiLSTM Output",
         "P(Attack) ∈ [0,1]  |  θ = 0.35", fontsize=9.5)
arrow(ax, 0.195, 0.440, 0.195, 0.426, BILSTM_COL, lw=2)

# Path A label banner
ax.text(0.195, 0.685, "PATH A  –  Temporal Classification",
        ha="center", va="center", fontsize=9, color=BILSTM_COL,
        fontweight="bold", alpha=0.9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=BILSTM_COL,
                  alpha=0.6, linewidth=1.2), zorder=9)

# ══════════════════════════════════════════════════════════════════════════════
# PATH B – Autoencoder  (right column)
# ══════════════════════════════════════════════════════════════════════════════
draw_box(ax, 0.565, 0.615, 0.26, 0.060, AE_COL,
         "Encoder",
         "9×10 input  →  Bottleneck latent space", fontsize=9.5)

# encoder funnel icon
ex_pts = np.array([[0.695, 0.648], [0.720, 0.648],
                    [0.715, 0.582], [0.700, 0.582]])
funnel = plt.Polygon(ex_pts, closed=True, color=AE_COL,
                     alpha=0.25, zorder=7)
ax.add_patch(funnel)

draw_box(ax, 0.565, 0.542, 0.20, 0.048, "#AB47BC",
         "Latent Space",
         "Compressed representation", fontsize=9)
arrow(ax, 0.565, 0.585, 0.565, 0.566, AE_COL, lw=2)

draw_box(ax, 0.565, 0.473, 0.26, 0.055, AE_COL,
         "Decoder",
         "Latent  →  Reconstructed input", fontsize=9.5)
arrow(ax, 0.565, 0.518, 0.565, 0.501, AE_COL, lw=2)

draw_box(ax, 0.565, 0.403, 0.26, 0.055, "#7B1FA2",
         "Reconstruction Error",
         "MSE(input, output)  |  τ = 0.000134", fontsize=9.5)
arrow(ax, 0.565, 0.446, 0.565, 0.431, AE_COL, lw=2)

draw_box(ax, 0.565, 0.335, 0.26, 0.055, "#4A148C",
         "Autoencoder Output",
         "Error > τ  →  Anomaly flagged", fontsize=9.5)
arrow(ax, 0.565, 0.375, 0.565, 0.363, AE_COL, lw=2)

ax.text(0.565, 0.685, "PATH B  –  Anomaly Detection",
        ha="center", va="center", fontsize=9, color=AE_COL,
        fontweight="bold", alpha=0.9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=AE_COL,
                  alpha=0.6, linewidth=1.2), zorder=9)

# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE FUSION
# ══════════════════════════════════════════════════════════════════════════════
draw_box(ax, 0.38, 0.278, 0.40, 0.068, FUSION_COL,
         "Ensemble Fusion  (OR Logic)",
         "Attack flagged if  BiLSTM = 1  OR  Autoencoder = 1",
         fontsize=11)

# Converging arrows from both paths
arrow(ax, 0.195, 0.370, 0.280, 0.300, BILSTM_COL, lw=2.2)
arrow(ax, 0.565, 0.307, 0.480, 0.296, AE_COL,     lw=2.2)

# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL FILTERING
# ══════════════════════════════════════════════════════════════════════════════
draw_box(ax, 0.38, 0.200, 0.40, 0.058, "#E65100",
         "Temporal Consistency Filter",
         "N=3 consecutive detections required to confirm alert",
         fontsize=10)
arrow(ax, 0.38, 0.244, 0.38, 0.229, FUSION_COL, lw=2)

# ══════════════════════════════════════════════════════════════════════════════
# FINAL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
draw_box(ax, 0.38, 0.123, 0.40, 0.062, OUTPUT_COL,
         "Final Prediction",
         "🟢  Normal  (0)          🔴  Attack  (1)",
         fontsize=11)
arrow(ax, 0.38, 0.171, 0.38, 0.155, OUTPUT_COL, lw=2.5)

# ══════════════════════════════════════════════════════════════════════════════
# ALERT OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
# Normal
draw_box(ax, 0.20, 0.058, 0.22, 0.052, OUTPUT_COL,
         "✅  SAFE FLIGHT",
         "Continue mission", fontsize=9.5)
# Attack
draw_box(ax, 0.56, 0.058, 0.22, 0.052, "#C62828",
         "🚨  INTRUSION ALERT",
         "Trigger countermeasure", fontsize=9.5)

arrow(ax, 0.285, 0.092, 0.31,  0.092, OUTPUT_COL,  lw=1.5)
arrow(ax, 0.475, 0.092, 0.450, 0.092, "#C62828",   lw=1.5)
arrow(ax, 0.38,  0.092, 0.31,  0.084, OUTPUT_COL,  lw=1.8)
arrow(ax, 0.38,  0.092, 0.45,  0.084, "#C62828",   lw=1.8)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND BOX
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (UAV_COL,     "UAV Input / Telemetry"),
    (PREPROC_COL, "Preprocessing"),
    (BILSTM_COL,  "BiLSTM Classifier Path"),
    (AE_COL,      "Autoencoder Path"),
    (FUSION_COL,  "Ensemble Fusion"),
    ("#E65100",   "Temporal Filtering"),
    (OUTPUT_COL,  "Final Decision"),
]
lx, ly = 0.755, 0.760
draw_box(ax, lx + 0.085, ly - 0.115, 0.235, 0.265,
         "#1E2D3D", "", radius=0.02, alpha=0.85)
ax.text(lx + 0.085, ly + 0.010, "LEGEND",
        ha="center", fontsize=9, fontweight="bold", color=TEXT_LIGHT, zorder=10)
for i, (col, label) in enumerate(legend_items):
    yy = ly - 0.030 - i * 0.034
    rect = FancyBboxPatch((lx, yy - 0.009), 0.028, 0.018,
                          boxstyle="round,pad=0.002", facecolor=col,
                          edgecolor="none", alpha=0.9, zorder=10)
    ax.add_patch(rect)
    ax.text(lx + 0.036, yy, label,
            va="center", fontsize=8, color=TEXT_LIGHT, zorder=10)

# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE STATS BOX
# ══════════════════════════════════════════════════════════════════════════════
stats = [
    ("KDD Cup 1999",  "Acc 95.73%  |  AUC 0.9859"),
    ("CICIDS 2017",   "Acc 98.24%  |  AUC 0.9985"),
    ("UAV Realistic", "Acc 96.43%  |  AUC 0.9858"),
]
sx, sy = 0.755, 0.420
draw_box(ax, sx + 0.085, sy - 0.075, 0.235, 0.175,
         "#1E2D3D", "", radius=0.02, alpha=0.85)
ax.text(sx + 0.085, sy + 0.005, "MODEL PERFORMANCE",
        ha="center", fontsize=9, fontweight="bold", color=TEXT_LIGHT, zorder=10)
stat_cols = [UAV_COL, "#4CAF50", "#FF5722"]
for i, (ds, val) in enumerate(stats):
    yy = sy - 0.028 - i * 0.048
    ax.text(sx + 0.010, yy + 0.010, ds,
            fontsize=8.2, fontweight="bold", color=stat_cols[i], zorder=10)
    ax.text(sx + 0.010, yy - 0.008, val,
            fontsize=7.8, color=TEXT_LIGHT, alpha=0.85, zorder=10)

# ══════════════════════════════════════════════════════════════════════════════
# TECH STACK BADGE
# ══════════════════════════════════════════════════════════════════════════════
tech = ["Python 3.x", "PyTorch", "BiLSTM", "Autoencoder",
        "Scikit-learn", "NumPy / Pandas"]
tx, ty = 0.755, 0.225
draw_box(ax, tx + 0.085, ty - 0.055, 0.235, 0.135,
         "#1E2D3D", "", radius=0.02, alpha=0.85)
ax.text(tx + 0.085, ty + 0.010, "TECH STACK",
        ha="center", fontsize=9, fontweight="bold", color=TEXT_LIGHT, zorder=10)
for i, t in enumerate(tech):
    bx = tx + 0.015 + (i % 2) * 0.115
    by = ty - 0.020 - (i // 2) * 0.036
    draw_box(ax, bx + 0.045, by, 0.10, 0.025,
             "#263238", t, fontsize=7.5, radius=0.012, alpha=0.9)

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.5, 0.012,
        "Hybrid BiLSTM–Autoencoder IDS for UAV Security  •  Bidirectional LSTM + Reconstruction-Error Anomaly Detection  •  OR-Fusion Ensemble",
        ha="center", va="center", fontsize=8, color=TEXT_LIGHT, alpha=0.5, zorder=9)

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
path = os.path.join(OUTPUT_DIR, "UAV_IDS_Architecture_Visual.png")
fig.savefig(path, facecolor=BG, edgecolor="none")
plt.close(fig)
print(f"\n✅  Saved: {path}\n")
