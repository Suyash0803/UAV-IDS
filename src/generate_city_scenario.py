"""
UAV IDS – City Scenario Attack Illustration
=============================================
Generates a top-down isometric-style city scene showing UAVs
patrolling an urban environment with attack scenarios visualised.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle, Wedge
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "results", "presentation_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)

# ── Canvas ─────────────────────────────────────────────────────────────────────
W, H = 28, 12
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W);  ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.axis('off')

# ── Colours ────────────────────────────────────────────────────────────────────
SKY_BG   = "#E8F4F8"
ROAD     = "#B0BEC5"
SIDEWALK = "#CFD8DC"
GRASS    = "#81C784"
GRASS2   = "#66BB6A"
PARK_BG  = "#A5D6A7"
WATER    = "#64B5F6"
BUILDING_COLORS = ["#EF9A9A","#FFCC80","#FFF59D","#B0BEC5",
                   "#CE93D8","#80DEEA","#BCAAA4","#80CBC4",
                   "#F48FB1","#A5D6A7","#90CAF9","#FFAB91"]
ROOF_COLORS     = ["#C62828","#E65100","#F9A825","#546E7A",
                   "#6A1B9A","#006064","#4E342E","#00695C",
                   "#880E4F","#1B5E20","#0D47A1","#BF360C"]
RED_ATTACK  = "#D32F2F"
BLUE_SIGNAL = "#1565C0"
ORANGE      = "#E65100"
WHITE       = "#FFFFFF"
DARK        = "#1A237E"

# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND – sky gradient feel
# ══════════════════════════════════════════════════════════════════════════════
bg = Rectangle((0, 0), W, H, color=SKY_BG, zorder=0)
ax.add_patch(bg)

# ══════════════════════════════════════════════════════════════════════════════
# ROAD NETWORK
# ══════════════════════════════════════════════════════════════════════════════
road_h = [3.8, 7.6]      # horizontal road y-centres
road_v = [5.6, 11.2, 16.8, 22.4]  # vertical road x-centres
road_w = 1.2

for y in road_h:
    ax.add_patch(Rectangle((0, y - road_w/2), W, road_w, color=ROAD, zorder=1))
    # centre line
    for x in np.arange(0, W, 1.0):
        ax.add_patch(Rectangle((x, y - 0.04), 0.6, 0.08, color=WHITE, alpha=0.6, zorder=2))

for x in road_v:
    ax.add_patch(Rectangle((x - road_w/2, 0), road_w, H, color=ROAD, zorder=1))
    for y in np.arange(0, H, 1.0):
        ax.add_patch(Rectangle((x - 0.04, y), 0.08, 0.6, color=WHITE, alpha=0.6, zorder=2))

# Sidewalks
for y in road_h:
    for side in [-1, 1]:
        ax.add_patch(Rectangle((0, y + side * road_w/2 - (0.15 if side == -1 else 0)),
                                W, 0.15, color=SIDEWALK, zorder=2))
for x in road_v:
    for side in [-1, 1]:
        ax.add_patch(Rectangle((x + side * road_w/2 - (0.15 if side == -1 else 0),
                                 0), 0.15, H, color=SIDEWALK, zorder=2))

# ══════════════════════════════════════════════════════════════════════════════
# CITY BLOCKS – helper
# ══════════════════════════════════════════════════════════════════════════════
def block_rect(col, row):
    """Return (x0,y0,w,h) of a city block given grid col/row."""
    xs = [0] + [v - road_w/2 for v in road_v] + [W]
    ys = [0] + [r - road_w/2 for r in road_h] + [H]
    x0 = xs[col * 2] + (0.15 if col > 0 else 0)
    x1 = xs[col * 2 + 1] - (0.15 if col < len(road_v) else 0)
    y0 = ys[row * 2] + (0.15 if row > 0 else 0)
    y1 = ys[row * 2 + 1] - (0.15 if row < len(road_h) else 0)
    return x0, y0, x1 - x0, y1 - y0

# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def draw_building(ax, cx, cy, w, h, color, roof_color, floors=3, zorder=5):
    # shadow
    ax.add_patch(FancyBboxPatch((cx - w/2 + 0.08, cy - h/2 - 0.08), w, h,
                                boxstyle="round,pad=0.02",
                                facecolor="#546E7A", alpha=0.25, zorder=zorder-1))
    # body
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="round,pad=0.03",
                                facecolor=color, edgecolor=roof_color,
                                linewidth=1.2, zorder=zorder))
    # roof
    ax.add_patch(FancyBboxPatch((cx - w/2, cy + h/2 - h*0.22), w, h*0.22,
                                boxstyle="round,pad=0.01",
                                facecolor=roof_color, edgecolor="none",
                                alpha=0.85, zorder=zorder+1))
    # windows
    ww, wh = 0.13, 0.10
    cols = max(1, int(w / 0.32))
    rows = max(1, floors)
    for r in range(rows):
        for c in range(cols):
            wx = cx - w/2 + 0.12 + c * (w - 0.12) / max(cols, 1)
            wy = cy - h/2 + 0.12 + r * (h * 0.72) / max(rows, 1)
            ax.add_patch(Rectangle((wx - ww/2, wy - wh/2), ww, wh,
                                   facecolor="#E3F2FD", edgecolor="#90CAF9",
                                   linewidth=0.5, zorder=zorder+2, alpha=0.9))


def draw_tree(ax, cx, cy, r=0.22, zorder=5):
    # trunk
    ax.add_patch(Rectangle((cx - 0.04, cy - r), 0.08, r * 0.5,
                            color="#5D4037", zorder=zorder))
    # canopy layers
    for i, (dr, dc) in enumerate([(r, GRASS2), (r*0.78, GRASS), (r*0.55, "#A5D6A7")]):
        ax.add_patch(Circle((cx, cy + dr * 0.3 * i * 0.3), dr,
                            color=dc, zorder=zorder + i, alpha=0.92))


def draw_park(ax, x0, y0, w, h, zorder=3):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0.05",
                                facecolor=PARK_BG, edgecolor=GRASS2,
                                linewidth=1.5, zorder=zorder))
    # pond
    ax.add_patch(mpatches.Ellipse((x0 + w*0.5, y0 + h*0.42),
                                   w*0.38, h*0.32,
                                   color=WATER, zorder=zorder+1, alpha=0.85))
    ax.add_patch(mpatches.Ellipse((x0 + w*0.5, y0 + h*0.42),
                                   w*0.26, h*0.20,
                                   color="#42A5F5", zorder=zorder+2, alpha=0.7))
    # paths
    for angle in [0, 90, 180, 270]:
        rad = np.radians(angle)
        px  = x0 + w*0.5 + np.cos(rad) * w*0.19
        py  = y0 + h*0.42 + np.sin(rad) * h*0.16
        ax.plot([x0 + w*0.5, px], [y0 + h*0.42, py],
                color=SIDEWALK, lw=1.8, zorder=zorder+2, alpha=0.8)


def draw_football_field(ax, x0, y0, w, h, zorder=3):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0.04",
                                facecolor="#388E3C", edgecolor=WHITE,
                                linewidth=1.5, zorder=zorder))
    # stripes
    for i in range(0, int(w / 0.4)):
        c = "#2E7D32" if i % 2 == 0 else "#388E3C"
        ax.add_patch(Rectangle((x0 + i*0.4, y0), 0.4, h,
                                facecolor=c, alpha=0.5, zorder=zorder+1))
    # lines
    ax.add_patch(Rectangle((x0+0.1, y0+0.1), w-0.2, h-0.2,
                            facecolor="none", edgecolor=WHITE,
                            linewidth=1.2, zorder=zorder+2))
    ax.add_patch(Rectangle((x0 + w*0.5 - 0.02, y0), 0.04, h,
                            facecolor=WHITE, zorder=zorder+2))
    ax.add_patch(Circle((x0 + w/2, y0 + h/2), h*0.18,
                        facecolor="none", edgecolor=WHITE,
                        linewidth=1.2, zorder=zorder+2))


def draw_stadium(ax, cx, cy, rx=1.1, ry=0.65, zorder=4):
    ax.add_patch(mpatches.Ellipse((cx, cy), rx*2, ry*2,
                                   facecolor="#B0BEC5", edgecolor="#546E7A",
                                   linewidth=2, zorder=zorder))
    ax.add_patch(mpatches.Ellipse((cx, cy), rx*1.5, ry*1.5,
                                   facecolor="#4CAF50", edgecolor=WHITE,
                                   linewidth=1.5, zorder=zorder+1))
    ax.add_patch(mpatches.Ellipse((cx, cy), rx*0.7, ry*0.5,
                                   facecolor="#388E3C", edgecolor=WHITE,
                                   linewidth=1, zorder=zorder+2))


def draw_car(ax, cx, cy, angle=0, color="#FFCA28", zorder=6):
    t = matplotlib.transforms.Affine2D().rotate_deg_around(cx, cy, angle)
    body = FancyBboxPatch((cx-0.18, cy-0.09), 0.36, 0.18,
                          boxstyle="round,pad=0.03",
                          facecolor=color, edgecolor="#37474F",
                          linewidth=0.8, zorder=zorder,
                          transform=t + ax.transData)
    ax.add_patch(body)
    # wheels
    for wx, wy in [(-0.12, -0.09), (0.08, -0.09),
                   (-0.12,  0.09), (0.08,  0.09)]:
        w_patch = Circle((cx + wx, cy + wy), 0.05,
                         color="#263238", zorder=zorder+1,
                         transform=t + ax.transData)
        ax.add_patch(w_patch)


def draw_bus(ax, cx, cy, color="#1976D2", zorder=6):
    ax.add_patch(FancyBboxPatch((cx-0.45, cy-0.12), 0.90, 0.24,
                                boxstyle="round,pad=0.03",
                                facecolor=color, edgecolor="#0D47A1",
                                linewidth=1, zorder=zorder))
    for i in range(4):
        ax.add_patch(Rectangle((cx - 0.35 + i*0.18, cy - 0.02),
                                0.12, 0.10,
                                facecolor="#E3F2FD", edgecolor="#90CAF9",
                                linewidth=0.5, zorder=zorder+1))


def draw_uav(ax, cx, cy, size=0.28, attacked=False, zorder=10):
    arm_color = "#212121"
    body_col  = "#F5F5F5" if not attacked else "#FFCDD2"
    border    = RED_ATTACK if attacked else "#212121"

    if attacked:
        box = FancyBboxPatch((cx - size*0.95, cy - size*0.95),
                             size*1.9, size*1.9,
                             boxstyle="round,pad=0.04",
                             facecolor=WHITE, edgecolor=RED_ATTACK,
                             linewidth=2.5, zorder=zorder-1, alpha=0.92)
        ax.add_patch(box)

    for ang in [45, 135, 225, 315]:
        rad = np.radians(ang)
        ex  = cx + np.cos(rad) * size * 0.75
        ey  = cy + np.sin(rad) * size * 0.75
        ax.plot([cx, ex], [cy, ey], color=arm_color, lw=2.5, zorder=zorder)
        # rotor
        ax.add_patch(Circle((ex, ey), size*0.28,
                            fill=False, edgecolor=arm_color,
                            lw=1.8, zorder=zorder, alpha=0.8))
        ax.add_patch(Circle((ex, ey), size*0.10,
                            color=arm_color, zorder=zorder+1, alpha=0.5))

    # body
    ax.add_patch(Circle((cx, cy), size*0.30,
                        color=body_col, edgecolor=border,
                        lw=2, zorder=zorder+1))
    ax.add_patch(Circle((cx, cy), size*0.12,
                        color="#37474F", zorder=zorder+2))
    # camera lens
    ax.add_patch(Circle((cx, cy - size*0.08), size*0.08,
                        color="#1A237E", zorder=zorder+3, alpha=0.8))


def draw_lightning(ax, cx, cy, size=0.28, color=ORANGE, zorder=12):
    pts = np.array([
        [cx,          cy + size],
        [cx - size*0.4, cy + size*0.1],
        [cx,          cy + size*0.15],
        [cx - size*0.5, cy - size],
        [cx + size*0.4, cy - size*0.1],
        [cx,          cy - size*0.15],
    ])
    ax.add_patch(Polygon(pts, closed=True,
                         facecolor=color, edgecolor=WHITE,
                         linewidth=0.8, zorder=zorder, alpha=0.95))


def draw_data_packet(ax, cx, cy, size=0.20, zorder=11):
    for i in range(3):
        ax.add_patch(FancyBboxPatch((cx - size, cy - size*0.6 + i*size*0.3),
                                    size*2, size*0.55,
                                    boxstyle="round,pad=0.02",
                                    facecolor=WHITE, edgecolor=BLUE_SIGNAL,
                                    linewidth=1.2, zorder=zorder + i, alpha=0.9))
        ax.plot([cx - size + 0.05, cx + size - 0.05],
                [cy - size*0.6 + i*size*0.3 + size*0.18,
                 cy - size*0.6 + i*size*0.3 + size*0.18],
                color=BLUE_SIGNAL, lw=1, alpha=0.6, zorder=zorder+3)


def draw_patrol_circle(ax, cx, cy, r, zorder=7):
    circle = plt.Circle((cx, cy), r, fill=False,
                         edgecolor="#212121", linestyle="--",
                         linewidth=1.8, zorder=zorder, alpha=0.7)
    ax.add_patch(circle)


def dashed_attack_arrow(ax, x1, y1, x2, y2, zorder=13):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=RED_ATTACK, lw=2.2,
                    linestyle="dashed",
                    connectionstyle="arc3,rad=0.25",
                    mutation_scale=18
                ), zorder=zorder)


def blue_signal_arrow(ax, x1, y1, x2, y2, zorder=12):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=BLUE_SIGNAL, lw=1.8,
                    connectionstyle="arc3,rad=-0.2",
                    mutation_scale=14
                ), zorder=zorder)


# ══════════════════════════════════════════════════════════════════════════════
# POPULATE CITY BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

# ── Block (0,2) top-left  ─────────────────────────────────────────────────────
bx, by, bw, bh = 0.15, 8.35, 5.0, 3.50
draw_building(ax, 1.6, 10.5, 1.2, 1.8, BUILDING_COLORS[0], ROOF_COLORS[0], floors=4)
draw_building(ax, 3.2, 10.2, 0.9, 1.5, BUILDING_COLORS[3], ROOF_COLORS[3], floors=3)
draw_building(ax, 4.5, 10.8, 1.0, 1.0, BUILDING_COLORS[7], ROOF_COLORS[7], floors=2)
for tx, ty in [(0.7,11.2),(2.8,11.5),(4.1,11.0),(1.0,9.0),(4.8,9.3)]:
    draw_tree(ax, tx, ty, r=0.20)

# ── Block (1,2) top-middle-left ───────────────────────────────────────────────
draw_park(ax, 6.2, 8.55, 4.6, 3.20)
for tx, ty in [(6.5,11.2),(10.4,11.3),(6.6,8.9),(10.3,8.8),(8.4,10.9),(8.4,9.0)]:
    draw_tree(ax, tx, ty, r=0.22)

# ── Block (2,2) top-middle ────────────────────────────────────────────────────
draw_building(ax, 13.1, 10.8, 1.8, 2.0, BUILDING_COLORS[4], ROOF_COLORS[4], floors=5)
draw_building(ax, 14.8, 10.2, 1.0, 1.2, BUILDING_COLORS[1], ROOF_COLORS[1], floors=3)
draw_building(ax, 13.5, 8.9,  1.0, 0.9, BUILDING_COLORS[9], ROOF_COLORS[9], floors=2)
for tx, ty in [(12.2,11.2),(15.8,11.0),(12.3,9.0),(15.5,8.9)]:
    draw_tree(ax, tx, ty, r=0.20)

# ── Block (3,2) top-right ─────────────────────────────────────────────────────
draw_football_field(ax, 17.5, 9.2, 3.5, 2.5)
for tx, ty in [(17.3,11.2),(21.3,11.1),(17.2,8.8),(21.2,9.0)]:
    draw_tree(ax, tx, ty, r=0.22)

# ── Block (4,2) far top-right ─────────────────────────────────────────────────
draw_stadium(ax, 25.5, 10.2, rx=1.3, ry=0.9)
for tx, ty in [(23.2,11.3),(23.3,8.9)]:
    draw_tree(ax, tx, ty, r=0.22)

# ── Block (0,1) middle-left ───────────────────────────────────────────────────
draw_building(ax, 1.2, 6.5, 0.9, 1.1, BUILDING_COLORS[2], ROOF_COLORS[2], floors=3)
draw_building(ax, 2.5, 5.5, 1.8, 1.8, BUILDING_COLORS[6], ROOF_COLORS[6], floors=5)  # tall tower
draw_building(ax, 4.2, 5.8, 0.8, 0.8, BUILDING_COLORS[10],ROOF_COLORS[10],floors=2)
for tx, ty in [(0.5,5.0),(4.9,6.9),(0.6,6.8),(5.0,5.2)]:
    draw_tree(ax, tx, ty, r=0.20)

# ── Block (1,1) middle ────────────────────────────────────────────────────────
draw_building(ax, 8.0, 6.0, 1.6, 2.2, BUILDING_COLORS[5], ROOF_COLORS[5], floors=5)  # glass tower
draw_building(ax, 9.8, 5.5, 0.9, 1.0, BUILDING_COLORS[11],ROOF_COLORS[11],floors=2)
for tx, ty in [(6.5,6.8),(10.5,6.8),(6.6,4.2),(10.4,4.3)]:
    draw_tree(ax, tx, ty, r=0.20)

# ── Block (2,1) middle-right ──────────────────────────────────────────────────
draw_building(ax, 13.0, 5.8, 1.4, 1.8, BUILDING_COLORS[8], ROOF_COLORS[8], floors=4)
draw_building(ax, 14.8, 6.2, 0.8, 1.0, BUILDING_COLORS[0], ROOF_COLORS[0], floors=2)
draw_building(ax, 13.5, 4.5, 1.0, 0.8, BUILDING_COLORS[3], ROOF_COLORS[3], floors=2)
for tx, ty in [(12.1,6.9),(15.8,4.5),(12.2,4.3)]:
    draw_tree(ax, tx, ty, r=0.20)

# ── Block (3,1) right ─────────────────────────────────────────────────────────
draw_building(ax, 19.0, 6.2, 1.8, 1.6, BUILDING_COLORS[2], ROOF_COLORS[2], floors=3)
draw_building(ax, 20.8, 5.5, 0.8, 1.0, BUILDING_COLORS[7], ROOF_COLORS[7], floors=2)
for tx, ty in [(17.3,6.8),(21.3,6.7),(17.4,4.3),(21.2,4.4)]:
    draw_tree(ax, tx, ty, r=0.20)

# ── Block (4,1) far right ─────────────────────────────────────────────────────
draw_building(ax, 24.5, 5.8, 1.0, 1.8, BUILDING_COLORS[9], ROOF_COLORS[9], floors=4)
draw_building(ax, 25.8, 5.2, 1.2, 1.0, BUILDING_COLORS[6], ROOF_COLORS[6], floors=3)
for tx, ty in [(23.2,6.8),(26.8,6.8),(23.3,4.3),(26.7,4.4)]:
    draw_tree(ax, tx, ty, r=0.20)

# ── Bottom blocks ─────────────────────────────────────────────────────────────
draw_building(ax,  1.5, 2.0, 1.0, 1.2, BUILDING_COLORS[1],  ROOF_COLORS[1],  floors=3)
draw_building(ax,  3.5, 2.2, 0.8, 0.8, BUILDING_COLORS[4],  ROOF_COLORS[4],  floors=2)
draw_park(ax, 6.2, 0.25, 4.6, 3.2)
for tx, ty in [(0.7,0.5),(5.0,0.6),(5.0,3.2),(0.6,3.2)]:
    draw_tree(ax, tx, ty, r=0.20)
draw_building(ax, 13.2, 2.0, 1.6, 2.0, BUILDING_COLORS[10], ROOF_COLORS[10], floors=4)
draw_building(ax, 19.2, 1.8, 1.4, 1.4, BUILDING_COLORS[5],  ROOF_COLORS[5],  floors=3)
draw_park(ax, 23.0, 0.25, 4.5, 3.3)
for tx, ty in [(12.2,3.3),(15.8,0.5),(17.3,3.2),(21.2,0.6),(21.3,3.2)]:
    draw_tree(ax, tx, ty, r=0.20)

# ── Cars & buses ──────────────────────────────────────────────────────────────
draw_car(ax, 3.5,  3.9,  0,   "#FFCA28")
draw_car(ax, 9.0,  3.9,  0,   "#EF5350")
draw_car(ax, 15.5, 3.8,  0,   "#66BB6A")
draw_car(ax, 21.5, 3.8,  0,   "#42A5F5")
draw_car(ax, 5.4,  7.7, 180,  "#AB47BC")
draw_bus(ax, 12.0, 7.7,       "#1976D2")
draw_car(ax, 19.0, 7.6,  0,   "#FFCA28")
draw_car(ax, 7.5,  7.7,  0,   "#78909C")

# ══════════════════════════════════════════════════════════════════════════════
# UAVs  (positions, attacked flag)
# ══════════════════════════════════════════════════════════════════════════════
uavs = [
    (2.5,  10.5, True),    # 0 top-left    – under attack
    (8.5,  10.2, True),    # 1 top-park    – under attack
    (14.5, 6.8,  True),    # 2 mid-right   – under attack
    (19.8, 10.5, False),   # 3 top-right   – normal
    (22.0, 6.0,  True),    # 4 right-mid   – under attack
    (7.5,  5.5,  False),   # 5 middle      – normal
    (13.5, 2.2,  True),    # 6 bottom-mid  – under attack
]

for cx, cy, att in uavs:
    draw_uav(ax, cx, cy, size=0.26, attacked=att)

# ══════════════════════════════════════════════════════════════════════════════
# PATROL CIRCLES
# ══════════════════════════════════════════════════════════════════════════════
draw_patrol_circle(ax, 2.5,  10.5, 1.0)
draw_patrol_circle(ax, 7.5,  5.5,  0.9)
draw_patrol_circle(ax, 19.8, 10.5, 1.0)
draw_patrol_circle(ax, 13.5, 2.2,  0.9)
draw_patrol_circle(ax, 22.0, 6.0,  0.9)

# ══════════════════════════════════════════════════════════════════════════════
# ATTACK ARROWS (red dashed)
# ══════════════════════════════════════════════════════════════════════════════
dashed_attack_arrow(ax, 2.5, 10.5,  8.5, 10.2)   # 0 → 1
dashed_attack_arrow(ax, 8.5, 10.2, 14.5,  6.8)   # 1 → 2
dashed_attack_arrow(ax, 14.5, 6.8, 22.0,  6.0)   # 2 → 4
dashed_attack_arrow(ax, 22.0, 6.0, 19.8, 10.5)   # 4 → 3 (spoofing)
dashed_attack_arrow(ax, 8.5, 10.2, 13.5,  2.2)   # 1 → 6

# ══════════════════════════════════════════════════════════════════════════════
# BLUE SIGNAL ARROWS (normal telemetry)
# ══════════════════════════════════════════════════════════════════════════════
blue_signal_arrow(ax, 7.5, 5.5, 2.5, 10.5)
blue_signal_arrow(ax, 7.5, 5.5, 19.8, 10.5)

# ══════════════════════════════════════════════════════════════════════════════
# LIGHTNING BOLTS (attack indicators)
# ══════════════════════════════════════════════════════════════════════════════
for lx, ly in [(2.8,9.6),(9.2,9.5),(15.0,6.1),(22.5,5.3),(13.8,1.5)]:
    draw_lightning(ax, lx, ly, size=0.22)

# ══════════════════════════════════════════════════════════════════════════════
# DATA PACKETS (telemetry icons)
# ══════════════════════════════════════════════════════════════════════════════
for px, py in [(1.8,9.7),(7.2,6.2),(20.5,9.8),(12.8,3.0),(21.2,5.2)]:
    draw_data_packet(ax, px, py, size=0.17)

# ══════════════════════════════════════════════════════════════════════════════
# ATTACK TYPE LABELS near UAVs
# ══════════════════════════════════════════════════════════════════════════════
attack_labels = [
    (2.5,  11.4,  "GPS Spoofing",       RED_ATTACK),
    (8.5,  11.2,  "Replay Attack",      RED_ATTACK),
    (14.5, 7.7,   "Command Injection",  RED_ATTACK),
    (22.0, 7.0,   "Sensor Anomaly",     RED_ATTACK),
    (13.5, 3.2,   "GPS Spoofing",       RED_ATTACK),
    (19.8, 11.5,  "Normal Flight ✓",    "#2E7D32"),
    (7.5,  6.5,   "Normal Flight ✓",    "#2E7D32"),
]
for lx, ly, txt, col in attack_labels:
    ax.text(lx, ly, txt, ha="center", va="center", fontsize=7.8,
            fontweight="bold", color=col, zorder=15,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=WHITE,
                      edgecolor=col, linewidth=1.2, alpha=0.92))

# ══════════════════════════════════════════════════════════════════════════════
# IDS DETECTION BOX  (centre overlay)
# ══════════════════════════════════════════════════════════════════════════════
ids_box = FancyBboxPatch((10.3, 4.55), 7.4, 2.6,
                          boxstyle="round,pad=0.12",
                          facecolor="#0D1B2A", edgecolor="#00E5FF",
                          linewidth=2.5, alpha=0.93, zorder=16)
ax.add_patch(ids_box)
ax.text(14.0, 6.75, "🛡  UAV IDS – Hybrid Detection Engine",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color="#00E5FF", zorder=17)
ax.text(14.0, 6.30, "Bidirectional LSTM  ⊕  Autoencoder  |  OR-Fusion",
        ha="center", va="center", fontsize=9, color=WHITE, alpha=0.85, zorder=17)

metrics = [
    ("KDD 1999",   "Acc 95.7%  AUC 0.986", "#2196F3"),
    ("CICIDS 2017","Acc 98.2%  AUC 0.999", "#4CAF50"),
    ("UAV Real",   "Acc 96.4%  AUC 0.986", "#FF5722"),
]
for i, (ds, val, col) in enumerate(metrics):
    bx2 = 10.7 + i * 2.4
    ax.add_patch(FancyBboxPatch((bx2, 4.70), 2.2, 1.30,
                                boxstyle="round,pad=0.06",
                                facecolor=col, edgecolor="none",
                                alpha=0.22, zorder=17))
    ax.text(bx2 + 1.1, 5.60, ds, ha="center", fontsize=8,
            fontweight="bold", color=col, zorder=18)
    ax.text(bx2 + 1.1, 5.22, val, ha="center", fontsize=7.5,
            color=WHITE, alpha=0.9, zorder=18)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (RED_ATTACK,    "solid",   "Attack Path (Red Arrow)"),
    (BLUE_SIGNAL,   "solid",   "Telemetry Signal (Blue Arrow)"),
    (ORANGE,        "solid",   "⚡ Attack Indicator"),
    (BLUE_SIGNAL,   "solid",   "📄 Data Packet"),
    ("#212121",     "dashed",  "UAV Patrol Zone"),
    (RED_ATTACK,    "solid",   "🔴 UAV Under Attack"),
    ("#2E7D32",     "solid",   "🟢 Normal UAV Flight"),
]
lg = FancyBboxPatch((0.15, 0.22), 5.0, 3.3,
                     boxstyle="round,pad=0.1",
                     facecolor="#0D1B2A", edgecolor="#00E5FF",
                     linewidth=1.8, alpha=0.92, zorder=19)
ax.add_patch(lg)
ax.text(2.65, 3.35, "LEGEND", ha="center", fontsize=9,
        fontweight="bold", color="#00E5FF", zorder=20)
for i, (col, ls, txt) in enumerate(legend_items):
    yy = 3.02 - i * 0.40
    ax.plot([0.35, 0.85], [yy, yy], color=col, lw=2,
            linestyle=ls, zorder=20)
    if ls == "solid":
        ax.annotate("", xy=(0.85, yy), xytext=(0.65, yy),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5),
                    zorder=20)
    ax.text(1.0, yy, txt, va="center", fontsize=8,
            color=WHITE, zorder=20)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(14.0, 11.70,
        "UAV Intrusion Detection System — Urban Threat Scenario",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color="#0D1B2A", zorder=20,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=WHITE,
                  edgecolor=RED_ATTACK, linewidth=2, alpha=0.92))
ax.text(14.0, 11.22,
        "Multi-UAV network patrolling city airspace  •  Red = Attack detected  •  Green = Safe",
        ha="center", va="center", fontsize=9.5, color="#37474F", zorder=20)

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib
path = os.path.join(OUTPUT_DIR, "UAV_IDS_City_Scenario.png")
fig.savefig(path, dpi=200, bbox_inches="tight",
            facecolor=SKY_BG, edgecolor="none")
plt.close(fig)
print(f"\n✅  Saved: {path}\n")
