"""Build the paper overview from vector artwork and CARD acquisition photos."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "glove_acquisition.png"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "pdf.fonttype": 42,
})
fig = plt.figure(figsize=(7.16, 3.2))
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(0, 45))
ax.set_axis_off()

ink = "#193349"
blue = "#2c648c"
teal = "#167567"
red = "#973a3e"
gray = "#627080"


def label(x, y, value, size=7.4, color=ink, weight="normal", align="center"):
    ax.text(x, y, value, fontsize=size, color=color, weight=weight,
            ha=align, va="center", linespacing=1.28, zorder=5)


def box(x, y, w, h, value, edge=blue, fill="#edf5fb", size=7.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.65",
                               ec=edge, fc=fill, lw=.65, zorder=2))
    label(x + w / 2, y + h / 2, value, size, edge)


def line(points, color=blue, arrow=True, dotted=False):
    ax.plot(*zip(*points), color=color, lw=.85, ls="--" if dotted else "-", zorder=3)
    if arrow:
        tail, head = points[-2:]
        ax.add_patch(FancyArrowPatch(tail, head, arrowstyle="-|>", mutation_scale=7,
                                     lw=.75, linestyle="--" if dotted else "-",
                                     color=color, zorder=4))


label(1, 43, "(a) CART architecture and training", 9.4, weight="bold", align="left")
label(67, 43, "(b) Glove training data", 9.4, weight="bold", align="left")
ax.plot([65, 65], [1, 43.9], lw=.65, color="#cbd6de")

box(6, 36, 28, 4.8, "Glove observations (t + 1 to t + 10)")
box(6, 29, 28, 5, "EMF–tactile cross-attention\n2 blocks · 4 heads · width 128")
box(6, 22, 28, 5, "State-conditioned pooling")
box(40, 22, 20, 5, "Current 26D state s(t)")
line([(20, 36), (20, 34)])
line([(20, 29), (20, 27)])
line([(40, 24.5), (34, 24.5)])

label(28, 19.9, "Independent finger and wrist decoders", 7.4, blue, "bold")
for i, name in enumerate(["Thumb", "Index", "Middle", "Ring", "Little", "Wrist"]):
    x = 6 + 7.3 * i
    box(x, 12.8, 6.7, 4.7, name + ("\n6 DoF" if name == "Wrist" else "\n4 joints"), size=6.1)
    line([(x + 3.35, 18.9), (x + 3.35, 17.5)])
    line([(x + 3.35, 12.8), (x + 3.35, 11.8)], arrow=False)
line([(9.35, 18.9), (45.85, 18.9)], arrow=False)
line([(8, 22), (8, 18.9), (9.35, 18.9)], arrow=False)
line([(9.35, 11.8), (45.85, 11.8)], arrow=False)
line([(24, 11.8), (24, 10.8)])
box(6, 6.6, 43.2, 4.2, "Residual update + smooth bounds → s(t + 1)")
line([(49.2, 8.7), (62.4, 8.7), (62.4, 24.5), (60, 24.5)])
label(57.6, 15.9, "State\nfeedback", 7, blue)
line([(23, 6.6), (23, 5.5)], teal)
box(6, .7, 43.2, 4.8, "Weighted rollout loss\nTracking · contact · smoothness · collision\nPad potentials, SDF, and self-collision models",
    teal, "#e9f6f1", 6.9)
line([(6, 3.1), (2, 3.1), (2, 24.5), (6, 24.5)], red, dotted=True)
ax.text(1, 11.7, "16-step BPTT", rotation=90, fontsize=7, ha="center", va="center", color=red)
box(38, 29, 22, 11.8, "Native-mesh supervision\n16 pad / 15 collision decoders\nValue / gradient / Hessian fitting\nValidation gate → freeze", gray, "#f1f4f6", 6.6)
line([(60, 30), (61, 30), (61, 3.1), (49.2, 3.1)], gray, dotted=True)
label(55.1, 5.25, "Frozen geometry", 6.2, gray)

with Image.open(ACQUISITION) as acquisition:
    ax.imshow(acquisition, extent=(67, 98.2, 29.4, 40.7), aspect="auto", zorder=1)
for panel, name in [(67, "Grasp"), (83, "Manipulation")]:
    label(panel + 7.6, 28, name, 7.2)

label(82.5, 25, "≈5 min · pinches, grasps, manipulation", 7.2, weight="bold")
box(67, 16, 31.2, 7, "Synchronized observations\n5 EMF fingertip poses\n460 taxels · activity and response", size=7.2)
line([(67, 19.5), (64.4, 19.5), (64.4, 41.7), (21, 41.7), (21, 40.8)])
line([(82.6, 16), (82.6, 14.2)], teal)
box(67, 6.8, 31.2, 7.4, "Targets from reconstructed hand mesh\nBone directions · tips · pad normals\nRegional contact anchors", teal, "#e9f6f1", 7)
line([(82.6, 6.8), (82.6, 1.35), (49.2, 1.35)], teal)

fig.savefig(HERE / "cart_overview.pdf")
