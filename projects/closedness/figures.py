"""Figures for the project page "On the (non-)closedness of the set of ReLU networks".

Generates:
  - ../../img/index_projects_cards/closedness.png : card picture for the front page
  - kinks.png                                     : straight vs. bent kinks (Lemma 3.8)
  - sobolev_loss.js                               : Sobolev training loss of u^eta, used by index.html

Run from anywhere:  python figures.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
relu = lambda t: np.maximum(t, 0)

# Parameters of the counterexample (Section 3 of the paper), as in Figures 4, 5 and 7.
LAM = 5.0
CS = (0.1, 0.2, 0.3)

# Same sequential ramp as the interactive plots of the page.
RAMP = LinearSegmentedColormap.from_list(
    "ramp", ["#152238", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"])
INK, MUTED, ORANGE = "#c3c2b7", "#898781", "#e8833f"
# Card picture: deep blue -> cyan -> mint, light enough at 0 for the flat base to show on the dark page.
CARD_CMAP = LinearSegmentedColormap.from_list(
    "card", ["#24418f", "#3367c9", "#2fa6d9", "#6fdbd2", "#e6fff4"])


def u_eta(x1, x2, eta):
    """Two-hidden-layer network u^eta in N_{4,6}; eta = 0 gives the limit f in N_{5,6}."""
    h1 = relu(x1 - 1 + 2 * eta * x2)
    h2 = relu(x1 - 2 + eta * x2)
    h3 = relu(3 - x1 + eta * x2)
    h4 = relu(4 - x1 + 2 * eta * x2)
    xr = (h1 - h2 - 1) / eta if eta > 0 else x2
    xl = (h4 - h3 - 1) / eta if eta > 0 else x2
    return sum(relu(xr - c - LAM * h3 - h4) + relu(xl - c - LAM * h2 - h1) for c in CS)


def u_eta_with_grad(x1, x2, eta):
    """Value and exact gradient of u^eta (the network is piecewise affine)."""
    one = np.ones_like(x1)
    a1, a2 = x1 - 1 + 2 * eta * x2, x1 - 2 + eta * x2
    a3, a4 = 3 - x1 + eta * x2, 4 - x1 + 2 * eta * x2
    h = [relu(a) for a in (a1, a2, a3, a4)]
    act = [(a > 0) * one for a in (a1, a2, a3, a4)]
    dh = [(act[0], 2 * eta * act[0]), (act[1], eta * act[1]), (-act[2], eta * act[2]), (-act[3], 2 * eta * act[3])]
    if eta > 0:
        xr, dxr = (h[0] - h[1] - 1) / eta, ((dh[0][0] - dh[1][0]) / eta, (dh[0][1] - dh[1][1]) / eta)
        xl, dxl = (h[3] - h[2] - 1) / eta, ((dh[3][0] - dh[2][0]) / eta, (dh[3][1] - dh[2][1]) / eta)
    else:
        xr = xl = x2
        dxr = dxl = (0 * one, one)
    val, g1, g2 = 0 * one, 0 * one, 0 * one
    for c in CS:
        for x, dx, i, j in ((xr, dxr, 2, 3), (xl, dxl, 1, 0)):
            arg = x - c - LAM * h[i] - h[j]
            on = arg > 0
            val += np.where(on, arg, 0)
            g1 += np.where(on, dx[0] - LAM * dh[i][0] - dh[j][0], 0)
            g2 += np.where(on, dx[1] - LAM * dh[i][1] - dh[j][1], 0)
    return val, g1, g2


# ---------------------------------------------------------------------------
# Card picture: the limit f as a faceted surface
# ---------------------------------------------------------------------------
def card_picture(path, cmap=CARD_CMAP):
    x1, x2 = np.meshgrid(np.linspace(0, 5, 900), np.linspace(0, 2, 500))
    z = u_eta(x1, x2, 0.0)
    zbox = 2.2

    # Facet shading: diffuse + specular, computed from the exact surface normals.
    gy, gx = np.gradient(z * zbox / 5.4, x2[:, 0], x1[0, :])
    n = np.stack([-gx, -gy, np.ones_like(z)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    light = np.array([-1.0, -1.5, 2.0]); light /= np.linalg.norm(light)
    view = np.array([-0.6, -1.0, 0.8]); view /= np.linalg.norm(view)
    half = (light + view) / np.linalg.norm(light + view)
    diff = np.clip(n @ light, 0, 1)
    spec = np.clip(n @ half, 0, 1) ** 40
    rgb = cmap(z / z.max())[..., :3] * (0.35 + 0.75 * diff[..., None]) + 0.45 * spec[..., None]

    fig = plt.figure(figsize=(6, 6), dpi=140)
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(x1, x2, z, facecolors=np.clip(rgb, 0, 1), rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((5, 2, zbox))
    ax.view_init(elev=38, azim=-40)
    ax.set_axis_off()
    ax.set_xlim(0, 5); ax.set_ylim(0, 2); ax.set_zlim(0, 5.4)
    fig.savefig(path, transparent=True)
    plt.close(fig)

    # Crop the transparent margins, then pad to a square like the other cards.
    im = Image.open(path)
    im = im.crop(im.getbbox())
    side = max(im.size) + 8
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    sq.save(path)


# ---------------------------------------------------------------------------
# Static figure: kinks of a shallow network vs. sigma(x2 - kappa(x1))
# ---------------------------------------------------------------------------
def kinks_figure(path):
    plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    t = np.linspace(-1, 1, 601)
    x1, x2 = np.meshgrid(t, t)

    shallow = relu(x1 + 0.4) + 0.8 * relu(x2 - 0.6 * x1 - 0.2) + 0.6 * relu(0.3 - x1 - x2)
    kappa = lambda s: 0.9 * relu(-s)
    deep = relu(x2 - kappa(x1))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), dpi=160)
    for ax, z, title in ((axes[0], shallow, "A shallow network"),
                         (axes[1], deep, r"$\sigma(x_2 - \kappa(x_1))$, $\kappa$ with a kink at $0$")):
        ax.imshow(z, extent=(-1, 1, -1, 1), origin="lower", cmap=RAMP, interpolation="bilinear")
        ax.set_title(title, pad=8)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$", rotation=0, labelpad=10)
        for s in ax.spines.values():
            s.set_visible(False)

    def kink(ax, xs, ys):
        ax.plot(xs, ys, color="black", lw=4.5, alpha=0.35, solid_capstyle="round")
        ax.plot(xs, ys, color=ORANGE, lw=2.2, solid_capstyle="round")

    # Shallow: every kink is a full line through the domain.
    kink(axes[0], [-0.4, -0.4], [-1, 1])
    kink(axes[0], t, 0.6 * t + 0.2)
    kink(axes[0], t, 0.3 - t)

    # Deep: three half-lines meeting at P = (0, 0).
    left = t[t <= 0]
    kink(axes[1], left, kappa(left))
    kink(axes[1], [0, 1], [0, 0])
    kink(axes[1], [0, 0], [0, 1])
    axes[1].plot([0], [0], "o", ms=8, mfc="white", mec="black", mew=1.2)
    axes[1].annotate(r"$P$", (0, 0), (0.12, -0.25), fontsize=13, color="white")

    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sobolev training loss E(u^eta) = ||u^eta - f||^2 + ||grad(u^eta - f)||^2 on Q
# ---------------------------------------------------------------------------
def eta_steps():
    """Must match etaFromStep() in index.html: 60 log-spaced values, then eta = 0."""
    return [0.4 * (0.001 / 0.4) ** (k / 59) for k in range(60)]


def sobolev_loss_table(path, nx=10000, ny=1000):
    # u^eta and f are symmetric w.r.t. x1 = 5/2: integrate over the right half only.
    xs = 2.5 + (np.arange(nx) + 0.5) * 2.5 / nx
    ys = (np.arange(ny) + 0.5) * 2.0 / ny
    cell = (2.5 / nx) * (2.0 / ny)
    rows = []
    for eta in eta_steps():
        sup, l2, h1 = 0.0, 0.0, 0.0
        for chunk in np.array_split(ys, 20):
            x1, x2 = np.meshgrid(xs, chunk)
            u, u1, u2 = u_eta_with_grad(x1, x2, eta)
            f, f1, f2 = u_eta_with_grad(x1, x2, 0.0)
            sup = max(sup, np.abs(u - f).max())
            l2 += 2 * cell * np.sum((u - f) ** 2)
            h1 += 2 * cell * np.sum((u1 - f1) ** 2 + (u2 - f2) ** 2)
        rows.append({"eta": eta, "sup": sup, "l2": l2, "grad": h1})
        print(f"eta={eta:.5f}  sup={sup:.4f}  L2^2={l2:.5f}  grad^2={h1:.5f}")
    with open(path, "w") as fh:
        fh.write("// Generated by figures.py: Sobolev training loss of the networks u^eta.\n")
        fh.write("const SOBOLEV_LOSS = ")
        json.dump([{k: float(f"{v:.6g}") for k, v in r.items()} for r in rows], fh)
        fh.write(";\n")


if __name__ == "__main__":
    card_picture(os.path.join(HERE, "..", "..", "img", "index_projects_cards", "closedness.png"))
    kinks_figure(os.path.join(HERE, "kinks.png"))
    sobolev_loss_table(os.path.join(HERE, "sobolev_loss.js"))
