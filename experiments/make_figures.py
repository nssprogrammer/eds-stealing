"""Regenerate the figures used in the note.  Run:  python experiments/make_figures.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eds_stealing import OneLayerModel, TwoLayerModel, recover_dimension, noise_sweep, intrinsic_dimension

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def fig_spectrum():
    m = OneLayerModel(seed=0)
    Z, _ = m.query(3 * m.h)
    _, sv = recover_dimension(Z)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.semilogy(np.arange(1, len(sv) + 1), sv, ".", ms=4, color="#1f4e79")
    ax.axvline(m.h, color="crimson", ls="--", lw=1, label=f"true $h={m.h}$")
    ax.set_xlabel("singular value index"); ax.set_ylabel("magnitude")
    ax.set_title("Logit spectrum: a clean cliff at $h$ (degree-1 part of the ideal)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "spectrum.pdf")); plt.close(fig)


def fig_main():
    m1 = OneLayerModel(seed=0)
    m2 = TwoLayerModel(seed=1)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.6))

    # left: noise -- R1 robust, R2/R3 fragile
    rows = noise_sweep(m1, [1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    sig = np.array([r["sigma"] for r in rows])
    gap = np.array([r["gap"] for r in rows])
    rms = np.array([r["orth_rms"] for r in rows])
    axL.loglog(sig, gap, "o-", color="#1f4e79", label="rank gap $\\sigma_h/\\sigma_{h+1}$ (R1)")
    axL.loglog(sig, rms, "s-", color="crimson", label="$W$ recovery RMS (R2/R3)")
    axL.axhline(1.0, color="gray", ls=":", lw=1)
    axL.set_xlabel("logit noise $\\sigma$"); axL.set_ylabel("value")
    axL.set_title("Rank robust ($\\sim\\!1/\\sigma$), quadric fragile ($\\sim\\!\\sigma$)")
    axL.legend(frameon=False, fontsize=9, loc="upper right")

    # right: span vs intrinsic dimension
    Z1, _ = m1.query(4000); s1, _ = recover_dimension(Z1)
    i1 = intrinsic_dimension(Z1, s1, rng=np.random.default_rng(1))
    Z2, _ = m2.query(4000); s2, _ = recover_dimension(Z2)
    i2 = intrinsic_dimension(Z2, s2, rng=np.random.default_rng(1))
    x = np.arange(2)
    axR.bar(x - 0.18, [s1, s2], 0.36, label="linear span (SVD)", color="#1f4e79")
    axR.bar(x + 0.18, [i1, i2], 0.36, label="intrinsic dim", color="#e0a106")
    axR.set_xticks(x); axR.set_xticklabels(["one-layer", "two-layer"])
    axR.set_ylabel("dimension")
    axR.set_ylim(0, 80)                                   # headroom so the legend clears the bars
    axR.set_title("Intrinsic dim exposes a hidden\nnonlinear bottleneck")
    axR.legend(frameon=False, fontsize=9, loc="upper right")  # clear region above two-layer bars
    for xi, (sp, idim) in zip(x, [(s1, i1), (s2, i2)]):
        axR.text(xi - 0.18, sp + 1.2, str(sp), ha="center", fontsize=9)
        axR.text(xi + 0.18, idim + 1.2, str(idim), ha="center", fontsize=9)

    fig.tight_layout(); fig.savefig(os.path.join(OUT, "main.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_spectrum()
    fig_main()
    print("figures written to", os.path.abspath(OUT))
