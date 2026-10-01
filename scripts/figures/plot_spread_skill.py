#!/usr/bin/env python3
"""Plot ensemble spread against RMSE from the final case metrics CSV."""

import argparse
from csv import DictReader
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotting_style as ns

ns.apply()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Final uncertainty case metrics CSV")
    ap.add_argument("--outdir", required=True, help="Output directory for the binned CSV.")
    ap.add_argument("--fig-dir",
                    default="figures", help="Figure output directory.")
    ap.add_argument("--nbins", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    rmse, spread, cover = [], [], []
    with open(args.metrics, newline="") as source:
        for row in DictReader(source):
            r = row.get("rmse")
            s = row.get("spread_mean")
            if r is None or s is None:
                continue
            r, s = float(r), float(s)
            if not (np.isfinite(r) and np.isfinite(s)):
                continue
            rmse.append(r)
            spread.append(s)
            cover.append(float(row.get("plot_cover", np.nan)))
    rmse = np.asarray(rmse)
    spread = np.asarray(spread)
    cover = np.asarray(cover)
    n = rmse.size
    if n == 0:
        raise RuntimeError("No valid (rmse, spread_mean) pairs found.")

    overall_rmse = float(np.sqrt(np.mean(rmse ** 2)))  # RMS of case RMSE
    mean_spread = float(np.mean(spread))
    ratio = mean_spread / overall_rmse if overall_rmse > 0 else np.nan
    corr = float(np.corrcoef(spread, rmse)[0, 1]) if n > 1 else np.nan

    # Binned spread-skill (bin by predicted spread quantiles).
    qs = np.quantile(spread, np.linspace(0, 1, args.nbins + 1))
    qs[-1] += 1e-6
    bin_spread, bin_rmse, bin_n = [], [], []
    for i in range(args.nbins):
        m = (spread >= qs[i]) & (spread < qs[i + 1])
        if m.sum() == 0:
            continue
        bin_spread.append(float(np.mean(spread[m])))
        bin_rmse.append(float(np.sqrt(np.mean(rmse[m] ** 2))))
        bin_n.append(int(m.sum()))
    bin_spread = np.asarray(bin_spread)
    bin_rmse = np.asarray(bin_rmse)

    csv = os.path.join(args.outdir, "spread_skill_bins.csv")
    with open(csv, "w") as f:
        f.write("bin,mean_spread,rms_of_rmse,n\n")
        for i in range(len(bin_spread)):
            f.write(f"{i},{bin_spread[i]:.4f},{bin_rmse[i]:.4f},{bin_n[i]}\n")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    lim = max(float(np.max(spread)), float(np.max(rmse))) * 1.05
    sc = axes[0].scatter(spread, rmse, c=cover * 100, cmap=ns.SCATTER_CMAP, s=16,
                         alpha=1.0, edgecolors="white", linewidths=0.25)
    axes[0].plot([0, lim], [0, lim], "--", color=ns.PALETTE["black"], lw=0.9,
                 label="1:1 (calibrated)")
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Mean ensemble spread (m s$^{-1}$)")
    axes[0].set_ylabel("Ensemble-mean RMSE (m s$^{-1}$)")
    axes[0].set_title(f"Case-level spread\u2013skill ($n$={n})")
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label("Sea cover (%)")
    cb.outline.set_linewidth(0.6)
    axes[0].legend(loc="upper left")
    axes[0].text(0.97, 0.05, f"$r$={corr:.2f}\nspread/RMSE={ratio:.2f}",
                 transform=axes[0].transAxes, ha="right", va="bottom",
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"), fontsize=7)

    axes[1].plot([0, lim], [0, lim], "--", color=ns.PALETTE["black"], lw=0.9,
                 label="1:1 (calibrated)")
    axes[1].plot(bin_spread, bin_rmse, "-o", color=ns.PALETTE["vermilion"],
                 lw=1.4, ms=4.5, label="binned")
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("Mean ensemble spread (m s$^{-1}$)")
    axes[1].set_ylabel("RMS of case RMSE (m s$^{-1}$)")
    axes[1].set_title("Binned spread\u2013skill reliability")
    axes[1].legend(loc="upper left")
    for ax in axes:
        ns.despine(ax)
    ns.panel_label(axes[0], "a")
    ns.panel_label(axes[1], "b")
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "spread_skill.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(args.fig_dir, "spread_skill.pdf"), bbox_inches="tight")
    plt.close(fig)

    print("Spread-skill: RMSE vs ensemble spread")
    print(f"n cases = {n}")
    print(f"RMS of case RMSE   = {overall_rmse:.3f} m/s")
    print(f"mean ensemble spread = {mean_spread:.3f} m/s")
    print(f"spread/RMSE ratio  = {ratio:.3f}   (<1 => under-dispersed / overconfident)")
    print(f"corr(spread, RMSE) = {corr:.3f}   (>0 => larger predicted spread tracks larger error)")
    print(f"Table:  {csv}")
    print(f"Figure: {os.path.join(args.fig_dir, 'spread_skill.png')}")


if __name__ == "__main__":
    main()
