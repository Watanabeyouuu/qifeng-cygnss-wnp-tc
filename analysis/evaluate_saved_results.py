"""Calculate the main paper results from the supplied evaluation data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "results"


def error_scores(errors):
    """Bias is estimate minus reference; wind speeds are in m/s."""
    return {
        "n": len(errors),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
    }


def summarize_withholding():
    cases = pd.read_csv(DATA_DIR / "withholding_case_metrics.csv")
    valid_cases = cases[cases["n"] > 0]
    with np.load(DATA_DIR / "withholding_pixel_data.npz") as pixels:
        errors = pixels["rec"] - pixels["ref"]
        pooled_scores = error_scores(errors)
    return {
        "total_cases": len(cases),
        "valid_cases": len(valid_cases),
        "case_mean_rmse": float(valid_cases["rmse"].mean()),
        "case_mean_mae": float(valid_cases["mae"].mean()),
        "case_mean_bias": float(valid_cases["bias"].mean()),
        "pixel_pooled": pooled_scores,
        "excluded_input_overlap_pairs": int(cases["n_overlap"].sum()),
        "negative_fraction": float(cases["negative"].sum() / cases["sea"].sum()),
    }


def pooled_reference_scores(case_statistics):
    """Pool error sums over identical valid sea-grid pairs."""
    totals = {
        name: sum(case[name] for case in case_statistics)
        for name in case_statistics[0]
    }
    count = totals["n"]
    reference_variance = totals["sxx"] - totals["sx"] ** 2 / count
    estimate_variance = totals["syy"] - totals["sy"] ** 2 / count
    covariance = totals["sxy"] - totals["sx"] * totals["sy"] / count
    return {
        "cases": totals["cases"],
        "npix": count,
        "rmse": float(np.sqrt(totals["sd2"] / count)),
        "mae": totals["sa"] / count,
        "bias": totals["sd"] / count,
        "corr": covariance / np.sqrt(reference_variance * estimate_variance),
    }


def summarize_reference_products():
    cases = json.loads(
        (DATA_DIR / "reference_comparison_statistics.json").read_text()
    )
    result = {}
    comparisons = [
        "ERA5_reconstruction", "ERA5_training_mean",
        "CCMP_reconstruction", "CCMP_training_mean",
    ]
    for comparison in comparisons:
        matched_statistics = []
        for case in cases:
            if comparison not in case["scores"]:
                continue
            if comparison.startswith("CCMP") and float(case["ccmp_dt"]) > 3.5:
                continue
            matched_statistics.append(case["scores"][comparison])
        result[comparison] = pooled_reference_scores(matched_statistics)
    return result


def summarize_uncertainty():
    cases = pd.read_csv(DATA_DIR / "uncertainty_case_metrics.csv")
    metric_names = [
        "rmse", "mae", "spread_mean", "z_mean", "z_std", "nll", "gaussian_crps",
        "picp_50", "picp_80", "picp_90", "picp_95",
    ]
    result = {"cases": len(cases)}
    for name in metric_names:
        result[name] = float(cases[name].mean())

    nominal_levels = np.array([0.50, 0.80, 0.90, 0.95])
    observed_coverage = np.array([
        result["picp_50"], result["picp_80"], result["picp_90"], result["picp_95"],
    ])
    result["ece"] = float(np.mean(np.abs(observed_coverage - nominal_levels)))
    rms_case_error = np.sqrt(np.mean(cases["rmse"] ** 2))
    result["spread_to_rmse_ratio"] = float(result["spread_mean"] / rms_case_error)

    result["coverage_bins"] = {}
    bin_edges = [0.00, 0.05, 0.10, 0.15, 0.20, np.inf]
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        subset = cases[
            (cases["plot_cover"] >= lower) & (cases["plot_cover"] < upper)
        ]
        label = f"{lower:.2f}-{upper:.2f}" if np.isfinite(upper) else ">=0.20"
        result["coverage_bins"][label] = {"cases": len(subset)}
        for name in ["rmse", "spread_mean", "nll", "gaussian_crps"]:
            result["coverage_bins"][label][name] = float(subset[name].mean())
    return result


def summarize_cyclone_intensity():
    cases = pd.read_csv(DATA_DIR / "cyclone_intensity_cases.csv")
    subsets = {
        "all": np.ones(len(cases), dtype=bool),
        "coverage_below_10": cases["cygnss_cov"] < 0.10,
        "coverage_above_10": cases["cygnss_cov"] > 0.10,
    }
    estimates = ["rec_vmax", "era5_vmax", "ccmp_vmax", "cygnss_vmax"]
    result = {}
    for label, selected in subsets.items():
        result[label] = {}
        for estimate in estimates:
            pairs = cases.loc[selected, [estimate, "ibt_vmax"]].dropna()
            errors = (pairs[estimate] - pairs["ibt_vmax"]).to_numpy()
            result[label][estimate] = error_scores(errors)
    return result


def summarize_interpolation():
    comparison = json.loads((DATA_DIR / "interpolation_comparison.json").read_text())
    result = {"matched_cases": comparison["n_cases"], "methods": {}}
    for method, statistics in comparison["statistics"].items():
        count, squared_error, absolute_error, signed_error = statistics["all"]
        result["methods"][method] = {
            "n": int(count),
            "rmse": float(np.sqrt(squared_error / count)),
            "mae": absolute_error / count,
            "bias": signed_error / count,
        }
    return result


def main():
    report = {
        "withhold": summarize_withholding(),
        "gridded_references": summarize_reference_products(),
        "uncertainty": summarize_uncertainty(),
        "tc_intensity": summarize_cyclone_intensity(),
        "interpolation": summarize_interpolation(),
    }
    output_dir = DATA_DIR / "reproduced"
    output_dir.mkdir(exist_ok=True)
    output = json.dumps(report, indent=2)
    (output_dir / "paper_scores.json").write_text(output)
    print(output)


if __name__ == "__main__":
    main()
