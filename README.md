# CYGNSS Western North Pacific wind reconstruction

Selected reconstruction results and companion utilities for *Diffusion-Based Reconstruction of Western North Pacific Surface Winds from CYGNSS for Tropical Cyclone Analysis*.

## Read and plot

```bash
python -m pip install -r requirements.txt
python examples/plot_cases.py
```

The example reads saved wind fields and writes PNG and PDF comparisons to `examples/output/`. It runs on CPU with NumPy and Matplotlib.

```python
import numpy as np
from metrics import deterministic_scores

with np.load("examples/data/soulik_20180818T180000.npz", allow_pickle=False) as case:
    wind = case["reconstruction"]
    print(wind.shape)
    print(deterministic_scores(wind, case["era5"], mask=case["sea_mask"]))
```

## Saved cases

| Storm | UTC | Grid | Paper figure |
|---|---|---|---|
| Soulik | 2018-08-18 18:00 | 160 × 160 | Figure 2 |
| Surigae | 2021-04-18 09:00, 04-19 09:00, 04-20 09:00, 04-21 00:00 | 32 × 32 | Selected frames from Figure 9 |

Wind speeds are in m/s. Arrays use `[latitude, longitude]` order, with increasing coordinates on a 0.25° grid. `lon` and `lat` give cell centres. Each file contains `reconstruction`, `cygnss`, `era5`, `obs_mask`, and `sea_mask`; the Soulik file also contains `ccmp`. Unsampled CYGNSS cells contain NaN.

The saved reconstructions retain the model outputs, including negative values. By default, plots and metrics use these values without clipping, consistent with the manuscript's treatment of reconstructed winds. The sampler does not enforce nonnegative wind speed, as discussed in the manuscript. These negative values are nonphysical model outputs. Soulik contains 31 negative sea cells out of 19,046 (0.163%), with a minimum of -0.439342 m/s; the four Surigae snapshots contain no negative sea values.

Plots share a scale from zero to the maximum reconstructed sea wind for each case. Colourbar extensions mark values outside that display range, including negative values when present. The display limits do not change the arrays used for scoring.

To optionally set finite negative reconstruction values to zero before both plotting and scoring:

```bash
python examples/plot_cases.py --clip-negative
```

This option changes only the reconstruction array in memory; source NPZ files, observations and reference fields are unchanged. NaN and other non-finite values remain excluded from scoring. Clipped figures use an `_clipped` filename suffix and metrics are saved to `case_metrics_clipped.csv`, so they do not overwrite the default outputs. The CSV `processing` column identifies `raw` or `clip_negative`. Clipping is an optional post-processing variant, not the processing used for the manuscript's reported results.

## Reproducing the saved-case metrics

The plotting command also writes `examples/output/case_metrics.csv`. By default, Bias, RMSE, MAE and correlation use the unmodified arrays, excluding land and non-finite pairs. The same definitions and masks are used with `--clip-negative`, applied to the clipped reconstruction. Rows labelled `input_cygnss` use observed sea cells and measure agreement with the reconstruction input; they are not the manuscript's independent, withheld-observation validation. ERA5 and CCMP rows use all finite sea-grid pairs in the saved domain or patch. Expected values for the default, unclipped mode are in `examples/expected_metrics.csv`. The manuscript does not list these per-snapshot error scores separately.

The sea-grid reconstruction maxima reproduce the corresponding Figure 2 and Figure 9 values (rounded to 0.1 m/s):

| Saved field | Reconstruction maximum (m/s) |
|---|---:|
| Soulik 2018-08-18 18:00 | 51.9 |
| Surigae 2021-04-18 09:00 | 48.8 |
| Surigae 2021-04-19 09:00 | 39.8 |
| Surigae 2021-04-20 09:00 | 34.7 |
| Surigae 2021-04-21 00:00 | 38.5 |

These files cover two storms at five timestamps. The manuscript's full-test RMSE, coverage-bin summaries and uncertainty scores use larger evaluation sets and cannot be recalculated from these five examples alone. The small negative-value count above likewise describes the released Soulik example, not the full withholding experiment.

## CYGNSS gridding

`mvc_grid.py` provides maximum-value compositing and a mean-compositing helper. Inputs are observation longitude, latitude, and wind speed arrays after quality screening and time selection. `lon_min` and `lat_min` specify the lower grid edges; empty cells return NaN, and `count` gives the number of observations per cell.

```python
from mvc_grid import mvc_grid

# Illustrative observations: two samples fall in the same grid cell.
grid, count = mvc_grid(
    lon=[120.10, 120.15], lat=[15.10, 15.15], wind=[12.0, 18.0],
    lon_min=120.0, lat_min=15.0, nlon=4, nlat=4,
)
print(grid[0, 0], count[0, 0])  # 18.0, 2
```

`metrics.py` provides Bias (reconstruction minus reference), RMSE, MAE, and correlation over valid masked cells.

## Data and license

Code uses the MIT license. Source data retain their provider terms: [CYGNSS L2 SWSP v1.2](https://doi.org/10.5067/CYGNN-22512), [ERA5](https://doi.org/10.24381/cds.adbb2d47), and [CCMP v3.1](https://data.remss.com/ccmp/v03.1/). Reconstruction arrays are study outputs. Cite the paper and the corresponding source products when using these examples.
