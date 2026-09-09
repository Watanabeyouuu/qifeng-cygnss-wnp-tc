# CYGNSS Western North Pacific wind reconstruction

Selected reconstruction results and companion utilities for *Diffusion-Based Reconstruction of Western North Pacific Sea Surface Winds from CYGNSS Observations*.

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

Plots share a scale from zero to the maximum reconstructed sea wind for each case. The colourbar extension indicates values above that scale. Printed observed-cell RMSE describes agreement with the input CYGNSS observations.

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
