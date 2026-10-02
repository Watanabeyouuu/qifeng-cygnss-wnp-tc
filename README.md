# Diffusion-Based Reconstruction of Western North Pacific Surface Winds from CYGNSS for Tropical Cyclone Analysis

Code examples and processed evaluation data accompanying the paper.

Version 1.0.0 is archived at [Zenodo](https://doi.org/10.5281/zenodo.23073928).

## Plot and evaluate the saved results

```bash
python -m pip install -r requirements.txt
python examples/plot_cases.py
python analysis/evaluate_saved_results.py
```

These commands run on CPU. The first draws the five saved wind fields and calculates their Bias, RMSE, MAE and correlation. The second calculates the principal withholding, reference-product, uncertainty and tropical-cyclone intensity results, plus the matched interpolation comparison. See [results/README.md](results/README.md) for the data definitions.

For the spread–skill diagnostic:

```bash
python scripts/figures/plot_spread_skill.py --metrics results/uncertainty_case_metrics.csv --outdir figures --fig-dir figures
```

The analysis uses the accepted paper's sample definitions: 4,420 main-test timestamps, 4,418 valid grid-disjoint withholding cases, 243 uncertainty cases and 1,563 cyclone-centred cases. The supplied results support the principal comparisons; the supplementary experiments use additional inputs.

## Saved reconstruction examples

| Case | UTC | Grid | Paper figure |
|---|---|---|---|
| Soulik | 2018-08-18 18:00 | 160 × 160 | Figure 2 |
| Surigae | 2021-04-18 09:00, 04-19 09:00, 04-20 09:00, 04-21 00:00 | 32 × 32 | Selected frames from Figure 9 |

Each NPZ contains `reconstruction`, gridded `cygnss` observations, `era5`, `obs_mask`, `sea_mask`, `lon` and `lat`; Soulik also contains `ccmp`. Wind speeds are in m/s. Arrays use `[latitude, longitude]` order with increasing coordinates on a 0.25° grid. Unsampled CYGNSS cells are NaN. These are gridded observation and reference fields, rather than complete source-product files.

```python
import numpy as np
from metrics import deterministic_scores

with np.load("examples/data/soulik_20180818T180000.npz") as case:
    print(deterministic_scores(case["reconstruction"], case["era5"], case["sea_mask"]))
```

The sea-grid reconstruction maxima are 51.9 m/s for Soulik and 48.8, 39.8, 34.7 and 38.5 m/s for the four Surigae frames, rounded to 0.1 m/s as in the paper.

The arrays preserve the model outputs, including negative values. Soulik has 31 negative sea cells out of 19,046; its minimum is -0.439342 m/s. Default plots and metrics use these unmodified outputs. Colourbar extensions mark values outside the display range. `python examples/plot_cases.py --clip-negative` is an optional post-processing variant, with separate output filenames; it is not the processing used for the paper results.

Expected default metrics are in `examples/expected_metrics.csv`. Rows labelled `input_cygnss` measure agreement with the reconstruction input and are separate from the independent withholding validation. Five saved cases alone do not represent the larger evaluation sets in `results/`.

## Preprocessing, training and reconstruction examples

The diffusion prior builds on [NVIDIA EDM](https://github.com/NVlabs/edm), and observation-guided reconstruction follows [Score-based Data Assimilation (SDA)](https://github.com/francois-rozet/sda). Refer to these repositories for their model and algorithm implementations and dependencies.

The examples show how to prepare single-channel wind fields, train with the upstream EDM network and loss, and reconstruct a saved CYGNSS case. Use an unmodified upstream EDM checkout on `PYTHONPATH` and install the upstream dependencies plus `requirements_pipeline.txt`.

```bash
python examples/prepare_training_data.py --source-dir /path/to/CCMP --output-dir /path/to/ccmp_train_normalized
python examples/train_example.py --data-dir /path/to/ccmp_train_normalized --outdir example_outputs/training
python examples/reconstruct_example.py --network example_outputs/training/network_example.pkl --input examples/data/soulik_20180818T180000.npz --output example_outputs/reconstruction.npz
```

Preprocessing selects CCMP fields before 2018-08-01 over 105–145°E and 2°S–38°N and divides wind speeds by 100. `examples/training_config.json` is a small example configuration. The training example passes these floating-point normalized fields directly to the EDM loss. It illustrates the training interface; the full EDM training workflow is described in the upstream repository. `configs/inference_settings.json` records the study's reconstruction settings. The reconstruction example accepts a compatible one-channel checkpoint and a grid matching that model's resolution. New training produces a new model; the archived NPZ fields and processed results are the fields used for paper evaluation.

`mvc_grid.py` implements maximum-value compositing and a mean-compositing helper. The regional CYGNSS preprocessing example uses the same domain, a ±3-hour window and IBTrACS timestamps from August 2018 through September 2022:

```bash
python scripts/preprocessing/grid_cygnss_observations.py --cygnss-dir /path/to/CYGNSS --ibtracs-csv /path/to/ibtracs.csv --outdir /path/to/gridded_cygnss --dedup-times --no-plot
```

The retained timestamps for paper evaluation are listed in `results/test_timestamps.csv`. `metrics.py` supplies deterministic metrics and the Gaussian uncertainty definitions used in the main analysis.

## Source data and license

Study code uses the MIT license in `LICENSE`. External implementations retain their own licenses in the linked repositories. Source-product arrays retain their providers' terms:

- [CYGNSS L2 Science Wind Speed Product v1.2](https://doi.org/10.5067/CYGNN-22512)
- [ERA5 hourly single-level data](https://doi.org/10.24381/cds.adbb2d47)
- [CCMP v3.1](https://data.remss.com/ccmp/v03.1/)
- [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)
- [NOAA STAR SAR winds](https://www.star.nesdis.noaa.gov/socd/mecb/sar/index.php)

Cite the paper and the corresponding source products when using the data.

Archive citation: Han, X., Li, X., Yang, J., Niu, Z., Han, G., Fu, N., Wang, J., Tao, W., Aouf, L., & Chen, D. (2026). *Diffusion-Based Reconstruction of Western North Pacific Surface Winds from CYGNSS for Tropical Cyclone Analysis* (Version 1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23073928
