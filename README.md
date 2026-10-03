# Diffusion-Based Reconstruction of Western North Pacific Surface Winds from CYGNSS for Tropical Cyclone Analysis

Code examples and processed evaluation data accompanying the study. Version 1.0.0 is archived in [Zenodo](https://doi.org/10.5281/zenodo.23073928).

## Contents

- `examples/`: saved wind fields and preprocessing, training and reconstruction examples.
- `results/`: processed evaluation data; see [data definitions](results/README.md).
- `configs/`: reconstruction settings.
- `analysis/` and `scripts/`: evaluation, gridding and plotting utilities.

## Usage

```bash
python -m pip install -r requirements.txt
python examples/plot_cases.py
python analysis/evaluate_saved_results.py
```

These commands plot the saved examples and reproduce the principal evaluation scores on CPU. Saved fields retain raw model outputs, including negative values. Agreement with input observations is distinct from independent withholding validation.

Pipeline examples require `requirements_pipeline.txt` and an upstream [NVIDIA EDM](https://github.com/NVlabs/edm) checkout on `PYTHONPATH`; observation-guided reconstruction follows [SDA](https://github.com/francois-rozet/sda). Use each script's `--help` for input options. Reconstruction requires a compatible trained checkpoint, which is not included.

Source products: [CYGNSS v1.2](https://doi.org/10.5067/CYGNN-22512), [ERA5](https://doi.org/10.24381/cds.adbb2d47), [CCMP v3.1](https://data.remss.com/ccmp/v03.1/), [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) and [SAR winds](https://www.star.nesdis.noaa.gov/socd/mecb/sar/index.php). Full source-product datasets are not included.

## Citation

Cite the paper and the [version 1.0.0 archive](https://doi.org/10.5281/zenodo.23073928), together with the original data sources used in your analysis.
