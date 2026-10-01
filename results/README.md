# Evaluation data

These files contain the processed outputs used in the accepted manuscript. Wind speeds and errors are in m/s; coverage is a fraction unless a column states otherwise. No negative reconstruction values were clipped.

- `test_timestamps.csv`: the 4,420 unique main-test timestamps, in the `t0` column.
- `withholding_case_metrics.csv`: one row per test timestamp; `n` is the number of evaluated sea cells after removing cells that also contain an input observation. `n_overlap` is the number removed. There are 4,418 cases with at least one evaluated cell.
- `withholding_pixel_data.npz`: pooled `ref`, `rec`, `distance` (km), `coverage`, and `case` arrays for the same grid-disjoint verification cells. `case` is a date/hour grouping label, not a complete timestamp.
- `uncertainty_case_metrics.csv`: 243 uncertainty cases from the main manifest, using the same input-cell exclusion. Interval coverage, NLL and CRPS use a Gaussian defined by the ensemble mean and population standard deviation. `plot_cover` is input sea coverage.
- `uncertainty_pixel_data.npz`: pooled pixel `spread`, absolute `error`, `distance` (km), and `case` arrays for the uncertainty diagnostics. The 243 distinct `case` labels, sorted in ascending order, correspond to the rows of `uncertainty_case_metrics.csv`; the labels are not contiguous row indices. Use `np.unique(case, return_inverse=True)` to obtain consecutive zero-based row indices.
- `uncertainty_error_spread_bins.csv`, `uncertainty_distance_spread_bins.csv`: the error–spread and distance–spread summaries in Figure 7. Distance summaries average within a case first, then across cases; `sem` is the standard error across cases.
- `reference_comparison_statistics.json`: per-case sufficient statistics for reconstruction and training-period CCMP climatology against ERA5 and CCMP, on identical valid sea cells. `sd` sums estimate minus reference; `sd2` sums its square; `sa` sums its absolute value. `sx` and `sy` sum reference and estimate, respectively. `ccmp_dt` is the matching-time difference in hours; retained CCMP pairs require a difference of at most 3.5 hours.
- `training_mean_wind.npy`, `training_data_summary.json`: the 160 × 160 climatological mean in m/s and the record of the 37,284 training frames used to form it. It is a deterministic baseline, not an unconditional EDM sample.
- `cyclone_intensity_cases.csv`: 1,563 storm-centred 32 × 32 cases and intensity estimates in m/s; one case lacks a CCMP match. `cygnss_cov` divides observed cells by all 1,024 box cells.
- `cyclone_intensity_by_coverage.csv`: interpolation comparisons on jointly eligible storm cases. Their sample sets differ from comparisons using every available reconstruction/reference pair.
- `interpolation_comparison.json`: paired nearest, linear, ordinary-kriging and SDA error accumulators on 4,281 eligible withholding cases. Each accumulator stores count, squared-error sum, absolute-error sum, and signed-error sum, in that order.

Run `python analysis/evaluate_saved_results.py` to recalculate the principal scores. The output is written to `results/reproduced/paper_scores.json`.

The files support the listed principal evaluations and saved-case examples. They do not contain every full-domain reconstruction, raw source-product file, original ensemble member, or trained model checkpoint. The separate ensemble-size sensitivity experiment uses empirical-member CRPS and is not reconstructed from these Gaussian summary data.

Source reference products retain their provider terms; see the source links in the repository README. Reconstruction outputs and analysis summaries were generated for this study.
