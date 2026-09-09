"""Plot saved wind fields."""
from pathlib import Path
import argparse
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import deterministic_scores


def plot_case(path, output):
    with np.load(path, allow_pickle=False) as z:
        fields = {k: z[k] for k in z.files}
    sea = fields['sea_mask'].astype(bool)
    reconstructed_sea = fields['reconstruction'][sea]
    vmin = float(np.nanmin(reconstructed_sea))
    vmax = float(np.nanmax(reconstructed_sea))
    negative_count = int(np.sum(reconstructed_sea < 0))
    names = [('cygnss', 'CYGNSS'), ('reconstruction', 'Reconstruction'),
             ('era5', 'ERA5'), ('ccmp', 'CCMP')]
    names = [(key, title) for key, title in names if key in fields]
    fig, axes = plt.subplots(1, len(names), figsize=(9.6, 3.8), squeeze=False)
    fig.subplots_adjust(left=.07, right=.99, bottom=.30, top=.82, wspace=.10)
    lon, lat = fields['lon'], fields['lat']
    extent = [lon[0]-.125, lon[-1]+.125, lat[0]-.125, lat[-1]+.125]
    norm = Normalize(0, vmax)
    for i, (key, title) in enumerate(names):
        ax = axes[0, i]
        image = ax.imshow(np.where(sea, fields[key], np.nan), origin='lower',
                          extent=extent, cmap='viridis', norm=norm,
                          interpolation='nearest')
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Longitude (°E)', fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.set_ylabel('Latitude (°N)', fontsize=7)
        else:
            ax.tick_params(labelleft=False)
    colorbar_ax = fig.add_axes([.32, .13, .40, .025])
    extension = 'both' if negative_count else 'max'
    cb = fig.colorbar(image, cax=colorbar_ax, orientation='horizontal', extend=extension)
    cb.set_label('Wind speed (m s$^{-1}$)', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle(path.stem.replace('_', ' '), fontsize=9, y=.97)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf']:
        fig.savefig(output / (path.stem + '.' + suffix), dpi=300)
    plt.close(fig)
    print(f'{path.stem}: reconstructed sea range {vmin:.6f}–{vmax:.6f} m/s; '
          f'{negative_count} negative sea cells (retained)')
    rows = []
    for key in ['cygnss', 'era5', 'ccmp']:
        if key not in fields:
            continue
        mask = sea.copy()
        if key == 'cygnss':
            mask &= fields['obs_mask'].astype(bool)
        valid = mask & np.isfinite(fields[key]) & np.isfinite(fields['reconstruction'])
        scores = deterministic_scores(fields['reconstruction'], fields[key], mask=valid)
        target = 'input_cygnss' if key == 'cygnss' else key
        row = dict(file=path.name, target=target, n=int(valid.sum()), **scores)
        rows.append(row)
        print(f"  {target}: n={row['n']}; Bias={scores['bias']:.6f}; "
              f"RMSE={scores['rmse']:.6f}; MAE={scores['mae']:.6f} m/s; "
              f"r={scores['corr']:.6f}")
    return rows



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).parent / 'data')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / 'output')
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'axes.linewidth': .6, 'axes.spines.top': False,
                         'axes.spines.right': False})
    paths = sorted(args.data_dir.glob('*.npz'))
    if not paths:
        raise FileNotFoundError(f'No NPZ cases in {args.data_dir}')
    rows = []
    for path in paths:
        rows.extend(plot_case(path, args.output_dir))
    with (args.output_dir / 'case_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['file', 'target', 'n', 'bias', 'rmse', 'mae', 'corr'])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
