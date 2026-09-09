"""Plot saved wind fields."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_case(path, output):
    with np.load(path, allow_pickle=False) as z:
        fields = {k: z[k] for k in z.files}
    sea = fields['sea_mask'].astype(bool)
    vmax = float(np.nanmax(fields['reconstruction'][sea]))
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
    cb = fig.colorbar(image, cax=colorbar_ax, orientation='horizontal', extend='max')
    cb.set_label('Wind speed (m s$^{-1}$)', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle(path.stem.replace('_', ' '), fontsize=9, y=.97)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf']:
        fig.savefig(output / (path.stem + '.' + suffix), dpi=300)
    plt.close(fig)
    valid = (sea & fields['obs_mask'].astype(bool)
             & np.isfinite(fields['cygnss']) & np.isfinite(fields['reconstruction']))
    # Agreement with input observations.
    difference = fields['reconstruction'][valid] - fields['cygnss'][valid]
    rmse = np.sqrt(np.mean(difference**2)) if difference.size else np.nan
    print(f'{path.stem}: wind scale 0–{vmax:.3f} m/s; '
          f'{valid.sum()} observed sea cells; observed-cell RMSE {rmse:.3f} m/s')


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
    for path in paths:
        plot_case(path, args.output_dir)


if __name__ == '__main__':
    main()
