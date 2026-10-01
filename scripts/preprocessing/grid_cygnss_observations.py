"""Grid CYGNSS L2 v1.2 winds around Western Pacific IBTrACS timestamps.

Maximum-value compositing uses a 160 by 160 grid over 105–145E, 2S–38N
and a six-hour observation window. Timestamps span 2018-08-01 to 2022-09-30.

Example:
    python scripts/preprocessing/grid_cygnss_observations.py \
        --cygnss-dir /path/to/CYGNSS_L2_V1p2 \
        --ibtracs-csv /path/to/ibtracs.csv \
        --outdir /path/to/gridded_cygnss --dedup-times --no-plot
"""

import argparse
import glob
import json
import os
from datetime import datetime, timedelta, date
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Optional plotting deps
try:  # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    _HAS_CARTOPY = True
except Exception:  # pragma: no cover
    _HAS_CARTOPY = False

_WARNED_NO_CARTOPY = False
_WARNED_NO_QC_FLAGS = False


def normalize_lon_deg(lon: np.ndarray) -> np.ndarray:
    """Normalize longitude to [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


def choose_var(ds: xr.Dataset, candidates) -> str:
    for c in candidates:
        if c in ds.variables:
            return c
    for c in ds.variables:
        if c.lower() in candidates:
            return c
    raise ValueError(f"None of {candidates} found in dataset")


def _fatal_flag_mask(da: xr.DataArray) -> Optional[int]:
    """Build combined bitmask for fatal_* flags from CF-style attrs."""
    masks = da.attrs.get("flag_masks")
    meanings = da.attrs.get("flag_meanings")
    if masks is None or meanings is None:
        return None
    if isinstance(meanings, str):
        meanings = meanings.split()
    try:
        masks = np.asarray(masks, dtype=np.int64)
    except Exception:
        return None
    fatal_mask = 0
    for m, meaning in zip(masks, meanings):
        if str(meaning).startswith("fatal_"):
            fatal_mask |= int(m)
    return fatal_mask if fatal_mask != 0 else None


def _named_flag_mask(da: xr.DataArray, names) -> Optional[int]:
    """Build combined bitmask for provided flag names if present in attrs."""
    masks = da.attrs.get("flag_masks")
    meanings = da.attrs.get("flag_meanings")
    if masks is None or meanings is None:
        return None
    if isinstance(meanings, str):
        meanings = meanings.split()
    try:
        masks = np.asarray(masks, dtype=np.int64)
    except Exception:
        return None
    wanted = {str(n).strip() for n in names}
    out_mask = 0
    for m, meaning in zip(masks, meanings):
        if str(meaning).strip() in wanted:
            out_mask |= int(m)
    return out_mask if out_mask != 0 else None


def parse_args():
    p = argparse.ArgumentParser(description="Fixed-box gridding of CYGNSS wind speed")
    p.add_argument("--cygnss-dir", required=True, help="Folder with CYGNSS L2 V1.2 daily .nc files")
    p.add_argument("--outdir", required=True, help="Output folder for gridded npy and metadata")
    p.add_argument(
        "--ibtracs-csv",
        default="example_data/ibtracs.since1980.list.v04r01.csv",
        help="IBTrACS CSV used to select times (default: since1980 v04r01)",
    )
    p.add_argument("--basin", type=str, default="WP", help="Basin code to keep (default WP, Western Pacific)")
    p.add_argument("--time-tol-min", type=float, default=180.0, help="Keep samples within +/- minutes of IBTrACS time")
    p.add_argument("--lon-min", type=float, default=105.0, help="West boundary (deg, default 105E)")
    p.add_argument("--lon-max", type=float, default=145.0, help="East boundary (deg, default 145E)")
    p.add_argument("--lat-min", type=float, default=-2.0, help="South boundary (deg, default 2S)")
    p.add_argument("--lat-max", type=float, default=38.0, help="North boundary (deg, default 38N)")
    p.add_argument("--grid-res-deg", type=float, default=0.25, help="Grid resolution (deg) if target-size is 0")
    p.add_argument(
        "--target-size",
        type=int,
        default=160,
        help="Output grid will be target_size x target_size. Set 0 to disable and use grid-res-deg instead.",
    )
    p.add_argument("--max-files", type=int, default=None, help="Optional limit on number of nc files")
    p.add_argument("--no-plot", action="store_true", help="Disable cartopy plot export (jpg)")
    p.add_argument("--plot-dpi", type=int, default=150, help="DPI for saved jpg plots")
    p.add_argument("--num-shards", type=int, default=1, help="Split IBTrACS times into N shards")
    p.add_argument("--shard-idx", type=int, default=0, help="Shard index [0, num_shards)")
    p.add_argument("--skip-existing", action="store_true", help="Skip if output npy already exists")
    p.add_argument("--dedup-times", action="store_true", help="Deduplicate IBTrACS timestamps before processing")
    return p.parse_args()


def parse_file_date(nc_path: str) -> Optional[date]:
    """Extract start date from CYGNSS filename pattern ...sYYYYMMDD-..."""
    base = os.path.basename(nc_path)
    marker = ".s"
    if marker not in base:
        return None
    idx = base.find(marker)
    if idx < 0 or idx + 10 > len(base):
        return None
    date_str = base[idx + 2 : idx + 10]
    try:
        return datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError:
        return None


def load_ibtracs_times(csv_path: str, basin: str) -> np.ndarray:
    df = pd.read_csv(csv_path)

    time_col = next((c for c in ["ISO_TIME", "iso_time", "ISO_TIME_DATE"] if c in df.columns), None)
    if time_col is None:
        raise ValueError("No time column found in IBTrACS CSV (expected ISO_TIME or iso_time)")

    basin_col = next((c for c in ["BASIN", "basin", "BASIN_WMO", "WMO_BASIN"] if c in df.columns), None)
    if basin and basin_col is None:
        raise ValueError("No basin column found in IBTrACS CSV (expected BASIN/BASIN_WMO)")

    if basin and basin_col:
        basin_mask = df[basin_col].astype(str).str.upper().str.contains(basin.upper())
        df = df[basin_mask]

    times = pd.to_datetime(df[time_col], utc=True, errors="coerce").dropna()

    start = pd.Timestamp("2018-08-01", tz="UTC")
    end = pd.Timestamp("2022-10-01", tz="UTC")
    times = times[(times >= start) & (times < end)]

    if times.empty:
        raise RuntimeError("No IBTrACS timestamps in the study period for this basin")

    return np.sort(times.values.astype("datetime64[ns]").astype(np.int64))




def grid_file_single_center(
    nc_path: str,
    args,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    center_ns: int,
    tol_ns: int,
    lon_step: float,
    lat_step: float,
):
    """Grid one CYGNSS daily file using a single IBTrACS center time ±tol."""
    global _WARNED_NO_QC_FLAGS
    ds = xr.open_dataset(nc_path)
    time_var = choose_var(ds, ["time", "sample_time", "Time"])
    lat_var = choose_var(ds, ["latitude", "lat", "Lat"])
    lon_var = choose_var(ds, ["longitude", "lon", "Lon"])
    ws_var = choose_var(ds, ["wind_speed", "wind_speed_surface", "windspeed", "wind_speed_mps"])

    times = pd.to_datetime(ds[time_var].values, utc=True, errors="coerce").values.astype("datetime64[ns]")
    valid_time = ~np.isnat(times)
    time_ns = times.astype(np.int64, copy=False)

    lat = np.asarray(ds[lat_var].values)
    lon = normalize_lon_deg(np.asarray(ds[lon_var].values))
    ws = np.asarray(ds[ws_var].values, dtype=np.float32)

    time_mask = valid_time & (np.abs(time_ns - center_ns) <= tol_ns)

    mask = (
        time_mask
        & np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(ws)
        & (ws >= 0)  # Filter out negative wind speeds (should not exist in wind_speed, but safety check)
        & (lat >= args.lat_min)
        & (lat <= args.lat_max)
        & (lon >= args.lon_min)
        & (lon <= args.lon_max)
    )
    qc_mask = mask
    flag_var = None
    for name in ["fds_sample_flags", "yslf_sample_flags", "mss_sample_flags", "sample_flags"]:
        if name in ds.variables:
            flag_var = name
            break
    if flag_var is not None:
        flags = np.asarray(ds[flag_var].values).astype(np.int64, copy=False)
        fatal_mask = _fatal_flag_mask(ds[flag_var])
        if fatal_mask is not None:
            qc_mask = qc_mask & ((flags & fatal_mask) == 0)
        else:
            # v1.2 sample_flags: filter poor-quality samples by named flags
            quality_mask = _named_flag_mask(
                ds[flag_var],
                [
                    "poor_overall_quality_flag",
                    "poor_quality_wind_sample_flag",
                    "poor_quality_wind_nst_flag",
                ],
            )
            if quality_mask is not None:
                qc_mask = qc_mask & ((flags & quality_mask) == 0)
            elif not _WARNED_NO_QC_FLAGS:
                print("[QC] Found flags but no recognized fatal/quality meanings; QC not applied.")
                _WARNED_NO_QC_FLAGS = True
    elif not _WARNED_NO_QC_FLAGS:
        print("[QC] No *_sample_flags found; QC not applied for this dataset.")
        _WARNED_NO_QC_FLAGS = True

    mask = qc_mask
    if not np.any(mask):
        ds.close()
        return None

    ts_masked = pd.to_datetime(times[mask])
    ts_min: Optional[datetime] = ts_masked.min() if len(ts_masked) else None
    ts_max: Optional[datetime] = ts_masked.max() if len(ts_masked) else None

    lat = lat[mask]
    lon = lon[mask]
    ws = ws[mask]

    lat_idx = np.floor((lat - args.lat_min) / lat_step).astype(int)
    lon_idx = np.floor((lon - args.lon_min) / lon_step).astype(int)

    grid = np.full((lat_grid.size, lon_grid.size), np.nan, dtype=np.float32)
    for la, lo, v in zip(lat_idx, lon_idx, ws):
        if la < 0 or la >= grid.shape[0] or lo < 0 or lo >= grid.shape[1]:
            continue
        prev = grid[la, lo]
        if np.isnan(prev) or v > prev:
            grid[la, lo] = v

    ds.close()
    return grid, ts_min, ts_max


def plot_grid_cartopy(
    grid: np.ndarray,
    args,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    ts_min: Optional[datetime],
    ts_max: Optional[datetime],
    jpg_path: str,
):
    """Quicklook plot with cartopy. Uses 0 where data are missing."""
    global _WARNED_NO_CARTOPY
    if not _HAS_CARTOPY:  # pragma: no cover
        if not _WARNED_NO_CARTOPY:
            print("cartopy/matplotlib not available, skip plotting (one-time message)")
            _WARNED_NO_CARTOPY = True
        return

    grid_plot = np.nan_to_num(grid, nan=0.0)
    if lon_grid.size > 1 and lat_grid.size > 1:
        lon_step = float(lon_grid[1] - lon_grid[0])
        lat_step = float(lat_grid[1] - lat_grid[0])
        extent = [
            float(lon_grid[0] - lon_step / 2),
            float(lon_grid[0] + (lon_grid.size - 0.5) * lon_step),
            float(lat_grid[0] - lat_step / 2),
            float(lat_grid[0] + (lat_grid.size - 0.5) * lat_step),
        ]
    else:
        extent = [args.lon_min, args.lon_max, args.lat_min, args.lat_max]

    fig = plt.figure(figsize=(6, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="110m", linewidth=0.6)
    ax.gridlines(draw_labels=True, linewidth=0.3, linestyle=":", color="gray")

    vmin = grid_plot.min() if grid_plot.size else 0.0
    vmax = grid_plot.max() if grid_plot.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    im = ax.imshow(
        grid_plot,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        interpolation="nearest",
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )

    ts_str = ""
    if ts_min is not None and ts_max is not None:
        ts_str = f"{ts_min.isoformat()}\n{ts_max.isoformat()}"
    elif ts_min is not None:
        ts_str = ts_min.isoformat()
    elif ts_max is not None:
        ts_str = ts_max.isoformat()

    if ts_str:
        ts_str += f"\n(±{int(args.time_tol_min)} min from IBTrACS)"
    else:
        ts_str = f"±{int(args.time_tol_min)} min from IBTrACS"

    ax.set_title(ts_str)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Wind speed (m/s)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(jpg_path), exist_ok=True)
    plt.savefig(jpg_path, dpi=args.plot_dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    track_times_ns = load_ibtracs_times(args.ibtracs_csv, args.basin)
    tol_ns = int(args.time_tol_min * 60 * 1e9)

    if args.dedup_times:
        track_times_ns = np.unique(track_times_ns)

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        raise ValueError("--shard-idx must be in [0, num_shards)")
    if args.num_shards > 1:
        track_times_ns = np.array(
            [t for i, t in enumerate(track_times_ns) if i % args.num_shards == args.shard_idx],
            dtype=track_times_ns.dtype,
        )
        if track_times_ns.size == 0:
            print(f"No IBTrACS times assigned to shard {args.shard_idx}/{args.num_shards}")
            return
        print(f"Shard {args.shard_idx}/{args.num_shards}: {len(track_times_ns)} IBTrACS times")

    use_target = args.target_size is not None and args.target_size > 0
    if use_target:
        lon_size = lat_size = int(args.target_size)
        lon_step = (args.lon_max - args.lon_min) / lon_size
        lat_step = (args.lat_max - args.lat_min) / lat_size
        # CCMP-aligned centers (0.25° + 0.125 phase)
        lon_grid = args.lon_min + (np.arange(lon_size) + 0.5) * lon_step
        lat_grid = args.lat_min + (np.arange(lat_size) + 0.5) * lat_step
    else:
        res = args.grid_res_deg
        lon_grid = np.arange(args.lon_min, args.lon_max, res)
        lat_grid = np.arange(args.lat_min, args.lat_max, res)
        lon_step = lat_step = res

    nc_files = sorted(glob.glob(os.path.join(args.cygnss_dir, "*.nc")))
    if args.max_files:
        nc_files = nc_files[: args.max_files]

    file_dates = {}
    for p in nc_files:
        d = parse_file_date(p)
        if d:
            file_dates[p] = d

    meta_log = open(os.path.join(args.outdir, "metadata.jsonl"), "a")
    saved = 0

    for center_ns in tqdm(track_times_ns, desc="IBTrACS times"):
        center_dt = pd.to_datetime(center_ns).to_pydatetime()
        tol_tag = f"{int(args.time_tol_min)}min"
        ts_tag = f"ibt_{center_dt.strftime('%Y%m%dT%H%M%S')}_{tol_tag}"
        out_name = f"{ts_tag}_fixed_box.npy"
        out_path = os.path.join(args.outdir, out_name)
        if args.skip_existing and os.path.exists(out_path):
            continue
        window_start = (center_dt - timedelta(minutes=args.time_tol_min)).date()
        window_end = (center_dt + timedelta(minutes=args.time_tol_min)).date()

        candidate_nc = []
        for p, d in file_dates.items():
            if d >= window_start - timedelta(days=1) and d <= window_end + timedelta(days=1):
                candidate_nc.append(p)
        if not candidate_nc:
            continue

        grid_acc = np.full((lat_grid.size, lon_grid.size), np.nan, dtype=np.float32)
        ts_min_all: Optional[datetime] = None
        ts_max_all: Optional[datetime] = None

        for nc_path in candidate_nc:
            try:
                result = grid_file_single_center(nc_path, args, lon_grid, lat_grid, int(center_ns), tol_ns, lon_step, lat_step)
            except Exception as e:
                print(f"Skip {os.path.basename(nc_path)} for {center_dt}: {e}")
                continue
            if result is None:
                continue
            grid, ts_min, ts_max = result

            mask_new = ~np.isnan(grid)
            replace = mask_new & (np.isnan(grid_acc) | (grid > grid_acc))
            grid_acc[replace] = grid[replace]

            if ts_min is not None:
                ts_min_all = ts_min if ts_min_all is None else min(ts_min_all, ts_min)
            if ts_max is not None:
                ts_max_all = ts_max if ts_max_all is None else max(ts_max_all, ts_max)

        if np.isnan(grid_acc).all():
            continue

        grid_acc = np.nan_to_num(grid_acc, nan=0.0)

        np.save(out_path, grid_acc)

        if not args.no_plot:
            jpg_name = out_name.replace(".npy", ".jpg")
            jpg_path = os.path.join(args.outdir, jpg_name)
            plot_grid_cartopy(grid_acc, args, lon_grid, lat_grid, center_dt, center_dt, jpg_path)

        meta = {
            "filename": os.path.basename(out_path),
            "source_nc": [os.path.basename(p) for p in candidate_nc],
            "ibtracs_csv": os.path.basename(args.ibtracs_csv),
            "basin": args.basin,
            "time_tol_min": float(args.time_tol_min),
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "grid_res_deg": float(lon_step),
            "lon_step_deg": float(lon_step),
            "lat_step_deg": float(lat_step),
            "lon_size": int(lon_grid.size),
            "lat_size": int(lat_grid.size),
            "target_size": int(args.target_size) if use_target else None,
            "ts_min": ts_min_all.isoformat() if ts_min_all is not None else None,
            "ts_max": ts_max_all.isoformat() if ts_max_all is not None else None,
            "ibt_center": center_dt.isoformat(),
            "time_tag": ts_tag,
        }
        meta_log.write(json.dumps(meta) + "\n")
        saved += 1

    meta_log.close()
    print(f"Done. IBTrACS centers processed: {len(track_times_ns)}, grids saved: {saved}")


if __name__ == "__main__":
    main()
