"""Prepare the paper's normalized CCMP training fields before 2018-08-01."""

import os
import glob
import json
from multiprocessing import Pool, cpu_count

import numpy as np
import xarray as xr
from tqdm import tqdm

# Configuration
SOURCE_DIR = "example_data/CCMP_V03p1_daily"
OUTPUT_DIR = "example_data/raw_data_ccmp_105E145E_2S38N_norm"

# Crop box (edge bounds); actual centers follow CCMP grid (0.25° + 0.125 phase)
LON_MIN = 105.0
LON_MAX = 145.0
LAT_MIN = -2.0
LAT_MAX = 38.0

# Fixed patch size (aligns with CCMP grid using index slicing)
PATCH_SIZE = 160

# Wind speeds in m/s are divided by 100.
SCALE_FACTOR = 100.0

# Time cutoff: keep only strictly before this time
CUTOFF_TIME = np.datetime64("2018-08-01T00:00:00")


def _format_time_str(tval) -> str:
    try:
        return np.datetime_as_string(tval, unit="h")
    except Exception:
        return str(tval)


def _index_slice(coord: np.ndarray, vmin: float, size: int) -> slice:
    # coord is expected ascending
    i0 = int(np.searchsorted(coord, vmin, side="left"))
    i1 = i0 + size
    if i0 < 0 or i1 > coord.size:
        raise ValueError(f"Index slice out of range: i0={i0}, i1={i1}, size={coord.size}")
    return slice(i0, i1)


def process_file(fpath: str):
    fname = os.path.basename(fpath)
    base_name = os.path.splitext(fname)[0]
    out_labels = []

    try:
        ds = xr.open_dataset(fpath)
        if "ws" not in ds:
            return out_labels

        lat = ds["latitude"].values
        lon = ds["longitude"].values
        lat_slice = _index_slice(lat, LAT_MIN, PATCH_SIZE)
        lon_slice = _index_slice(lon, LON_MIN, PATCH_SIZE)

        da = ds["ws"].isel(latitude=lat_slice, longitude=lon_slice)
        da = da.load()

        for idx in range(da.sizes.get("time", 1)):
            if "time" not in da.dims:
                raise ValueError("CCMP input needs a time coordinate for the training cutoff")
            tval = da["time"].values[idx]
            tcmp = np.datetime64(tval)
            if np.isnat(tcmp) or tcmp >= CUTOFF_TIME:
                continue
            tstr = _format_time_str(tval).replace(":", "")
            out_name = f"{base_name}_{tstr}.npy"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            if os.path.exists(out_path):
                out_labels.append([out_name, 0])
                continue

            if "time" in da.dims:
                arr = da.isel(time=idx).values
            else:
                arr = da.values

            arr = arr.astype(np.float32)
            if not np.isfinite(arr).all():
                continue
            arr_norm = arr / SCALE_FACTOR
            if arr_norm.ndim == 2:
                arr_norm = arr_norm[np.newaxis, :, :]

            if not np.isfinite(arr_norm).all():
                continue

            np.save(out_path, arr_norm)
            out_labels.append([out_name, 0])

        return out_labels
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return out_labels
    finally:
        try:
            ds.close()
        except Exception:
            pass


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SOURCE_DIR, "**", "*.nc"), recursive=True))
    print(f"Found {len(files)} files.")

    labels = []
    num_workers = max(1, cpu_count() - 2)
    with Pool(num_workers) as p:
        for res in tqdm(p.imap_unordered(process_file, files), total=len(files)):
            if res:
                labels.extend(res)

    dataset_json_path = os.path.join(OUTPUT_DIR, "dataset.json")
    with open(dataset_json_path, "w") as f:
        json.dump({"labels": labels}, f)

    print(f"Dataset generated at {OUTPUT_DIR}")
    print(f"Total samples: {len(labels)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare normalized CCMP training fields")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    SOURCE_DIR = args.source_dir
    OUTPUT_DIR = args.output_dir
    main()