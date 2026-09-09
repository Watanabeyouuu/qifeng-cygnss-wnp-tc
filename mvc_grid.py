import numpy as np


def mvc_grid(lon, lat, wind, lon_min, lat_min,
             dlon=0.25, dlat=0.25, nlon=160, nlat=160):
    """Cell maxima and sample counts; empty cells are NaN."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    wind = np.asarray(wind, float)

    ix = np.floor((lon - lon_min) / dlon).astype(int)
    iy = np.floor((lat - lat_min) / dlat).astype(int)
    keep = (ix >= 0) & (ix < nlon) & (iy >= 0) & (iy < nlat) & np.isfinite(wind)
    ix, iy, wind = ix[keep], iy[keep], wind[keep]

    grid = np.full((nlat, nlon), -np.inf)
    np.maximum.at(grid, (iy, ix), wind)

    count = np.zeros((nlat, nlon), int)
    np.add.at(count, (iy, ix), 1)

    grid[count == 0] = np.nan
    return grid, count


def mean_grid(lon, lat, wind, lon_min, lat_min,
              dlon=0.25, dlat=0.25, nlon=160, nlat=160):
    """Cell means and sample counts; empty cells are NaN."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    wind = np.asarray(wind, float)

    ix = np.floor((lon - lon_min) / dlon).astype(int)
    iy = np.floor((lat - lat_min) / dlat).astype(int)
    keep = (ix >= 0) & (ix < nlon) & (iy >= 0) & (iy < nlat) & np.isfinite(wind)
    ix, iy, wind = ix[keep], iy[keep], wind[keep]

    total = np.zeros((nlat, nlon))
    count = np.zeros((nlat, nlon), int)
    np.add.at(total, (iy, ix), wind)
    np.add.at(count, (iy, ix), 1)

    grid = np.full((nlat, nlon), np.nan)
    nz = count > 0
    grid[nz] = total[nz] / count[nz]
    return grid, count
