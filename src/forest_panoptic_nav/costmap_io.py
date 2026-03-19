"""I/O utilities for traversability cost maps.

Supports saving / loading as ``.npz`` and optional GeoTIFF export
(requires ``rasterio``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .traversability import CostMap


def save_costmap(cost_map: CostMap, path: str | Path) -> None:
    """Save a ``CostMap`` to a ``.npz`` file with metadata.

    Args:
        cost_map: The cost map to persist.
        path: Destination file path (should end in ``.npz``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        grid=cost_map.grid,
        cost=cost_map.cost,
        origin=cost_map.origin,
        resolution=np.float64(cost_map.resolution),
    )


def load_costmap(path: str | Path) -> CostMap:
    """Load a ``CostMap`` from a ``.npz`` file written by :func:`save_costmap`.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        A ``CostMap`` instance.
    """
    data = np.load(str(path), allow_pickle=False)
    grid = data["grid"].astype(np.float32)
    origin = data["origin"].astype(np.float64)
    resolution = float(data["resolution"])
    return CostMap(grid=grid, cost=grid, origin=origin, resolution=resolution)


def export_costmap_geotiff(
    cost_map: CostMap,
    path: str | Path,
    crs: str = "EPSG:32635",
) -> None:
    """Export a ``CostMap`` as a single-band GeoTIFF for use in GIS software.

    Requires the ``rasterio`` package.

    Args:
        cost_map: The cost map to export.
        path: Output ``.tif`` file path.
        crs: Coordinate reference system string (default: UTM zone 35N,
             typical for Finnish datasets).
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise ImportError(
            "GeoTIFF export requires rasterio. Install with: pip install rasterio"
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grid = cost_map.grid
    nrows, ncols = grid.shape
    res = cost_map.resolution

    # rasterio expects top-left origin; our grid has bottom-left origin.
    # Flip vertically so row 0 = northernmost row.
    flipped = np.flipud(grid)

    transform = from_origin(
        west=cost_map.origin[0],
        north=cost_map.origin[1] + nrows * res,
        xsize=res,
        ysize=res,
    )

    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=flipped.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(flipped, 1)
