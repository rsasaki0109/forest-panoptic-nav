# forest-panoptic-nav

[![CI](https://github.com/rsasaki0109/forest-panoptic-nav/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/forest-panoptic-nav/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Forest Traversability Mapping + Path Planning** from LiDAR point clouds.

Generate traversability cost maps from semantically segmented forest point clouds and plan obstacle-aware paths through them with A* search. No other open-source tool produces traversability cost grids directly from forest LiDAR data.

Built on the [FinnWoodlands](https://doi.org/10.1016/j.dib.2023.109700) dataset (2023) -- 5,170 stereo RGB frames and corresponding LiDAR scans collected in Finnish forests with a backpack-mounted Ouster OS1 + ZED2 setup. 300 frames include panoptic annotations covering tree trunks (Spruce, Birch, Pine), Ground, Track, and Lake.

**No training required.** The default zero-shot pipeline uses RANSAC ground estimation, DBSCAN clustering, and geometric shape analysis -- no ML model or GPU needed.

## Key Features

- **Traversability cost maps** -- Convert labelled forest point clouds into 2D cost grids where each cell encodes how difficult it is to traverse (ground/track = low cost, trees/water = impassable)
- **A\* path planning** -- Find the lowest-cost route through the forest on the generated cost grid, with 8-connected search and obstacle avoidance
- **Cost map I/O** -- Save/load cost maps as `.npz`; optional GeoTIFF export for GIS integration
- **Panoptic segmentation** -- Zero-shot (RANSAC + DBSCAN + shape analysis), heuristic, or ML-based methods
- **LiDAR-RGB fusion** -- Project LiDAR points onto stereo images for per-point colour features
- **Evaluation** -- Per-class IoU, mIoU, confusion matrix against ground-truth annotations

## Architecture

```
                      FinnWoodlands Dataset
                      /                  \
              Point Clouds           Stereo Images
                  |                       |
                  v                       v
         +----------------+     +------------------+
         |  loader.py     |     |  loader.py       |
         |  FinnWoodlands |     |  CalibrationData |
         |  Loader        |     |  AnnotationData  |
         +-------+--------+     +--------+---------+
                 |                        |
                 v                        v
            +----+------------------------+----+
            |         fusion.py                |
            |     LidarRgbFusion               |
            |  (LiDAR-to-camera projection,    |
            |   per-point RGB features)        |
            +---------------+------------------+
                            |
                            v
            +---------------+------------------+
            |       segmentation.py            |
            |     PanopticSegmenter            |
            |  method = zero_shot | heuristic  |
            |            | ml                  |
            +------+--------+---------+--------+
                   |        |         |
         +---------+  +-----+---+  +--+--------+
         |zero_shot|  |heuristic|  | ml model  |
         |  .py    |  |(builtin)|  |(user ckpt)|
         +---------+  +---------+  +-----------+
                   \        |         /
                    v       v        v
                      PanopticResult
                            |
                            v
            +---------------+------------------+
            |      traversability.py           |
            |    TraversabilityMapper           |
            |  (2D cost map from labeled       |
            |   point cloud, smoothing)        |
            +---------------+------------------+
                            |
                            v  CostMap
            +---------------+------------------+
            |      path_planner.py             |
            |    A* search on cost grid        |
            |  (8-connected, obstacle-aware)   |
            +----------------------------------+
                            |
                            v  PathResult
                     Waypoints + Viz
```

### Zero-Shot Pipeline (default)

The `zero_shot` method requires no training data and works out of the box:

1. **Ground plane estimation** -- RANSAC on the lowest 30% of points to fit a plane
2. **Ground removal** -- Points below 0.3m above the plane are classified as Ground/Track
3. **Track detection** -- Grid-based local flatness analysis identifies smooth path regions
4. **Height filtering** -- Points 0.5m--8m above ground are trunk candidates
5. **DBSCAN clustering** -- Groups nearby above-ground points into individual objects
6. **Shape classification** -- Each cluster's aspect ratio determines its class:
   - Tall + thin (aspect ratio > 1.5, radius < 0.5m) = tree trunk
   - Species assigned by height: Pine (tallest) > Spruce > Birch (shortest)

## Semantic Class Definitions

| ID | Class      | Category | Cost | Traversable | Description                        |
|----|------------|----------|------|-------------|------------------------------------|
| 0  | Unlabeled  | --       | 0.5  | Maybe       | Unknown / uncertain                |
| 1  | Ground     | Stuff    | 0.1  | Yes         | Natural ground surface             |
| 2  | Track      | Stuff    | 0.0  | Yes         | Forest trail or path (best surface)|
| 3  | Lake       | Stuff    | 1.0  | No          | Water body (impassable)            |
| 4  | Spruce     | Thing    | 1.0  | No          | Spruce tree trunk (obstacle)       |
| 5  | Birch      | Thing    | 1.0  | No          | Birch tree trunk (obstacle)        |
| 6  | Pine       | Thing    | 1.0  | No          | Pine tree trunk (obstacle)         |

**Stuff classes** (1-3) receive semantic-only labels. **Thing classes** (4-6) receive both semantic labels and per-instance IDs, enabling individual tree trunk identification.

## Installation

```bash
pip install -e .
```

For GPU support, install PyTorch with CUDA first: https://pytorch.org/get-started/locally/

## Usage

### Generate traversability cost map

```bash
# Segment first, then build cost map
forest-panoptic-nav segment /path/to/finnwoodlands --output output/seg
forest-panoptic-nav traversability output/seg --output output/trav --resolution 0.1
```

### Plan a path on the cost map

```bash
forest-panoptic-nav plan output/trav/frame_000000.npz \
    --start 1.0,1.0 --goal 19.0,19.0 \
    -o path.png
```

### Run panoptic segmentation

```bash
# Zero-shot (default, no training required)
forest-panoptic-nav segment /path/to/finnwoodlands --output output/seg

# Or explicitly choose a method
forest-panoptic-nav segment /path/to/finnwoodlands --method zero_shot --output output/seg
forest-panoptic-nav segment /path/to/finnwoodlands --method heuristic --output output/seg
```

### Evaluate against ground truth

```bash
forest-panoptic-nav evaluate output/seg /path/to/finnwoodlands/annotations
```

Output includes per-class IoU, mIoU, and overall accuracy.

### Visualize results

```bash
forest-panoptic-nav visualize output/seg --mode segmentation
forest-panoptic-nav visualize output/trav --mode traversability
```

## Evaluation Metrics

The `evaluate` command and `evaluation.py` module compute:

- **mIoU** (mean Intersection over Union) -- averaged over all present classes
- **Per-class IoU** -- IoU(c) = TP / (TP + FP + FN) for each class
- **Per-class accuracy** -- TP / (TP + FN) for each class
- **Overall accuracy** -- total correct predictions / total points
- **Confusion matrix** -- (7 x 7) matrix for all semantic classes

## Dataset Layout

The loader expects the following directory structure:

```
dataset_root/
    calibration/
        calib.npz
    point_clouds/
        000000.npy  (or .pcd / .ply)
        ...
    images/
        left/
            000000.png
            ...
        right/
            000000.png
            ...
    annotations/
        000000.npz
        ...
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
