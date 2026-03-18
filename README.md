# forest-panoptic-nav

[![CI](https://github.com/rsasaki0109/forest-panoptic-nav/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/forest-panoptic-nav/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Panoptic segmentation in forest environments for autonomous navigation, built on the [FinnWoodlands](https://doi.org/10.1016/j.dib.2023.109700) dataset (2023).

FinnWoodlands provides 5,170 stereo RGB frames and corresponding LiDAR point clouds collected with a backpack-mounted Ouster OS1 + ZED2 setup in Finnish forests. 300 frames include panoptic annotations covering:

- **Things** (instance): Spruce, Birch, Pine tree trunks
- **Stuff** (semantic): Ground, Track, Lake

This tool segments the scene into these classes, fuses LiDAR and camera data, and generates traversability cost maps for navigation.

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
            |  (ML model or heuristic          |
            |   height-based fallback)         |
            +---------------+------------------+
                            |
                            v  PanopticResult
            +---------------+------------------+
            |      traversability.py           |
            |    TraversabilityMapper           |
            |  (2D cost map from labeled       |
            |   point cloud, smoothing)        |
            +---------------+------------------+
                            |
                            v  CostMap
                     Path Planning
```

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

### Run panoptic segmentation

```bash
forest-panoptic-nav segment /path/to/finnwoodlands --output output/seg
```

### Generate traversability map

```bash
forest-panoptic-nav traversability output/seg --output output/trav --resolution 0.1
```

### Visualize results

```bash
forest-panoptic-nav visualize output/seg --mode segmentation
forest-panoptic-nav visualize output/trav --mode traversability
```

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
