# forest-panoptic-nav

Panoptic segmentation in forest environments for autonomous navigation, built on the [FinnWoodlands](https://doi.org/10.1016/j.dib.2023.109700) dataset (2023).

FinnWoodlands provides 5,170 stereo RGB frames and corresponding LiDAR point clouds collected with a backpack-mounted Ouster OS1 + ZED2 setup in Finnish forests. 300 frames include panoptic annotations covering:

- **Things** (instance): Spruce, Birch, Pine tree trunks
- **Stuff** (semantic): Ground, Track, Lake

This tool segments the scene into these classes, fuses LiDAR and camera data, and generates traversability cost maps for navigation.

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

## Dataset layout

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

## Architecture

- **loader** — FinnWoodlands data reader (point clouds, stereo images, calibration, annotations)
- **fusion** — LiDAR-to-camera projection and per-point RGB feature extraction
- **segmentation** — Panoptic segmentation with a stub ML model and height-based heuristic fallback
- **traversability** — 2D cost map generation with configurable per-class costs and spatial smoothing
- **cli** — Click-based command-line interface tying it all together

## License

MIT
