"""Command-line interface for forest-panoptic-nav."""

from pathlib import Path

import click


@click.group()
@click.version_option()
def cli() -> None:
    """Panoptic segmentation in forest environments for autonomous navigation."""


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--frame-id", "-f", type=int, default=None, help="Process a single frame by ID.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output directory.")
@click.option("--use-fusion/--no-fusion", default=True, help="Enable LiDAR-RGB fusion.")
def segment(data_dir: Path, frame_id: int | None, output: Path | None, use_fusion: bool) -> None:
    """Run panoptic segmentation on FinnWoodlands data.

    DATA_DIR is the root directory of the FinnWoodlands dataset.
    """
    from .loader import FinnWoodlandsLoader
    from .segmentation import PanopticSegmenter
    from .fusion import LidarRgbFusion

    output = output or Path("output/segmentation")
    output.mkdir(parents=True, exist_ok=True)

    loader = FinnWoodlandsLoader(data_dir)
    segmenter = PanopticSegmenter()
    fusion = LidarRgbFusion() if use_fusion else None

    frame_ids = [frame_id] if frame_id is not None else loader.list_frame_ids()
    click.echo(f"Processing {len(frame_ids)} frame(s)...")

    for fid in frame_ids:
        sample = loader.load_sample(fid)
        if fusion is not None:
            fused = fusion.fuse(sample.point_cloud, sample.left_image, sample.calibration)
            result = segmenter.predict(fused.point_cloud, fused.features)
        else:
            result = segmenter.predict(sample.point_cloud)
        result.save(output / f"frame_{fid:06d}.npz")
        click.echo(f"  Frame {fid}: {result.num_instances} instances, {result.num_semantic_classes} semantic classes")

    click.echo(f"Results saved to {output}")


@cli.command()
@click.argument("segmentation_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output directory.")
@click.option("--resolution", "-r", type=float, default=0.1, help="Grid resolution in meters.")
def traversability(segmentation_dir: Path, output: Path | None, resolution: float) -> None:
    """Generate traversability map from segmentation results.

    SEGMENTATION_DIR contains the output of the 'segment' command.
    """
    import numpy as np
    from .traversability import TraversabilityMapper

    output = output or Path("output/traversability")
    output.mkdir(parents=True, exist_ok=True)

    mapper = TraversabilityMapper(resolution=resolution)
    seg_files = sorted(segmentation_dir.glob("frame_*.npz"))
    click.echo(f"Generating traversability maps for {len(seg_files)} frame(s)...")

    for seg_file in seg_files:
        data = np.load(seg_file, allow_pickle=True)
        points = data["points"]
        semantic_labels = data["semantic_labels"]
        cost_map = mapper.compute_cost_map(points, semantic_labels)
        out_path = output / seg_file.with_suffix(".npz").name
        np.savez(out_path, grid=cost_map.grid, cost=cost_map.cost, origin=cost_map.origin, resolution=resolution)
        click.echo(f"  {seg_file.name}: grid {cost_map.grid.shape}, traversable {cost_map.traversable_ratio:.1%}")

    click.echo(f"Traversability maps saved to {output}")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--mode", "-m", type=click.Choice(["segmentation", "traversability", "overlay"]), default="segmentation")
@click.option("--save", "-s", type=click.Path(path_type=Path), default=None, help="Save figure to file instead of showing.")
def visualize(input_path: Path, mode: str, save: Path | None) -> None:
    """Visualize segmentation or traversability results.

    INPUT_PATH is a .npz result file or a directory of results.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    files = sorted(input_path.glob("*.npz")) if input_path.is_dir() else [input_path]

    for filepath in files:
        data = np.load(filepath, allow_pickle=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        if mode == "traversability" and "cost" in data:
            im = ax.imshow(data["cost"], cmap="RdYlGn_r", origin="lower")
            plt.colorbar(im, ax=ax, label="Cost")
            ax.set_title(f"Traversability: {filepath.stem}")
        elif "semantic_labels" in data:
            points = data["points"]
            labels = data["semantic_labels"]
            scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=1)
            plt.colorbar(scatter, ax=ax, label="Class")
            ax.set_title(f"Segmentation: {filepath.stem}")
            ax.set_aspect("equal")
        else:
            click.echo(f"Skipping {filepath.name}: unsupported data format for mode '{mode}'")
            plt.close(fig)
            continue

        if save is not None:
            save.mkdir(parents=True, exist_ok=True)
            fig.savefig(save / f"{filepath.stem}_{mode}.png", dpi=150, bbox_inches="tight")
            click.echo(f"  Saved {filepath.stem}_{mode}.png")
        else:
            plt.show()
        plt.close(fig)
