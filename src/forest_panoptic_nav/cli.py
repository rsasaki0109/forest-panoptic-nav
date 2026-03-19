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
@click.option(
    "--method", "-m",
    type=click.Choice(["zero_shot", "heuristic", "ml"]),
    default="zero_shot",
    help="Segmentation method (default: zero_shot).",
)
@click.option(
    "--visualize", "-v",
    type=click.Path(path_type=Path),
    default=None,
    help="Save visualization PNG to this path.",
)
def segment(data_dir: Path, frame_id: int | None, output: Path | None, use_fusion: bool, method: str, visualize: Path | None) -> None:
    """Run panoptic segmentation on FinnWoodlands data.

    DATA_DIR is the root directory of the FinnWoodlands dataset.
    """
    from .loader import FinnWoodlandsLoader
    from .segmentation import PanopticSegmenter
    from .fusion import LidarRgbFusion

    output = output or Path("output/segmentation")
    output.mkdir(parents=True, exist_ok=True)

    loader = FinnWoodlandsLoader(data_dir)
    segmenter = PanopticSegmenter(method=method)
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

        if visualize is not None:
            from .zero_shot import visualize_segmentation
            visualize.mkdir(parents=True, exist_ok=True)
            viz_path = visualize / f"frame_{fid:06d}_segmentation.png"
            visualize_segmentation(result, str(viz_path))
            click.echo(f"  Visualization saved to {viz_path}")

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
@click.argument("segmentation_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("annotation_dir", type=click.Path(exists=True, path_type=Path))
def evaluate(segmentation_dir: Path, annotation_dir: Path) -> None:
    """Evaluate segmentation against ground-truth annotations.

    SEGMENTATION_DIR contains .npz files from the 'segment' command.
    ANNOTATION_DIR contains ground-truth .npz files with semantic_labels.
    """
    import numpy as np
    from .evaluation import evaluate_segmentation

    seg_files = sorted(segmentation_dir.glob("frame_*.npz"))
    if not seg_files:
        click.echo("No segmentation files found.")
        return

    all_preds = []
    all_gt = []

    for seg_file in seg_files:
        frame_id = seg_file.stem.replace("frame_", "")
        ann_file = annotation_dir / f"{frame_id}.npz"
        if not ann_file.exists():
            continue
        seg_data = np.load(seg_file)
        ann_data = np.load(ann_file)
        all_preds.append(seg_data["semantic_labels"])
        all_gt.append(ann_data["semantic_labels"])

    if not all_preds:
        click.echo("No matching annotation files found.")
        return

    preds = np.concatenate(all_preds)
    gt = np.concatenate(all_gt)
    result = evaluate_segmentation(preds, gt)
    click.echo(result.summary())


@cli.command()
@click.argument("cost_map_file", type=click.Path(exists=True, path_type=Path))
@click.option("--start", "-s", required=True, help="Start position as x,y (e.g. 1.0,1.0).")
@click.option("--goal", "-g", required=True, help="Goal position as x,y (e.g. 19.0,19.0).")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Save path visualization PNG.")
def plan(cost_map_file: Path, start: str, goal: str, output: Path | None) -> None:
    """Plan a path on a traversability cost map using A* search.

    COST_MAP_FILE is a .npz file produced by the 'traversability' command.
    """
    from .costmap_io import load_costmap
    from .path_planner import plan_path, plot_path

    cost_map = load_costmap(cost_map_file)

    def _parse_xy(s: str) -> tuple[float, float]:
        parts = s.split(",")
        if len(parts) != 2:
            raise click.BadParameter(f"Expected x,y but got '{s}'")
        return float(parts[0]), float(parts[1])

    start_xy = _parse_xy(start)
    goal_xy = _parse_xy(goal)

    result = plan_path(cost_map, start_xy, goal_xy)

    if result.is_feasible:
        click.echo(f"Path found: {result.num_waypoints} waypoints, "
                    f"distance {result.distance:.1f}m, cost {result.total_cost:.2f}")
    else:
        click.echo("No feasible path found between start and goal.")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        plot_path(cost_map, result, str(output))
        click.echo(f"Visualization saved to {output}")


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
