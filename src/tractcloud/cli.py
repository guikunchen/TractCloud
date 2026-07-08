"""TractCloud command-line interface."""

import argparse
import logging
import sys

from .progress import ProgressReporter, NullReporter


def main():
    parser = argparse.ArgumentParser(
        description="TractCloud: Registration-free tractography parcellation",
        epilog=(
            "Example: tractcloud -i brain.vtk -o results/\n"
            "  Parcellates whole-brain tractography into 42 anatomical tracts."
        ),
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input tractography file (.vtk or .vtp)")
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Output directory")
    parser.add_argument(
        "--mrb", action="store_true",
        help="Also create a Slicer-compatible MRB file")
    parser.add_argument(
        "--include-other", action="store_true",
        help="Include 'Other' bundle for unclassified streamlines")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Compute device (default: auto)")
    parser.add_argument(
        "--batch-size", type=int, default=2048,
        help="Inference batch size (default: 2048)")
    parser.add_argument(
        "--num-points", type=int, default=15,
        help="Points per streamline resampling (default: 15)")
    parser.add_argument(
        "--data-dir",
        help="Override model data cache directory")
    parser.add_argument(
        "--hemisphere-atlas-dir",
        help=(
            "Run Hemisphere export mode using an ORG-Atlases-1.1.1 path. If passed, runs TractCloud inference, uses raw fine cluster_preds for hemisphere assignment, runs WMA registration, and writes only hemisphere outputs."
        ))
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output on stdout")

    args = parser.parse_args()

    if args.mrb and args.hemisphere_atlas_dir:
        parser.error("--mrb cannot be used with --hemisphere-atlas-dir")

    # Configure logging to stderr (keep stdout clean for JSON progress)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    reporter = NullReporter() if args.quiet else ProgressReporter()

    device = None
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda:0"
    # else auto (None)

    from .pipeline import TractCloudPipeline

    pipeline = TractCloudPipeline(
        reporter=reporter,
        device=device,
        batch_size=args.batch_size,
        num_points=args.num_points,
        include_other=args.include_other,
        data_dir=args.data_dir,
    )

    pipeline.run_on_file(
        args.input,
        args.output_dir,
        create_mrb=args.mrb,
        hemisphere_atlas_dir=args.hemisphere_atlas_dir,
    )


if __name__ == "__main__":
    main()
