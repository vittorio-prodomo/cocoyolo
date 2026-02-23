"""Command-line interface for cocoyolo.

Provides three entry points:

- ``cocoyolo`` — unified CLI with ``coco2yolo`` / ``yolo2coco`` subcommands
- ``coco2yolo`` — direct shortcut for COCO → YOLO conversion
- ``yolo2coco`` — direct shortcut for YOLO → COCO conversion
"""

import argparse
import logging
from pathlib import Path


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", type=Path, help="Input dataset directory.")
    p.add_argument("output", type=Path, help="Output dataset directory.")
    p.add_argument(
        "--image-mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help=(
            "How to transfer images to the output directory. "
            "'copy' makes full file copies (default). "
            "'symlink' creates relative symbolic links (saves disk space; "
            "requires OS support). "
            "'hardlink' creates hard links (saves disk space; "
            "requires same filesystem)."
        ),
    )
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output.")


def _add_coco2yolo_args(p: argparse.ArgumentParser) -> None:
    _add_common_args(p)
    p.add_argument(
        "--task",
        choices=["auto", "detect", "segment"],
        default="auto",
        help=(
            "YOLO output task type. "
            "'auto' detects from annotations and errors if mixed. "
            "'detect' forces bounding-box output. "
            "'segment' forces polygon output "
            "(default: auto)."
        ),
    )
    p.add_argument(
        "--contour-approx",
        type=float,
        default=0.001,
        help="Contour approximation factor (default: 0.001).",
    )
    p.add_argument(
        "--hole-strategy",
        choices=["fill", "bridge"],
        default="bridge",
        help=(
            "How to handle holes in RLE masks. "
            "'fill' ignores holes; 'bridge' preserves them via "
            "inverse bridges (default: bridge)."
        ),
    )
    p.add_argument(
        "--disjoint-strategy",
        choices=["split", "bridge"],
        default="bridge",
        help=(
            "How to handle disjoint regions. "
            "'split' writes each as a separate annotation; "
            "'bridge' connects them with zero-width bridges "
            "(default: bridge)."
        ),
    )


def _add_yolo2coco_args(p: argparse.ArgumentParser) -> None:
    _add_common_args(p)
    p.add_argument(
        "--keep-zero-indexing",
        action="store_true",
        help=(
            "Keep 0-based category IDs (YOLO convention). "
            "By default, IDs are shifted to 1-based (COCO convention)."
        ),
    )


def _run_coco2yolo(args: argparse.Namespace) -> None:
    from .coco_to_yolo import convert

    info, stats = convert(
        args.input,
        args.output,
        contour_approx_factor=args.contour_approx,
        hole_strategy=args.hole_strategy,
        disjoint_strategy=args.disjoint_strategy,
        task=args.task,
        image_mode=args.image_mode,
        verbose=not args.quiet,
    )

    print()
    print("Conversion complete (COCO → YOLO).")
    print(f"  Classes:  {len(info.class_names)}")
    print(f"  Images:   {info.total_images}")
    print(f"  Splits:   {info.split_names}")
    print(f"  Output:   {args.output}")
    print()
    print(stats.format_summary(args.hole_strategy, args.disjoint_strategy))


def _run_yolo2coco(args: argparse.Namespace) -> None:
    from .yolo_to_coco import convert_yolo_to_coco

    info, stats = convert_yolo_to_coco(
        args.input,
        args.output,
        keep_zero_indexing=args.keep_zero_indexing,
        image_mode=args.image_mode,
        verbose=not args.quiet,
    )

    print()
    print("Conversion complete (YOLO → COCO).")
    print(f"  Classes:  {len(info.class_names)}")
    print(f"  Images:   {info.total_images}")
    print(f"  Splits:   {info.split_names}")
    print(f"  Output:   {args.output}")
    print()
    print(stats.format_summary())


# ------------------------------------------------------------------
# Entry points
# ------------------------------------------------------------------


def main(argv=None):
    """Unified ``cocoyolo`` CLI with subcommands."""
    p = argparse.ArgumentParser(
        prog="cocoyolo",
        description=(
            "Convert between COCO and YOLO instance segmentation formats "
            "with proper handling of holes and disjoint regions."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    c2y = sub.add_parser(
        "coco2yolo",
        help="Convert COCO → YOLO.",
        description="Convert a COCO dataset to YOLO format.",
    )
    _add_coco2yolo_args(c2y)

    y2c = sub.add_parser(
        "yolo2coco",
        help="Convert YOLO → COCO.",
        description="Convert a YOLO dataset to COCO format.",
    )
    _add_yolo2coco_args(y2c)

    args = p.parse_args(argv)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level, format="%(name)s - %(levelname)s - %(message)s"
    )

    if args.command == "coco2yolo":
        _run_coco2yolo(args)
    elif args.command == "yolo2coco":
        _run_yolo2coco(args)


def main_coco2yolo(argv=None):
    """Direct ``coco2yolo`` CLI entry point."""
    p = argparse.ArgumentParser(
        prog="coco2yolo",
        description=(
            "Convert a COCO instance segmentation dataset to YOLO format "
            "with proper handling of holes and disjoint regions."
        ),
    )
    _add_coco2yolo_args(p)
    args = p.parse_args(argv)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level, format="%(name)s - %(levelname)s - %(message)s"
    )
    _run_coco2yolo(args)


def main_yolo2coco(argv=None):
    """Direct ``yolo2coco`` CLI entry point."""
    p = argparse.ArgumentParser(
        prog="yolo2coco",
        description="Convert a YOLO dataset to COCO format.",
    )
    _add_yolo2coco_args(p)
    args = p.parse_args(argv)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level, format="%(name)s - %(levelname)s - %(message)s"
    )
    _run_yolo2coco(args)


if __name__ == "__main__":
    main()
