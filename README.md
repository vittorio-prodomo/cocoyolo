# cocoyolo

Bidirectional **COCO (json) ↔ YOLO Ultralytics (txt)** label format converter for **object detection** and **instance segmentation** datasets.

Works out of the box for the simple cases, and handles the hard ones too: **RLE masks**, **holes**, **disjoint regions**, and **mixed annotation types** — all the edge cases that other tools silently break on.

## The Two Formats at a Glance

### Object detection

In COCO, a bounding box is stored as absolute pixel coordinates `[x, y, w, h]` (top-left origin) inside a JSON annotation object:

```json
{
  "id": 1,
  "image_id": 42,
  "category_id": 3,
  "bbox": [120.0, 45.5, 200.0, 310.0],
  "area": 62000.0,
  "iscrowd": 0
}
```

In YOLO Ultralytics, the same box becomes a single line in a `.txt` file — class index followed by the **normalised** centre coordinates, width, and height (all divided by image dimensions) `[xc, yc, w, h]`:

```
2 0.3143 0.3226 0.2857 0.4968
```

The mapping is one-to-one: `class_id xc yc w h`.

### Instance segmentation

In COCO for instance segmentation, the annotation object keeps the same fields as above (`id`, `image_id`, `category_id`, `bbox`, `area`, `iscrowd`) but adds a `segmentation` field describing the mask. This field can take two forms.

As a **polygon list** — a list of lists of `[x, y]` vertex pairs in absolute pixels, where each inner list is a separate polygon.  A single annotation can carry **multiple polygons** (disjoint parts of the same object), but it doesn't natively support the concept of "holes" in polygons (other tools/formats may use conventions on the winding order to also model holes, but not COCO):

```json
{
  // ... same fields as above ...
  "segmentation": [
    [100.0, 50.0, 200.0, 50.0, 200.0, 180.0, 100.0, 180.0], // first polygon
    [250.0, 60.0, 320.0, 60.0, 320.0, 170.0, 250.0, 170.0]  // second polygon
  ]
}
```

Or as an **RLE-encoded binary mask** — the full rasterized mask compressed in Run-Length Encoding, which naturally supports holes and arbitrary shapes:

```json
{
  // ... same fields as above ...
  "segmentation": {
    "counts": "Pfh01o10O10O001N1O2M2N1O2N1O1...",
    "size": [480, 640]
  }
}
```

In YOLO Ultralytics, only the "polygon" form exists and it gets condensed into a single text line — class index followed by **normalised** `x y` vertex pairs forming one polygon:

```
2 0.1429 0.0806 0.2857 0.0806 0.2857 0.2903 0.1429 0.2903
```

The mapping is strictly **one line → one polygon → one object**.  There is currently no way to represent multiple disjoint polygons per object, and no concept of holes (inner rings) within a polygon.  This is the fundamental limitation that `cocoyolo` is designed to work around.

## Why Another Converter?

COCO-to-YOLO conversion is one of those tasks that seems trivial until you try it on a real-world dataset.  If all your annotations are simple polygon outlines, or maybe just bounding boxes, any converter will likely do.  The trouble starts when your dataset contains:

- **RLE-encoded masks** (common when exporting mask/brush tool shapes from CVAT, SA, or in the original COCO dataset itself)
- **Holes** in masks (e.g., a donut, or an object with a window through it)
- **Disjoint regions** in a single annotation (e.g., an occluded object visible in two separate areas)
- **A mix of bounding boxes and segmentation masks** in the same JSON file

Every major ML framework or Data management library ships some version of this conversion.  None of them handled all of the above correctly when I needed them to:

| Tool | RLE masks | Disjoint regions | Holes | Mixed bbox/seg |
|------|-----------|-------------------|-------|----------------|
| **Ultralytics** `convert_coco` | Silently skipped ([#4931](https://github.com/ultralytics/ultralytics/issues/4931)) | Merged via lossy bridge | Filled in — no support ([#19153](https://github.com/ultralytics/ultralytics/issues/19153)) | Can produce mixed output |
| **FiftyOne** | Imported as masks, dropped on YOLO export until late 2025 ([#6421](https://github.com/voxel51/fiftyone/issues/6421)) | Was incorrectly chained; now split into separate instances | No positive/negative space concept | Was bbox-only for masks |
| **Datumaro** (upstream) | No segmentation export at all ([#1114](https://github.com/open-edge-platform/datumaro/issues/1114)) | N/A | N/A | Bbox-only |
| **Datumaro** (CVAT fork) | Silently skipped | Exported as separate instances | Not handled | — |
| **Roboflow** Supervision | Added mid-2024; previously crashed | Crashes the loader ([#1209](https://github.com/roboflow/supervision/issues/1209)) | Not documented | — |
| **cocoyolo** | Fully decoded and contoured | Configurable: bridge or split | Configurable: bridge or fill | Auto-detected; enforced uniform output |

So I decided to design a new, powerful and flexible converter, `cocoyolo`, specifically to fill these gaps.

### What cocoyolo does differently

- **Full RLE support.**  RLE masks (compressed or uncompressed) are decoded to binary masks, contoured with `cv2.findContours`, and converted to polygons.  Nothing is silently skipped.

- **Hole-aware conversion.**  Using `cv2.RETR_CCOMP` hierarchy, outer contours and their child holes are identified.  You choose what to do with them: `--hole-strategy bridge` preserves holes via zero-width inverse bridges (lossless when rasterised), or `--hole-strategy fill` discards them.

- **Disjoint region handling.**  Multiple connected components in a single annotation are detected and handled via `--disjoint-strategy bridge` (connect them with a greedy nearest-neighbour chain of zero-width bridges into one polygon) or `--disjoint-strategy split` (emit each region as a separate YOLO annotation).

- **Uniform output guarantee.**  YOLO expects either all bounding boxes or all polygons — never both in the same dataset.  As a safety check, `cocoyolo` pre-scans the json annotations, auto-detects the task type, and raises a clear error on mixed datasets with explicit instructions (`--task detect` to force bounding boxes, `--task segment` to keep only segmentation).

- **Conversion statistics.**  Every run prints exactly what happened: how many annotations of each type were processed, how many edge cases were encountered, and how each was resolved.

## Installation

```bash
pip install .
```

Or in development mode:

```bash
pip install -e .
```

For faster image size reading (header-only via pyvips):

```bash
pip install -e ".[fast]"
```

## Quick Start

### Command line

```bash
# COCO → YOLO (default: bridge holes + bridge disjoint regions)
coco2yolo path/to/coco path/to/yolo

# COCO → YOLO with explicit strategies
coco2yolo path/to/coco path/to/yolo --hole-strategy fill --disjoint-strategy split

# Force bounding-box output (for mixed datasets)
coco2yolo path/to/coco path/to/yolo --task detect

# YOLO → COCO
yolo2coco path/to/yolo path/to/coco

# Unified CLI with subcommands
cocoyolo coco2yolo path/to/coco path/to/yolo
cocoyolo yolo2coco path/to/yolo path/to/coco
```

### Python API

```python
from cocoyolo import coco_to_yolo, yolo_to_coco

# COCO → YOLO (default strategies: bridge both)
info, stats = coco_to_yolo("path/to/coco", "path/to/yolo")

# COCO → YOLO with explicit strategies
info, stats = coco_to_yolo(
    "path/to/coco",
    "path/to/yolo",
    hole_strategy="bridge",      # or "fill"
    disjoint_strategy="bridge",  # or "split"
    task="auto",                 # or "detect", "segment"
    workers=8,                   # default: all CPU cores
)

# YOLO → COCO
info, stats = yolo_to_coco("path/to/yolo", "path/to/coco")
```

### Example output

```
Conversion complete (COCO → YOLO).
  Classes:  80
  Images:   5000
  Splits:   ['val']
  Output:   path/to/yolo

  YOLO task type:        segment
  Annotations processed: 36781
    Polygon (single):    32813
    Polygon (disjoint):  3522
    RLE (simple):        25
    RLE (with holes):    236
    RLE (disjoint):      185
  Edge cases: 3943
    Holes (bridge):    396 contour(s) bridged
    Disjoint (bridge): 3931 annotation(s) bridged
```

## Supported Dataset Layouts

All layouts are **auto-detected** — just point the tool at the root directory.

### COCO layouts

`cocoyolo` doesn't assume a rigid directory structure for images.  Each COCO JSON file lists image filenames in its `"images"` array; the loader resolves each filename by trying — in order — the exact relative path stored in the JSON, the same path under an `images/` subdirectory, and finally a recursive leaf-name search (matching by basename only, ignoring any directory prefix).  This means your images can live at any depth, and `file_name` in the JSON can be a bare filename or a relative path — both work. I tried to be as flexible as possible to save people from the torture of always going and having to micro-adjust names and directory structures to always "make the tool on duty happy".

As practical examples of what surely works, the three common COCO layouts you'll encounter in the wild:

**COCO-A** — the standard COCO layout.  Annotation JSONs live in `annotations/`, images are split into subdirectories under `images/`.  This is what you get from the [official COCO dataset](https://cocodataset.org/) or most tools that follow the COCO standard.

```
dataset/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
└── images/
    ├── train/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    └── val/
        ├── 000101.jpg
        └── ...
```

**COCO-B** — the Roboflow layout.  Each split is a self-contained folder with its own `_annotations.coco.json` and images placed alongside it.  This is what you get when exporting a dataset from [Roboflow](https://roboflow.com/) in COCO format, or what their [RF-DETR model expects](https://rfdetr.roboflow.com/learn/train/#dataset-structure).

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── val/
    ├── _annotations.coco.json
    ├── img_101.jpg
    └── ...
```

**COCO-C** — flat layout with no split subdirectories.  A single `annotations/` folder and a single `images/` folder.  Common for small or single-split datasets (similar to what you get when you export from CVAT).

```
dataset/
├── annotations/
│   └── instances.json
└── images/
    ├── photo_a.png
    ├── photo_b.png
    └── ...
```

### YOLO layouts

The primary discovery mechanism for YOLO datasets is the **data YAML file** (`data.yaml`, `dataset.yaml`, or any file matching `data*.yaml` in the dataset directory, or a custom-named yaml directly provided in input).  This file explicitly lists per-split image sub-paths, and `cocoyolo` reads them directly — there is no hardcoded assumption about where images live.

**Label paths** are derived from image paths by replacing the first occurrence of `images` with `labels` in the path — exactly as Ultralytics pipelines do internally.  This is important to know: it means your label directories must mirror your image directories with `images` swapped for `labels`.

Example `data.yaml`:

```yaml
train: images/train
val: images/val
nc: 80
names:
  0: person
  1: bicycle
  # ...
```

The two most common directory structures you'll encounter:

**YOLO-A** — the standard Ultralytics layout.  Images and labels live under top-level `images/` and `labels/` directories, each split into subdirectories by split name.

```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── val/
│       ├── 000101.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── 000001.txt
    │   ├── 000002.txt
    │   └── ...
    └── val/
        ├── 000101.txt
        └── ...
```

**YOLO-B** — split-first layout.  Each split is a self-contained folder with its own `images/` and `labels/` subdirectories.  Also used by some Roboflow YOLO exports.

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── 000001.jpg
│   │   └── ...
│   └── labels/
│       ├── 000001.txt
│       └── ...
└── val/
    ├── images/
    │   ├── 000101.jpg
    │   └── ...
    └── labels/
        ├── 000101.txt
        └── ...
```


#### Passing a YAML file directly

For the `yolo2coco` direction, the input can be either a directory (the usual case) or a **direct path to a YAML file** that lives anywhere on disk:

```bash
yolo2coco /path/to/custom_config.yaml /path/to/output
```

When a YAML file is given, the dataset root is determined from the `path` entry inside it (resolved relative to the YAML file's own directory), falling back to the YAML's parent directory when `path` is absent.

#### A note on the `path` entry and Ultralytics quirks

This `path` entry in the YOLO data YAML format is supposed to specify the dataset root directory. In principle, this allows decoupling the YAML file from the dataset:

```yaml
path: /data/my_dataset        # dataset root (absolute or relative)
train: images/train
val: images/val
# ...
```

`cocoyolo` honors this field: when `path` is present, split sub-paths (`train`, `val`, etc.) are resolved relative to it.

That said, **I strongly recommend keeping your `data.yaml` inside the dataset root directory and either omitting the `path` entry entirely or setting it to `path: .`**.  The reason is that the Ultralytics framework itself is notoriously finicky about how it resolves the `path` field.  Internally, Ultralytics prepends a global `datasets_dir` setting (defaulting to `~/datasets`) to any relative `path` value — which silently produces wrong paths when you're working outside that directory.  This has been the source of a long trail of user-reported bugs:

- Relative `path` values get concatenated with `datasets_dir`, producing doubled or tripled paths ([#16911](https://github.com/ultralytics/ultralytics/issues/16911))
- `path: ./my_dataset` resolves against `~/datasets` instead of the YAML's location ([#2221](https://github.com/ultralytics/ultralytics/issues/2221))
- The resolution logic changed across versions, silently breaking existing configs ([#873](https://github.com/ultralytics/ultralytics/issues/873))
- Even absolute paths can interact unexpectedly with `datasets_dir` depending on the Ultralytics version ([#9503](https://github.com/ultralytics/ultralytics/issues/9503))

The only reliable workaround on the Ultralytics side is to override the global setting with `yolo settings datasets_dir='.'` — but that's a per-machine configuration that doesn't travel with your project.

Bottom line: keep it simple.  Put `data.yaml` inside your dataset directory, don't use the `path` entry (or set it to `"."`), and your dataset will work identically across `cocoyolo`, Ultralytics, and any other tool that reads the YAML.

## Task Type Detection

YOLO requires uniform label types within a dataset directory: every `.txt` file must contain either all bounding boxes (`class xc yc w h`) or all polygons (`class x1 y1 x2 y2 ...`), never a mix of the two.  COCO, on the other hand, can store both annotation types in the same JSON file. Because of this, maybe in some unintended scenari, or malformed COCO json files, some objects may end up having only a bounding box, while others carry full segmentation masks.

as a safety check, `cocoyolo` pre-scans the COCO annotations before writing anything, and uses the `--task` flag to decide what to do:

- **`--task auto`** (default) — Counts how many annotations have segmentation data and how many are bbox-only.  If all annotations have segmentation, the output is YOLO segmentation format.  If none do, the output is YOLO detection format.  If the dataset is **mixed** (some with segmentation, some without), conversion stops with a clear error message telling you exactly how many of each type were found, and suggesting the two flags below.

- **`--task detect`** — Forces bounding-box output for every annotation, regardless of whether it has segmentation data or not.  Every COCO annotation already carries a `bbox` field, so nothing is lost and nothing is skipped — you just get detection labels.

- **`--task segment`** — Forces polygon output.  Annotations that have segmentation data are converted normally; annotations that lack it are **skipped** (the conversion stats will report how many were dropped, so you know exactly what happened).

## COCO → YOLO: Strategies for Segmentation of Edge Cases

The images below show an example image with with three annotations (that we can imagine come from a COCO json file) that cover all the tricky cases: apart from a simple polygon, we have two disjoint polygons, and two disjoint RLE-encoded shapes with holes, one of which even has two holes.  Each strategy combination produces a visually different YOLO result, depending on the user choice.

### COCO ground truth

<p align="center"><img src="assets/01_coco_ground_truth.png" width="700"></p>

The source COCO dataset contains three annotations:
- **#0 `simple_poly`** (yellow) — a plain polygon.  No edge cases.
- **#1 `disjoint_polys`** (green) — a single COCO annotation with two disjoint sub-polygons.
- **#2 `disjoint_rle_with_holes`** (blue) — an RLE-encoded mask with two disjoint connected components, each containing one or more holes.

### `holes=bridge, disjoint=bridge` (default)

<p align="center"><img src="assets/02_yolo_bridge_bridge.png" width="700"></p>

Lossless representation.  Every COCO annotation maps to exactly one YOLO line.  Holes are carved out via zero-width inverse bridges (visible as thin lines in the polygon outline, but invisible when rasterised/filled).  Disjoint regions are connected by zero-width bridges between their closest vertices. This strategy tries to mimic in an exact manner the input annotation, despite the YOLO Ultralytics inherent limitation in having no support for multi-polygon annotations or holes. **3 annotations in, 3 annotations out.**

### `holes=fill, disjoint=bridge`

<p align="center"><img src="assets/03_yolo_fill_bridge.png" width="700"></p>

Holes are filled in — the blue shapes become solid circles.  Disjoint regions are still bridged into a single polygon.  Useful when holes don't matter for your task.  **3 annotations in, 3 annotations out.**

### `holes=bridge, disjoint=split`

<p align="center"><img src="assets/04_yolo_bridge_split.png" width="700"></p>

Holes are preserved, but each disjoint region becomes a separate YOLO annotation i.e., a separate instance (note the different integer IDs).  The green pair splits into #1 and #2, the blue pair splits into #3 and #4.  Instance count increases.  **3 annotations in, 5 annotations out.**

### `holes=fill, disjoint=split`

<p align="center"><img src="assets/05_yolo_fill_split.png" width="700"></p>

Maximum simplification. This is what the majority of other tools do, sometimes without even warning the user. Holes are filled and disjoint regions are split.  Each output annotation is a simple, solid polygon.  **3 annotations in, 5 annotations out.**

### How the algorithms work

**Hole bridging** uses `cv2.RETR_CCOMP` hierarchy to identify outer contours and their child holes.  In `bridge` mode, the converter walks the outer boundary and at each hole's closest point, splices in a detour: bridge into the hole, trace the hole boundary in reverse, bridge back, continue the outer boundary.  Each bridge is traversed twice in opposite directions, producing a zero-width seam.  Multiple holes are sorted by position along the outer boundary and spliced in order.

**Disjoint bridging** builds a greedy nearest-neighbour chain through all polygons.  For each adjacent pair in the chain, the closest vertex pair is computed.  Each polygon is entered at its bridge point from the previous polygon and exited at its bridge point toward the next.  The first and last polygons do a full ring; middle polygons do a partial ring between their entry and exit bridge points.  The implicit polygon close creates the final back-bridge.

### Strategy summary

| holes | disjoint | Effect |
|-------|----------|--------|
| `bridge` | `bridge` | Holes preserved, disjoint regions merged into one polygon.  Lossless.  **(default)** |
| `fill` | `bridge` | Holes filled, disjoint regions merged.  Simplest single-polygon output. |
| `bridge` | `split` | Holes preserved, each region becomes a separate instance. |
| `fill` | `split` | Holes filled, regions split.  Maximum simplification. |

## Parallel Processing

COCO-to-YOLO conversion can be CPU-intensive, especially for datasets rich in RLE masks, holes, and disjoint regions — each annotation requires mask decoding, contour extraction, polygon approximation, and potentially bridge computation.  Since every image is independent, `cocoyolo` processes them in parallel using `multiprocessing`.

By default, all available CPU cores are used.  You can control this with `--workers`:

```bash
coco2yolo path/to/coco path/to/yolo --workers 8
coco2yolo path/to/coco path/to/yolo --workers 1   # disable multiprocessing
```

On a dataset of 100 images with dense annotations and a high proportion of edge cases (RLE masks with holes, disjoint regions requiring bridging), conversion time dropped from **5 minutes 34 seconds** to **20 seconds** — a **~17x speedup**.

The actual speedup depends on the dataset: simple bounding-box conversions are I/O-bound and benefit less, while segmentation-heavy datasets with RLE masks and complex geometries see the largest gains.

## YOLO → COCO

The reverse direction is straightforward.  Each YOLO label line is already a self-contained annotation:

- **Detection** (`class xc yc w h`): Denormalised to absolute pixel bbox in COCO format.
- **Segmentation** (`class x1 y1 x2 y2 ...`): Denormalised to absolute pixel polygon, stored as a single polygon in COCO's `segmentation` field.  Bounding box is recomputed from the polygon vertices.

Category IDs are shifted from YOLO's 0-based indexing to COCO's 1-based convention by default (use `--keep-zero-indexing` to preserve 0-based IDs).

## CLI Reference

### `coco2yolo`

```
usage: coco2yolo [-h] [--task {auto,detect,segment}] [--contour-approx FLOAT]
                 [--hole-strategy {fill,bridge}] [--disjoint-strategy {split,bridge}]
                 [--image-mode {copy,symlink,hardlink}] [--workers N]
                 [--quiet] input output

positional arguments:
  input                 Input COCO dataset directory.
  output                Output YOLO dataset directory.

options:
  --task                YOLO output type: auto, detect, or segment (default: auto).
  --contour-approx      Contour approximation factor (default: 0.001).
  --hole-strategy       How to handle holes in RLE masks (default: bridge).
  --disjoint-strategy   How to handle disjoint regions (default: bridge).
  --image-mode          How to transfer images: copy, symlink, or hardlink (default: copy).
  --workers N           Number of parallel workers (default: all available cores).
  -q, --quiet           Suppress progress output.
```

### `yolo2coco`

```
usage: yolo2coco [-h] [--keep-zero-indexing] [--image-mode {copy,symlink,hardlink}]
                 [--quiet] input output

positional arguments:
  input                 Input YOLO dataset directory (must contain a data*.yaml
                        file), or direct path to a YAML file.
  output                Output COCO dataset directory.

options:
  --keep-zero-indexing  Keep 0-based category IDs (default: shift to 1-based).
  --image-mode          How to transfer images: copy, symlink, or hardlink (default: copy).
  -q, --quiet           Suppress progress output.
```

### `cocoyolo` (unified)

```
usage: cocoyolo {coco2yolo,yolo2coco} ...

subcommands:
  coco2yolo   Convert COCO → YOLO.
  yolo2coco   Convert YOLO → COCO.
```

## How Images Are Handled

By default, images are **copied** from the source to the output directory.  This is the safest option and works on every OS and filesystem.

If you want to save disk space, you can use `--image-mode symlink` or `--image-mode hardlink` to create links instead of copies:

```bash
# Save disk space with symbolic links
coco2yolo path/to/coco path/to/yolo --image-mode symlink

# Save disk space with hard links (must be on the same filesystem)
yolo2coco path/to/yolo path/to/coco --image-mode hardlink
```

Or via the Python API:

```python
from cocoyolo import coco_to_yolo

info, stats = coco_to_yolo("path/to/coco", "path/to/yolo", image_mode="symlink")
```

| Mode | Disk usage | OS support | Notes |
|------|-----------|------------|-------|
| `copy` (default) | Full copy | All | Safest; output is fully self-contained. |
| `symlink` | No extra space | Linux, macOS, Windows (dev mode) | Output breaks if source moves. |
| `hardlink` | No extra space | Linux, macOS, Windows | Source and output must be on the same filesystem. |

If symlinks or hard links are not supported on the current platform, the converter logs a warning and falls back to copy automatically.

## In-place Conversion (Two Formats, One Directory)

You can pass the **same directory** as both input and output.  When the source and destination resolve to the same file, `cocoyolo` silently skips the image transfer — no wasted copies, no broken symlinks, no data corruption.

This is useful when you want both COCO and YOLO labels to **co-exist** in a single dataset directory.  If you adopt the standard COCO-A layout and YOLO-A layout, the two formats share the same `images/{split}/` tree and only add their own metadata on top:

```
dataset/
├── data.yaml                         # YOLO
├── annotations/                      # COCO
│   ├── instances_train.json
│   └── instances_val.json
├── images/                           # shared by both
│   ├── train/
│   │   ├── 000001.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── labels/                           # YOLO
    ├── train/
    │   ├── 000001.txt
    │   └── ...
    └── val/
        └── ...
```

This makes it easy to bounce between frameworks that expect different formats (e.g. Ultralytics for YOLO, most other tooling for COCO) like I frequently do, without maintaining separate copies of your dataset:

```bash
# Generate YOLO labels alongside existing COCO annotations
coco2yolo my_dataset/ my_dataset/

# Or the other way around
yolo2coco my_dataset/ my_dataset/
```

This "two formats in one directory" approach works cleanly with COCO-A + YOLO-A because the two formats use non-overlapping metadata paths (`annotations/` vs `labels/` + `data.yaml`), while sharing the same `images/` tree.  Other layout combinations (e.g. COCO-B, YOLO-B) place images in split-specific folders that may not align, so in-place conversion is best suited for the standard A-layouts.

## Duplicate Filenames

Both COCO and YOLO resolve images by **basename** — the filename without any directory prefix.  This means that if two images share the same name in different subdirectories (e.g. `train/images/photo.jpg` and `val/images/photo.jpg`), the lookup becomes ambiguous and downstream pipelines (not just `cocoyolo`, but Ultralytics, FiftyOne, and others) will silently match the wrong file or crash.

`cocoyolo` guards against this as well: when building its image index it scans for duplicates and **raises an error** listing the offending filenames and their full paths.  This happens early, before any conversion work begins.

If you encounter this error, you must rename the conflicting files in your source dataset before converting.  Unique filenames across the entire dataset are not just a `cocoyolo` requirement — they are a prerequisite for any reliable training or evaluation pipeline. In other words, the lack of name uniqueness is simply a **recipe for disaster** I would strongly discourage.

## License

MIT
