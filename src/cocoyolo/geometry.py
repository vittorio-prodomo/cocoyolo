"""Geometry algorithms for bridging holes and disjoint polygons.

Implements two bridging techniques for converting COCO segmentation masks
to single closed YOLO polygons:

1. **Hole bridging** — walk the outer boundary and splice in each hole as
   a reversed-ring detour, connected by zero-width bridges.
2. **Disjoint bridging** — splice each disjoint polygon's full ring into an
   accumulator via a doubled (out-and-back) zero-width bridge, so even-odd
   fill equals their union for any fragment count (the doubled-keyhole).
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


# ------------------------------------------------------------------
# Contour grouping (RETR_CCOMP hierarchy)
# ------------------------------------------------------------------


def group_contours(
    contours: tuple,
    hierarchy: Optional[np.ndarray],
) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """Group outer contours with their child holes.

    Uses the two-level hierarchy from ``cv2.RETR_CCOMP``: outer contours
    have ``parent == -1``, their child holes are linked via
    ``first_child`` / ``next_sibling``.

    Returns:
        List of ``(outer_contour, [hole_contours])`` tuples.
    """
    if hierarchy is None:
        return [(c, []) for c in contours if len(c) >= 3]

    h = hierarchy[0]
    groups: List[Tuple[np.ndarray, List[np.ndarray]]] = []

    for i, c in enumerate(contours):
        if len(c) < 3:
            continue
        if h[i][3] != -1:
            continue  # skip holes; picked up via their parent

        holes: List[np.ndarray] = []
        child_idx = h[i][2]  # first child
        while child_idx != -1:
            if len(contours[child_idx]) >= 3:
                holes.append(contours[child_idx])
            child_idx = h[child_idx][0]  # next sibling
        groups.append((c, holes))

    return groups


# ------------------------------------------------------------------
# Contour approximation
# ------------------------------------------------------------------


def approx_contour(
    contour: np.ndarray, factor: float
) -> Optional[np.ndarray]:
    """Approximate a single contour.  Returns *None* if < 3 points remain."""
    eps = factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    return approx if len(approx) >= 3 else None


# ------------------------------------------------------------------
# Hole bridging (inverse-bridge / splice approach)
# ------------------------------------------------------------------


def bridge_holes(
    outer_pts: List[List[float]],
    holes: List[np.ndarray],
) -> List[List[float]]:
    """Bridge holes into an outer polygon in pixel space.

    Walks the outer boundary and, at each hole's closest point on the
    outer boundary, makes a detour: bridge into the hole, trace the
    hole boundary in reverse (counter-clockwise), bridge back to the
    outer boundary, and continue.

    Each bridge is traversed twice in opposite directions, producing a
    zero-width seam when rasterised.

    For an outer polygon ``[0..N]`` with two holes spliced at indices
    ``oi1 < oi2``::

        outer[0], ..., outer[oi1],            <- walk outer to 1st bridge
        H1[hi1], H1[hi1-1], ..., H1[hi1],    <- detour: hole 1 reversed ring
        outer[oi1],                           <- bridge back (zero-width)
        outer[oi1+1], ..., outer[oi2],        <- continue outer to 2nd bridge
        H2[hi2], H2[hi2-1], ..., H2[hi2],    <- detour: hole 2 reversed ring
        outer[oi2],                           <- bridge back (zero-width)
        outer[oi2+1], ..., outer[N-1]         <- finish outer
    """
    n_outer = len(outer_pts)

    # For each hole, find the closest point pair to the outer boundary
    hole_info: List[Tuple[int, int, List[List[float]]]] = []
    for hole in holes:
        hole_pts = hole.squeeze().tolist()
        if len(hole_pts) < 3:
            continue

        best_dist = float("inf")
        oi, hi = 0, 0
        for i, op in enumerate(outer_pts):
            for j, hp in enumerate(hole_pts):
                d = (hp[0] - op[0]) ** 2 + (hp[1] - op[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    oi, hi = i, j

        hole_info.append((oi, hi, hole_pts))

    if not hole_info:
        return list(outer_pts)

    # Sort holes by their insertion point along the outer boundary
    hole_info.sort(key=lambda x: x[0])

    # Walk the outer boundary, splicing in hole detours
    result: List[List[float]] = []
    outer_idx = 0

    for oi, hi, hole_pts in hole_info:
        # Trace outer from current position up to and including bridge point
        while outer_idx <= oi:
            result.append(outer_pts[outer_idx])
            outer_idx += 1
        # Last emitted point is now outer[oi] -- the bridge departure.

        # Detour: trace hole reversed (full ring back to entry point)
        n_hole = len(hole_pts)
        for k in range(n_hole + 1):
            result.append(hole_pts[(hi - k) % n_hole])

        # Bridge back to the outer departure point (zero-width)
        result.append(outer_pts[oi])

    # Emit remaining outer vertices after the last hole
    while outer_idx < n_outer:
        result.append(outer_pts[outer_idx])
        outer_idx += 1

    return result


# ------------------------------------------------------------------
# Disjoint bridging (doubled-keyhole union)
# ------------------------------------------------------------------


def bridge_disjoint(
    point_lists: List[List[List[float]]],
) -> List[List[float]]:
    """Connect disjoint polygons via zero-width bridges in pixel space.

    Delegates to :func:`bridge_keyhole`, which splices each polygon's full
    ring into an accumulator with a doubled (out-and-back) zero-width bridge,
    so even-odd fill of the result equals the union of the inputs for any
    fragment count.

    Return contract: ``[]`` for no input, the sole ``point_lists[0]``
    verbatim for a single polygon, otherwise one bridged ring.
    """
    if len(point_lists) <= 1:
        return point_lists[0] if point_lists else []
    return bridge_keyhole(point_lists)


def closest_points(
    poly1: List[List[float]], poly2: List[List[float]]
) -> Tuple[int, int, float]:
    """Find the closest point pair between two polygons.

    Returns ``(idx_in_poly1, idx_in_poly2, squared_distance)``.
    """
    best_i, best_j, best_d = 0, 0, float("inf")
    for i, p1 in enumerate(poly1):
        for j, p2 in enumerate(poly2):
            d = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
            if d < best_d:
                best_i, best_j, best_d = i, j, d
    return best_i, best_j, best_d


def _closest_pair(out: np.ndarray, ring: np.ndarray) -> Tuple[int, int]:
    """Indices ``(i in out, j in ring)`` of the closest point pair (vectorised)."""
    d = ((out[:, None, :] - ring[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(d.argmin()), d.shape)
    return int(i), int(j)


def _signed_area(ring: np.ndarray) -> float:
    """Shoelace signed area of a closed ring (>= 0 is our canonical winding)."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def bridge_keyhole(
    rings: List[List[List[float]]],
) -> List[List[float]]:
    """Bridge N disjoint simple polygons into one even-odd-fillable ring.

    Splices each ring into the accumulator at the closest point pair,
    tracing the ring's FULL loop back to the bridge vertex so the bridge is a
    DOUBLED coincident segment (zero area, even crossing-parity). Even-odd
    fill of the result equals the union of the rings for any N -- a bridge may
    route through another fragment without un-filling it.

    Args:
        rings: polygon point lists ``[[x, y], ...]`` in pixel space.

    Returns:
        One polygon point list ``[[x, y], ...]``. ``[]`` if no ring has
        >= 3 points; the sole ring (as a list) if exactly one qualifies.
        The multi-ring result is oriented counter-clockwise (deterministic).
    """
    rs = [np.asarray(r, dtype=np.float64).reshape(-1, 2) for r in rings]
    rs = [r for r in rs if len(r) >= 3]
    if not rs:
        return []
    if len(rs) == 1:
        return rs[0].tolist()
    # Largest ring first: a substantial base (affects seam routing only).
    rs.sort(key=len, reverse=True)
    out = rs[0].copy()
    for ring in rs[1:]:
        i, j = _closest_pair(out, ring)
        # Full loop of `ring` starting and ending at ring[j].
        ring_loop = np.concatenate([ring[j:], ring[:j], ring[j:j + 1]], axis=0)
        # out[:i+1] + ring_loop + out[i:]  ->  doubled bridge out[i] <-> ring[j].
        out = np.concatenate([out[:i + 1], ring_loop, out[i:]], axis=0)
    # Cosmetic: canonical CCW winding for deterministic output.
    if _signed_area(out) < 0:
        out = out[::-1]
    return out.tolist()


# ------------------------------------------------------------------
# Mask to polygons
# ------------------------------------------------------------------


def mask_to_polygons(
    mask: np.ndarray,
    approx_factor: float = 0.0005,
    hole_strategy: str = "bridge",
    disjoint_strategy: str = "bridge",
) -> List[List[List[float]]]:
    """Convert a binary mask to polygon point lists in pixel space.

    Decodes the mask into OpenCV contours, groups outers with holes,
    applies hole and disjoint strategies, and returns pixel-space
    point lists ready for clipping or normalization.

    Args:
        mask: 2-D ``uint8`` array where non-zero pixels are foreground.
        approx_factor: Contour simplification factor passed to
            :func:`approx_contour`.  Smaller values keep more detail.
        hole_strategy: ``"bridge"`` (connect holes via zero-width seams)
            or ``"fill"`` (discard holes, keep only outer boundary).
        disjoint_strategy: ``"bridge"`` (connect disjoint regions) or
            ``"split"`` (return separate polygons).

    Returns:
        List of polygon point lists.  Each point list is
        ``[[x, y], [x, y], ...]`` in pixel coordinates.  When
        *disjoint_strategy* is ``"split"``, multiple lists may be
        returned; otherwise exactly one (or zero if the mask is empty).
    """
    # Ensure binary mask with 255 foreground
    binary = mask.copy()
    binary[binary > 0] = 255

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []

    groups = group_contours(contours, hierarchy)
    if not groups:
        return []

    # Phase 1 — apply hole strategy per outer contour (pixel space)
    processed: List[List[List[float]]] = []
    for outer, holes in groups:
        outer_approx = approx_contour(outer, approx_factor)
        if outer_approx is None:
            continue
        outer_pts = outer_approx.squeeze().tolist()

        if hole_strategy == "fill" or not holes:
            processed.append(outer_pts)
        else:
            hole_contours = [
                h for h in
                (approx_contour(h, approx_factor) for h in holes)
                if h is not None
            ]
            if hole_contours:
                processed.append(bridge_holes(outer_pts, hole_contours))
            else:
                processed.append(outer_pts)

    if not processed:
        return []

    # Phase 2 — apply disjoint strategy
    if len(processed) == 1:
        return [processed[0]]

    if disjoint_strategy == "split":
        return processed

    bridged = bridge_disjoint(processed)
    return [bridged]
