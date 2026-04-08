"""Geometry algorithms for bridging holes and disjoint polygons.

Implements two bridging techniques for converting COCO segmentation masks
to single closed YOLO polygons:

1. **Hole bridging** — walk the outer boundary and splice in each hole as
   a reversed-ring detour, connected by zero-width bridges.
2. **Disjoint bridging** — build a greedy nearest-neighbour chain through
   all disjoint polygons and traverse it so that each transition is a
   zero-width bridge.
"""

from typing import Dict, List, Optional, Tuple

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
# Disjoint bridging (greedy nearest-neighbour chain)
# ------------------------------------------------------------------


def bridge_disjoint(
    point_lists: List[List[List[float]]],
) -> List[List[float]]:
    """Connect disjoint polygons via zero-width bridges in pixel space.

    Builds a greedy nearest-neighbour chain through all polygons,
    then traverses the chain so that each polygon is entered and exited
    at the bridge points, producing proper zero-width bridges.

    For the chain ``[A -- B -- C]``:

    * Start at A's bridge point toward B, trace A's full ring.
    * Cross zero-width bridge to B (A_exit == A_bridge, B_entry == B_bridge_A).
    * On B, trace from entry (bridge to A) around to exit (bridge to C).
    * Cross zero-width bridge to C.
    * Trace C's full ring.
    * Implicit close: back to A's start creates the final back-bridge.
    """
    if len(point_lists) <= 1:
        return point_lists[0] if point_lists else []

    chain, bridges = _build_chain(point_lists)
    return _traverse_chain(point_lists, chain, bridges)


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


def _build_chain(
    point_lists: List[List[List[float]]],
) -> Tuple[List[int], Dict[Tuple[int, int], Tuple[int, int]]]:
    """Build a greedy nearest-neighbour chain through *point_lists*.

    Returns:
        chain: Ordered list of polygon indices forming the path.
        bridges: ``{(chain[k], chain[k+1]): (exit_idx, entry_idx)}``
            mapping each directed edge to the vertex indices used for
            the bridge on each polygon.
    """
    n = len(point_lists)

    # Pre-compute the best bridge for every unordered pair
    pair_info: Dict[Tuple[int, int], Tuple[int, int, float]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            i1, i2, d = closest_points(point_lists[i], point_lists[j])
            pair_info[(i, j)] = (i1, i2, d)

    def _bridge(a: int, b: int) -> Tuple[int, int, float]:
        """Return (idx_on_a, idx_on_b, dist^2) for the pair a->b."""
        if a < b:
            return pair_info[(a, b)]
        i2, i1, d = pair_info[(b, a)]
        return i1, i2, d

    # Greedy nearest-neighbour: start from polygon 0, always pick the
    # closest unvisited polygon.
    visited = {0}
    chain = [0]
    for _ in range(n - 1):
        current = chain[-1]
        best_next, best_dist = -1, float("inf")
        for cand in range(n):
            if cand in visited:
                continue
            _, _, d = _bridge(current, cand)
            if d < best_dist:
                best_next, best_dist = cand, d
        visited.add(best_next)
        chain.append(best_next)

    # Record bridge vertex indices for each consecutive pair in chain
    bridges: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for k in range(len(chain) - 1):
        a, b = chain[k], chain[k + 1]
        ia, ib, _ = _bridge(a, b)
        bridges[(a, b)] = (ia, ib)

    return chain, bridges


def _traverse_chain(
    polys: List[List[List[float]]],
    chain: List[int],
    bridges: Dict[Tuple[int, int], Tuple[int, int]],
) -> List[List[float]]:
    """Traverse the chain, emitting one connected polygon.

    Each polygon in the chain is entered at a bridge point and exited
    at another (or the same, for end-of-chain polygons).  The traversal
    always follows the polygon's vertex order between entry and exit.

    For chain ``[A, B, C]``::

        Start at A's bridge-to-B point.
        Trace A ring (full loop back to bridge-to-B).
        Jump to B's bridge-from-A point.           <- zero-width bridge
        Trace B from bridge-from-A to bridge-to-C.
        Jump to C's bridge-from-B point.           <- zero-width bridge
        Trace C ring (full loop back to bridge-from-B).
        Implicit close back to A start.            <- zero-width bridge
    """
    result: List[List[float]] = []

    for pos, poly_idx in enumerate(chain):
        poly = polys[poly_idx]
        n = len(poly)

        is_first = pos == 0
        is_last = pos == len(chain) - 1

        # Determine entry point (where we arrive from the previous polygon)
        if is_first:
            # Enter at the bridge point toward the next polygon
            entry_vtx = bridges[(chain[0], chain[1])][0]
        else:
            prev = chain[pos - 1]
            entry_vtx = bridges[(prev, poly_idx)][1]

        # Determine exit point (where we depart toward the next polygon)
        if is_last:
            exit_vtx = entry_vtx  # full ring, exit == entry
        else:
            nxt = chain[pos + 1]
            exit_vtx = bridges[(poly_idx, nxt)][0]

        # Trace from entry around to exit (following vertex order)
        if entry_vtx == exit_vtx:
            # Full ring: emit N+1 vertices (return to start) so that the
            # bridge departure / implicit close lands on the bridge point,
            # not one vertex before it.
            for k in range(n + 1):
                result.append(poly[(entry_vtx + k) % n])
        else:
            # Partial ring from entry -> exit (inclusive on both ends)
            k = entry_vtx
            while True:
                result.append(poly[k % n])
                if k % n == exit_vtx:
                    break
                k += 1

    return result


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
