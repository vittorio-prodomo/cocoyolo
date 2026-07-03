"""Doubled-keyhole disjoint bridging: recall (no dropped pixel), the even-odd
stress case, the bridge_disjoint contract, CCW output, and the hole+disjoint
two-phase interaction. Rasterized with cv2 even-odd fillPoly (the YOLO consumer)."""
import cv2
import numpy as np
import pytest

from cocoyolo.geometry import bridge_disjoint, mask_to_polygons


def _fill(poly, H, W):
    m = np.zeros((H, W), np.uint8)
    p = np.asarray(poly, np.float32).reshape(-1, 2)
    if len(p) >= 3:
        cv2.fillPoly(m, [p.round().astype(np.int32)], 1)
    return m


def _union(rings, H, W):
    m = np.zeros((H, W), np.uint8)
    for r in rings:
        m |= _fill(r, H, W)
    return m


def _iou(a, b):
    u = int(np.logical_or(a, b).sum())
    return (int(np.logical_and(a, b).sum()) / u) if u else 1.0


def _recall(true_m, pred_m):
    t = int(true_m.sum())
    return 1.0 if t == 0 else int(np.logical_and(true_m, pred_m).sum()) / t


def _square(cx, cy, s):
    h = s / 2.0
    return np.array([[cx - h, cy - h], [cx + h, cy - h],
                     [cx + h, cy + h], [cx - h, cy + h]], float)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_diagonal_squares_full_recall(n):
    H = W = 340
    rings = [_square(45 + 58 * k, 45 + 58 * k, 40) for k in range(n)]
    pred = _fill(bridge_disjoint([r.tolist() for r in rings]), H, W)
    union = _union(rings, H, W)
    assert _recall(union, pred) == 1.0        # RED on current code (N>=3 drops middles)
    assert _iou(pred, union) >= 0.98


def test_collinear_fragments_full_recall():
    # Close-set collinear fragments (small gaps, like a segmented crack) so the
    # seam stays negligible; recall is the hard property, IoU the seam sanity.
    H, W = 120, 300
    rings = [_square(45 + 65 * k, 60, 40) for k in range(4)]
    pred = _fill(bridge_disjoint([r.tolist() for r in rings]), H, W)
    union = _union(rings, H, W)
    assert _recall(union, pred) == 1.0
    assert _iou(pred, union) >= 0.98


def test_dense_scatter_cluster_full_recall():
    # 3x3 grid of small squares -> many short bridges. The union property must
    # hold across all 9 fragments (recall 1.0, each fragment intact). Real
    # bridge-over-fragment crossings are covered by the 90-ann gate (Task 3);
    # greedy-nearest never bridges over a between-fragment (triangle ineq.).
    H = W = 130
    rings = [_square(25 + 40 * c, 25 + 40 * r, 22) for r in range(3) for c in range(3)]
    pred = _fill(bridge_disjoint([r.tolist() for r in rings]), H, W)
    assert _recall(_union(rings, H, W), pred) == 1.0
    for r in rings:                            # each fragment individually intact
        assert _recall(_fill(r, H, W), pred) == 1.0


def test_bridge_disjoint_contract_and_delegation():
    from cocoyolo.geometry import bridge_keyhole
    assert bridge_disjoint([]) == []
    one = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    assert bridge_disjoint([one]) == one       # single ring: verbatim passthrough
    two = [_square(30, 30, 20).tolist(), _square(80, 80, 20).tolist()]
    merged = bridge_disjoint(two)
    assert isinstance(merged, list) and all(len(p) == 2 for p in merged)
    assert bridge_disjoint(two) == bridge_keyhole(two)   # delegates for N>=2


def test_bridged_output_is_ccw():
    from cocoyolo.geometry import bridge_keyhole
    two = [_square(30, 30, 20).tolist(), _square(80, 80, 20).tolist()]
    out = np.asarray(bridge_keyhole(two), float)
    x, y = out[:, 0], out[:, 1]
    signed = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    assert signed >= 0.0                       # canonical CCW winding


def test_mask_to_polygons_hole_plus_disjoint():
    # Regression guard for the two-phase path (holes phase-1, disjoint phase-2):
    # a blob with a hole + a disjoint blob must bridge to ONE ring that keeps
    # the hole empty and both blob bodies filled.
    H = W = 200
    mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask, (20, 60), (120, 140), 1, -1)   # blob A (outer)
    cv2.rectangle(mask, (55, 90), (85, 110), 0, -1)     # hole in A
    cv2.rectangle(mask, (150, 80), (185, 120), 1, -1)   # blob B (disjoint)
    polys = mask_to_polygons(mask, disjoint_strategy="bridge")
    assert len(polys) == 1
    pred = _fill(polys[0], H, W)
    assert pred[100, 70] == 0                            # hole stays empty
    assert pred[100, 30] == 1                            # blob A body filled
    assert pred[100, 167] == 1                           # blob B filled
