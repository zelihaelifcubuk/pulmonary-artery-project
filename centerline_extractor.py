"""
centerline_extractor.py  (step 3, pulmonary artery version)
===========================================================
Extracts the PA centerline tree and splits it at the bifurcation into
MPA (main), LPA (left) and RPA (right) centerlines.

Input  (in the case output folder):
    pa.nii.gz                    cleaned PA mask (step 2)
    resampled_rv.nii.gz          right ventricle  -> marks the proximal (valve) end of the MPA
    resampled_lung_left.nii.gz   \\ lung centroids decide which branch is left / right
    resampled_lung_right.nii.gz  /

Output:
    centerline_mpa.vtp, centerline_lpa.vtp, centerline_rpa.vtp
    pipeline_mode.json  (runtime, bifurcation point and branch-assignment QC are appended)

Method: skeleton graph -> pick the 3 endpoints of the Y -> vmtk centerlines from the
MPA end to the two branch ends -> split at the skeleton-derived bifurcation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import networkx as nx
import numpy as np
import SimpleITK as sitk
import vtk
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from vtk.util import numpy_support

MIN_BRANCH_MM   = 10.0   # an arm must be at least this long past the divergence point
MAX_RV_GAP_MM   = 15.0   # warn if the MPA end is farther than this from the RV
_MIN_BRANCH_PTS = 10     # vmtk path with fewer points than this is "degenerate"
_MAX_TARGET_GAP = 20.0   # mm; vmtk path ending farther than this from its seed is degenerate


# ── Helper 1: SimpleITK image -> VTK ImageData ────────────────────────────
def nifti_to_vtk_image(sitk_image: sitk.Image) -> vtk.vtkImageData:
    arr       = sitk.GetArrayFromImage(sitk_image).astype(np.float32)  # (Z, Y, X)
    spacing   = sitk_image.GetSpacing()
    origin    = sitk_image.GetOrigin()
    direction = sitk_image.GetDirection()

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(arr.shape[2], arr.shape[1], arr.shape[0])
    vtk_img.SetSpacing(*spacing)
    vtk_img.SetOrigin(*origin)

    vtk_matrix = vtk.vtkMatrix3x3()
    for i in range(3):
        for j in range(3):
            vtk_matrix.SetElement(i, j, direction[i * 3 + j])
    vtk_img.SetDirectionMatrix(vtk_matrix)

    vtk_arr = numpy_support.numpy_to_vtk(arr.ravel(order="C"), deep=True,
                                         array_type=vtk.VTK_FLOAT)
    vtk_arr.SetName("ImageScalars")
    vtk_img.GetPointData().SetScalars(vtk_arr)
    return vtk_img


# ── Helper 2: VTK ImageData -> surface mesh ───────────────────────────────
def get_surface(vtk_image: vtk.vtkImageData) -> vtk.vtkPolyData:
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_image)
    mc.SetValue(0, 0.5)
    mc.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(mc.GetOutputPort())
    cleaner.Update()

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputConnection(cleaner.GetOutputPort())
    conn.SetExtractionModeToAllRegions()
    conn.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(conn.GetOutputPort())
    smoother.SetNumberOfIterations(20)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.Update()

    # Decimate before vmtk: Voronoi cost scales with vertex count.
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputConnection(smoother.GetOutputPort())
    decimate.SetTargetReduction(0.75)
    decimate.PreserveTopologyOn()
    decimate.Update()

    surface = decimate.GetOutput()
    print(f"  Surface vertices : {surface.GetNumberOfPoints()}")
    print(f"  Surface cells    : {surface.GetNumberOfCells()}")
    return surface


# ── Helper 3: geometry utilities ──────────────────────────────────────────
def _voxels_to_physical(zyx, image: sitk.Image) -> np.ndarray:
    """(N, 3) voxel indices in (z, y, x) order -> (N, 3) physical points (x, y, z) in mm."""
    zyx       = np.asarray(zyx, dtype=np.float64).reshape(-1, 3)
    xyz       = zyx[:, ::-1]
    spacing   = np.array(image.GetSpacing())
    origin    = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    return origin + (xyz * spacing) @ direction.T


def _load_on_grid(path: str, ref: sitk.Image) -> sitk.Image:
    """Read a mask; resample onto ref's grid (nearest) only if the grids differ."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required mask not found: '{path}'. Run resampling.py (step 1) first."
        )
    img = sitk.ReadImage(path)
    same = (img.GetSize() == ref.GetSize()
            and np.allclose(img.GetSpacing(),   ref.GetSpacing(),   atol=1e-4)
            and np.allclose(img.GetOrigin(),    ref.GetOrigin(),    atol=1e-4)
            and np.allclose(img.GetDirection(), ref.GetDirection(), atol=1e-4))
    if not same:
        print(f"  [INFO] {os.path.basename(path)} not on the PA grid -> resampling (nearest).")
        img = sitk.Resample(img, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    return img


def _mask_points_mm(path: str, ref: sitk.Image) -> np.ndarray:
    """Physical coordinates (mm) of every foreground voxel of a mask."""
    img = _load_on_grid(path, ref)          # keep a reference: array views die with their image
    arr = sitk.GetArrayFromImage(img) > 0
    zyx = np.argwhere(arr)
    if len(zyx) == 0:
        raise ValueError(f"Mask is empty: '{path}'")
    return _voxels_to_physical(zyx, ref)


def _mask_centroid_mm(path: str, ref: sitk.Image) -> np.ndarray:
    img = _load_on_grid(path, ref)
    arr = sitk.GetArrayFromImage(img) > 0
    zyx = np.argwhere(arr)
    if len(zyx) == 0:
        raise ValueError(f"Mask is empty: '{path}'")
    return _voxels_to_physical(zyx.mean(axis=0), ref)[0]


# ── Helper 4: skeleton graph ──────────────────────────────────────────────
def _build_skeleton_graph(mask_image: sitk.Image) -> nx.Graph:
    arr = sitk.GetArrayFromImage(mask_image) > 0            # (Z, Y, X)
    if not arr.any():
        raise ValueError("Mask is empty -- no foreground voxels found.")

    nz  = np.argwhere(arr)
    pad = 2
    mn  = np.maximum(nz.min(axis=0) - pad, 0)
    mx  = np.minimum(nz.max(axis=0) + pad + 1, np.array(arr.shape))
    crop = arr[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
    print(f"  Bounding box crop: {arr.shape} -> {crop.shape}")

    print("  Skeletonizing PA mask (Lee 3D thinning) ...")
    skel   = skeletonize(crop, method="lee") > 0
    coords = np.argwhere(skel) + mn
    if coords.shape[0] < 3:
        raise ValueError("Skeleton has fewer than 3 voxels -- cannot find the PA branches.")
    coord_set = set(map(tuple, coords))
    print(f"  Skeleton voxel count : {len(coord_set)}")

    sx, sy, sz = mask_image.GetSpacing()
    offsets = [(dz, dy, dx)
               for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) != (0, 0, 0)]

    graph = nx.Graph()
    for (z, y, x) in coord_set:
        graph.add_node((z, y, x))
        for dz, dy, dx in offsets:
            nb = (z + dz, y + dy, x + dx)
            if nb in coord_set:
                w = math.sqrt((dx * sx) ** 2 + (dy * sy) ** 2 + (dz * sz) ** 2)
                graph.add_edge((z, y, x), nb, weight=w)

    comps   = list(nx.connected_components(graph))
    largest = max(comps, key=len)
    if len(comps) > 1:
        print(f"  [INFO] Skeleton had {len(comps)} components; keeping the largest "
              f"({len(largest)} / {len(coord_set)} voxels).")
    return graph.subgraph(largest).copy()


def _smooth_resample(pts, step: float = 0.5, smooth_mm: float = 6.0) -> np.ndarray:
    """Resample a skeleton path to `step` mm and smooth it (ends stay fixed). Used for fallbacks."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return pts
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if s[-1] < step:
        return pts
    s_new = np.linspace(0.0, s[-1], int(round(s[-1] / step)) + 1)
    res   = np.column_stack([np.interp(s_new, s, pts[:, i]) for i in range(3)])
    sm    = gaussian_filter1d(res, sigma=max(smooth_mm / step / 2.0, 0.5), axis=0, mode="nearest")
    w     = np.clip(np.minimum(s_new, s_new[-1] - s_new) / smooth_mm, 0.0, 1.0)[:, None]
    return res * (1.0 - w) + sm * w


def _common_prefix_len(path_a: list, path_b: list) -> int:
    k = 0
    for a, b in zip(path_a, path_b):
        if a != b:
            break
        k += 1
    return k


# ── Helper 5: seed points for the PA Y-topology ───────────────────────────
def compute_seed_points_pa_y(mask_image: sitk.Image, rv_points_mm: np.ndarray,
                              lung_left_c: np.ndarray, lung_right_c: np.ndarray):
    """
    Pick the three Y endpoints:
      MPA end : skeleton endpoint closest to the right ventricle (pulmonary valve end)
      arm 1   : endpoint farthest (geodesic) from the MPA end
      arm 2   : farthest endpoint whose path diverges from arm 1 by >= MIN_BRANCH_MM
                (this rejects short spurs on the same branch)
    Left/right: the arm closer to the left lung centroid is the LPA.
    """
    graph     = _build_skeleton_graph(mask_image)
    endpoints = [n for n in graph.nodes if graph.degree(n) == 1]
    print(f"  Candidate endpoints (degree-1 nodes) : {len(endpoints)}")
    if len(endpoints) < 3:
        raise ValueError(
            f"Found only {len(endpoints)} skeleton endpoint(s); a Y-shaped PA needs >= 3. "
            "Check pa.nii.gz in Slicer (is one branch missing or cut off?)."
        )

    ep_phys = _voxels_to_physical(np.array(endpoints), mask_image)

    # MPA end = endpoint nearest to the RV
    d_rv, _ = cKDTree(rv_points_mm).query(ep_phys)
    i_mpa   = int(np.argmin(d_rv))
    mpa_pt  = endpoints[i_mpa]
    rv_gap  = float(d_rv[i_mpa])
    print(f"  MPA end (nearest to RV): voxel {tuple(int(v) for v in mpa_pt)}, "
          f"distance to RV = {rv_gap:.1f} mm")
    if rv_gap > MAX_RV_GAP_MM:
        print(f"  [WARN] MPA end is {rv_gap:.1f} mm from the RV (> {MAX_RV_GAP_MM} mm). "
              "The proximal end of the PA mask may be cut off or the RV mask misplaced.")

    dist, paths = nx.single_source_dijkstra(graph, mpa_pt, weight="weight")
    cands = [e for e in endpoints if e != mpa_pt and e in dist]
    if len(cands) < 2:
        raise ValueError("Fewer than 2 skeleton endpoints reachable from the MPA end.")

    # arm 1
    u      = max(cands, key=lambda e: dist[e])
    path_u = paths[u]

    # arm 2
    def arm_len_after_divergence(e):
        k = _common_prefix_len(path_u, paths[e])
        return dist[e] - dist[paths[e][k - 1]], dist[u] - dist[paths[e][k - 1]]

    valid = []
    for e in cands:
        if e == u:
            continue
        len_e, len_u = arm_len_after_divergence(e)
        if len_e >= MIN_BRANCH_MM and len_u >= MIN_BRANCH_MM:
            valid.append(e)
    if not valid:
        raise ValueError(
            f"No second branch found: every other endpoint diverges < {MIN_BRANCH_MM} mm "
            "from the longest arm, so the skeleton looks like a tube, not a Y. "
            "Check pa.nii.gz in Slicer."
        )
    v = max(valid, key=lambda e: dist[e])

    k       = _common_prefix_len(paths[u], paths[v])
    bif_vox = paths[u][k - 1]

    # left / right by lung centroids (orientation-independent)
    p_u = _voxels_to_physical(np.array([u]), mask_image)[0]
    p_v = _voxels_to_physical(np.array([v]), mask_image)[0]

    def left_score(p):          # > 0  =>  closer to the left lung
        return float(np.linalg.norm(p - lung_right_c) - np.linalg.norm(p - lung_left_c))

    s_u, s_v = left_score(p_u), left_score(p_v)
    if s_u >= s_v:
        left_pt, right_pt, s_left, s_right = u, v, s_u, s_v
    else:
        left_pt, right_pt, s_left, s_right = v, u, s_v, s_u
    print(f"  Left-lung score  : LPA arm {s_left:+.1f} mm | RPA arm {s_right:+.1f} mm")
    side_ok = (s_left > 0) and (s_right < 0)
    if not side_ok:
        print("  [WARN] Both arms are closer to the same lung. Left/right was decided "
              "by the relative score only; check this case in Slicer.")

    def to_phys_list(vox):
        z, y, x = vox
        return list(mask_image.TransformIndexToPhysicalPoint((int(x), int(y), int(z))))

    source_point      = to_phys_list(mpa_pt)
    target_left       = to_phys_list(left_pt)
    target_right      = to_phys_list(right_pt)
    bifurcation_point = to_phys_list(bif_vox)

    print(f"  Source (MPA end) : {source_point}")
    print(f"  Target (LPA end) : {target_left}")
    print(f"  Target (RPA end) : {target_right}")
    print(f"  Bifurcation      : {bifurcation_point}")

    def voxels_to_physical_path(vpath):
        return np.array([to_phys_list(n) for n in vpath], dtype=np.float64)

    skel_left_pts  = voxels_to_physical_path(paths[left_pt][k - 1:])
    skel_right_pts = voxels_to_physical_path(paths[right_pt][k - 1:])
    skel_trunk_pts = voxels_to_physical_path(paths[u][:k])        # MPA end -> bifurcation
    print(f"  Skeleton MPA: {len(skel_trunk_pts)} nodes | LPA: {len(skel_left_pts)} | "
          f"RPA: {len(skel_right_pts)}")

    info = {
        "mpa_end_to_rv_mm":   round(rv_gap, 2),
        "left_score_mm":      round(s_left, 2),
        "right_score_mm":     round(s_right, 2),
        "sides_unambiguous":  bool(side_ok),
        "bifurcation_point_mm": [round(c, 2) for c in bifurcation_point],
    }
    return (source_point, target_left, target_right, bifurcation_point,
            skel_left_pts, skel_right_pts, skel_trunk_pts, info)


# ── Helper 6: surface + seeds -> centerlines (vmtk) ───────────────────────
def get_centerlines(surface: vtk.vtkPolyData, output_vtp_path: str,
                    source_point: list, target_points: list) -> vtk.vtkPolyData:
    """vmtk centerlines. Returns an EMPTY polydata if vmtk fails, so the caller can fall back."""
    from vmtk import vmtkscripts      # imported here so the rest of the file loads without vmtk

    try:
        cl = vmtkscripts.vmtkCenterlines()
        cl.Surface          = surface
        cl.SeedSelectorName = "pointlist"
        cl.SourcePoints     = source_point
        cl.TargetPoints     = target_points
        cl.AppendEndPoints  = 1
        cl.Execute()

        rs = vmtkscripts.vmtkCenterlineResampling()
        rs.Centerlines = cl.Centerlines
        rs.Length      = 0.5              # one point every 0.5 mm
        rs.Execute()
        result = rs.Centerlines
    except Exception as e:
        print(f"  [WARN] vmtk failed ({type(e).__name__}: {e}); using skeleton centerlines.")
        return vtk.vtkPolyData()

    if result is None or result.GetNumberOfCells() == 0 or result.GetPoints() is None:
        print("  [WARN] vmtk returned no centerlines; using skeleton centerlines.")
        return vtk.vtkPolyData()

    print(f"  Centerline paths (cells)           : {result.GetNumberOfCells()}")
    print(f"  Centerline points (after resample) : {result.GetPoints().GetNumberOfPoints()}")

    print(f"Saving raw centerlines: {output_vtp_path}")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_vtp_path)
    writer.SetInputData(result)
    writer.Write()
    return result


# ── Helper 7: split the vmtk centerlines at the bifurcation ───────────────
def find_bifurcation_and_split(centerlines: vtk.vtkPolyData, source_point: list,
                               target_left: list, target_right: list,
                               bifurcation_point: list,
                               skel_trunk_pts=None, skel_left_pts=None, skel_right_pts=None):
    """
    Returns (mpa_pts, lpa_pts, rpa_pts, fallback_used). Each array is (N, 3), proximal -> distal.
    vmtk gives 2 paths when both targets are reached. With 0 or 1 paths, or a degenerate path,
    the missing segment(s) are taken from the skeleton instead and listed in fallback_used.
    """
    paths = []
    for cell_idx in range(centerlines.GetNumberOfCells()):
        cell = centerlines.GetCell(cell_idx)
        paths.append(np.array([centerlines.GetPoint(cell.GetPointId(j))
                               for j in range(cell.GetNumberOfPoints())]))
    if len(paths) > 2:
        raise ValueError(f"vmtk returned {len(paths)} centerline paths (expected at most 2).")

    src, bif = np.array(source_point), np.array(bifurcation_point)
    tl, tr   = np.array(target_left), np.array(target_right)

    for i in range(len(paths)):                      # enforce source -> target ordering
        if np.linalg.norm(paths[i][0] - src) > np.linalg.norm(paths[i][-1] - src):
            paths[i] = paths[i][::-1]
            print(f"  [INFO] Path {i} reversed to enforce source->target direction.")

    path_left = path_right = None
    if len(paths) == 2:
        if np.linalg.norm(paths[0][-1] - tl) <= np.linalg.norm(paths[0][-1] - tr):
            path_left, path_right = paths[0], paths[1]
        else:
            path_left, path_right = paths[1], paths[0]
    elif len(paths) == 1:
        side = "LPA" if (np.linalg.norm(paths[0][-1] - tl) <=
                         np.linalg.norm(paths[0][-1] - tr)) else "RPA"
        print(f"  [WARN] vmtk returned only 1 path (it reached the {side} target only).")
        if side == "LPA":
            path_left = paths[0]
        else:
            path_right = paths[0]
    else:
        print("  [WARN] vmtk returned no paths.")

    def usable(path, target):                        # present, long enough, reaches its target
        return (path is not None and len(path) >= _MIN_BRANCH_PTS
                and np.linalg.norm(path[-1] - target) <= _MAX_TARGET_GAP)

    left_ok, right_ok = usable(path_left, tl), usable(path_right, tr)
    idx_left  = int(np.argmin(np.linalg.norm(path_left  - bif, axis=1))) if left_ok  else None
    idx_right = int(np.argmin(np.linalg.norm(path_right - bif, axis=1))) if right_ok else None
    fallback = []

    def _need(arr, name):
        if arr is None or len(arr) < 2:
            raise ValueError(f"vmtk did not give a usable {name} centerline and no skeleton "
                             "fallback is available.")
        fallback.append(name.lower())
        print(f"  [WARN] {name} taken from the skeleton (vmtk path missing or degenerate).")
        return _smooth_resample(arr)

    # MPA: the pre-bifurcation part of whichever usable path has more points
    cands = []
    if left_ok:
        cands.append((idx_left,  path_left[:idx_left + 1],   f"LPA path, idx {idx_left}"))
    if right_ok:
        cands.append((idx_right, path_right[:idx_right + 1], f"RPA path, idx {idx_right}"))
    if cands:
        _, mpa_pts, trunk_tag = max(cands, key=lambda c: len(c[1]))
        mpa_pts = mpa_pts.copy()
    else:
        mpa_pts, trunk_tag = _need(skel_trunk_pts, "MPA"), "skeleton"

    if left_ok:
        lpa_pts, left_tag = path_left[idx_left:].copy(), "vmtk"
    else:
        lpa_pts, left_tag = _need(skel_left_pts, "LPA"), "skeleton"
    if right_ok:
        rpa_pts, right_tag = path_right[idx_right:].copy(), "vmtk"
    else:
        rpa_pts, right_tag = _need(skel_right_pts, "RPA"), "skeleton"

    lpa_pts[0] = mpa_pts[-1]            # continuity at the junction
    rpa_pts[0] = mpa_pts[-1]

    print(f"  Bifurcation (mm) : {bif.tolist()}")
    print(f"  MPA points       : {len(mpa_pts)}  ({trunk_tag})")
    print(f"  LPA points       : {len(lpa_pts)}  ({left_tag})")
    print(f"  RPA points       : {len(rpa_pts)}  ({right_tag})")
    return mpa_pts, lpa_pts, rpa_pts, fallback


# ── Helper 8: numpy points -> vtkPolyData polyline ────────────────────────
def points_to_polydata(pts: np.ndarray) -> vtk.vtkPolyData:
    vtk_pts = vtk.vtkPoints()
    line    = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(len(pts))
    for i, (px, py, pz) in enumerate(pts):
        vtk_pts.InsertNextPoint(float(px), float(py), float(pz))
        line.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)
    pd.SetLines(cells)
    return pd


def write_polydata(pd: vtk.vtkPolyData, path: str) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(pd)
    writer.Write()


def _update_json(path: str, updates: dict) -> None:
    data = {}
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
    data.update(updates)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Main function ─────────────────────────────────────────────────────────
def extract_centerline(input_path: str, output_dir: str = None) -> dict:
    t_start = time.perf_counter()

    print(f"\n[1/7] Loading : {input_path}")
    image = sitk.ReadImage(input_path)
    print(f"      Size    : {image.GetSize()}")
    print(f"      Spacing : {image.GetSpacing()}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path))
    os.makedirs(output_dir, exist_ok=True)

    mpa_vtp = os.path.join(output_dir, "centerline_mpa.vtp")
    lpa_vtp = os.path.join(output_dir, "centerline_lpa.vtp")
    rpa_vtp = os.path.join(output_dir, "centerline_rpa.vtp")
    raw_vtp = os.path.join(output_dir, "centerline_vmtk_raw.vtp")   # temporary
    mode_json = os.path.join(output_dir, "pipeline_mode.json")

    # Required auxiliary masks
    rv_path   = os.path.join(output_dir, "resampled_rv.nii.gz")
    lung_l    = os.path.join(output_dir, "resampled_lung_left.nii.gz")
    lung_r    = os.path.join(output_dir, "resampled_lung_right.nii.gz")

    binary = sitk.Cast(sitk.Greater(image, 0), sitk.sitkUInt8)
    binary.CopyInformation(image)

    # ── Surface (cropped to the foreground box) ───────────────────────────
    print("\n[2/7] Converting to VTK image (cropped to foreground bbox) ...")
    arr = sitk.GetArrayFromImage(binary)                 # (Z, Y, X)
    nz  = np.argwhere(arr)
    pad = 5
    mn  = np.maximum(nz.min(axis=0) - pad, 0)
    mx  = np.minimum(nz.max(axis=0) + pad + 1, np.array(arr.shape))
    roi_index = [int(mn[2]), int(mn[1]), int(mn[0])]     # sitk wants (x, y, z)
    roi_size  = [int(mx[2] - mn[2]), int(mx[1] - mn[1]), int(mx[0] - mn[0])]
    crop = sitk.RegionOfInterest(binary, size=roi_size, index=roi_index)
    print(f"  Crop: {binary.GetSize()} -> {crop.GetSize()}")
    vtk_image = nifti_to_vtk_image(sitk.Cast(crop, sitk.sitkFloat32))

    print("\n[3/7] Extracting surface (Marching Cubes) ...")
    surface = get_surface(vtk_image)

    # ── Seed points ───────────────────────────────────────────────────────
    print("\n[4/7] Computing seed points (PA Y-topology) ...")
    rv_pts = _mask_points_mm(rv_path, image)
    c_left  = _mask_centroid_mm(lung_l, image)
    c_right = _mask_centroid_mm(lung_r, image)
    (source_point, target_left, target_right, bifurcation_point,
     skel_left_pts, skel_right_pts, skel_trunk_pts, info) = compute_seed_points_pa_y(
        binary, rv_pts, c_left, c_right)

    # ── Centerlines (vmtk) ────────────────────────────────────────────────
    print("\n[5/7] Extracting centerlines (vmtk, 'pointlist' seeds) ...")
    centerlines = get_centerlines(surface, raw_vtp, source_point,
                                  target_left + target_right)

    # ── Split at the bifurcation ──────────────────────────────────────────
    print("\n[6/7] Splitting at bifurcation ...")
    mpa_pts, lpa_pts, rpa_pts, fallback = find_bifurcation_and_split(
        centerlines, source_point, target_left, target_right, bifurcation_point,
        skel_trunk_pts=skel_trunk_pts, skel_left_pts=skel_left_pts, skel_right_pts=skel_right_pts)
    info["fallback_used"] = fallback

    write_polydata(points_to_polydata(mpa_pts), mpa_vtp)
    write_polydata(points_to_polydata(lpa_pts), lpa_vtp)
    write_polydata(points_to_polydata(rpa_pts), rpa_vtp)
    if os.path.isfile(raw_vtp):
        os.remove(raw_vtp)
    print(f"\n[7/7] Saved MPA VTP -> {mpa_vtp}")
    print(f"      Saved LPA VTP -> {lpa_vtp}")
    print(f"      Saved RPA VTP -> {rpa_vtp}")

    elapsed = round(time.perf_counter() - t_start, 2)
    _update_json(mode_json, {"runtime_centerline_sec": elapsed, "centerline": info})
    print(f"  Centerline runtime: {elapsed} sec (saved to pipeline_mode.json)")

    return {"mpa": mpa_vtp, "lpa": lpa_vtp, "rpa": rpa_vtp}


# ── CLI entry-point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="PA centerline extraction (vmtk + automated seed points). "
                    "Outputs MPA, LPA and RPA centerlines."
    )
    parser.add_argument("input_path", help="Path to pa.nii.gz (from pa_extractor.py)")
    parser.add_argument("--output-dir", default=None,
                        help="Per-case output directory (default: next to the input file).")
    args = parser.parse_args()

    try:
        result = extract_centerline(args.input_path, output_dir=args.output_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("\nOutput files:")
    for key, path in result.items():
        print(f"  {key:5s} -> {path}")

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input_path))
    elapsed = time.perf_counter() - t0
    print("=" * 60)
    print(f" PROCESS COMPLETED! Outputs saved to: {out_dir}")
    print(f" Total runtime: {elapsed:.2f} sec ({elapsed / 60:.2f} min)")
    print("=" * 60)