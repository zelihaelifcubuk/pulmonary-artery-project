"""
pa_extractor.py  (step 2)
=========================
Cleans the resampled pulmonary artery (PA) mask and checks the auxiliary masks.

Input  (all in the case output folder, created by resampling.py):
    resampled_pa.nii.gz          required
    resampled_aorta.nii.gz       optional (checked only)
    resampled_rv.nii.gz          optional (checked only)
    resampled_lung_left.nii.gz   optional (checked only)
    resampled_lung_right.nii.gz  optional (checked only)

Output:
    pa.nii.gz            cleaned PA mask, single label (value 1)
    pipeline_mode.json   QC summary; later steps append their runtimes to it

Cleaning = binarize -> keep largest connected component -> fill internal holes.
(The aortic version needed two masks, thrombus handling and iliac clipping.
 The PA has one mask and one fixed Y shape, so none of that is needed here.)
"""

import argparse
import json
import os
import sys

import numpy as np
import SimpleITK as sitk

PA_LABEL    = 1      # label value written to pa.nii.gz
_MIN_VOXELS = 50     # below this the PA mask is treated as absent
_SIBLING_WARN_FRAC = 0.10   # a dropped component >= 10% of the largest = suspicious

# Masks that later steps need; only checked here, never modified.
_AUX_MASKS = {
    "aorta":      "resampled_aorta.nii.gz",
    "rv":         "resampled_rv.nii.gz",
    "lung_left":  "resampled_lung_left.nii.gz",
    "lung_right": "resampled_lung_right.nii.gz",
}


def _count(img: sitk.Image) -> int:
    return int(np.count_nonzero(sitk.GetArrayViewFromImage(img)))


def _same_grid(a: sitk.Image, b: sitk.Image, tol: float = 1e-4) -> bool:
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(),   b.GetSpacing(),   atol=tol)
            and np.allclose(a.GetOrigin(),    b.GetOrigin(),    atol=tol)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol))


def _clean_pa(image: sitk.Image):
    """Binarize, keep the largest connected component, fill internal holes."""
    binary = sitk.Cast(image > 0, sitk.sitkUInt8)
    n_raw  = _count(binary)
    print(f"  [PA] Raw voxels : {n_raw}")
    if n_raw < _MIN_VOXELS:
        raise ValueError(
            f"PA mask has only {n_raw} voxels (< {_MIN_VOXELS}); "
            "check resampled_pa.nii.gz and the original pulmonary_artery label."
        )

    # 26-connectivity, same neighbourhood the skeleton graph uses later.
    cc = sitk.ConnectedComponent(binary, True)
    cc = sitk.RelabelComponent(cc, sortByObjectSize=True)      # label 1 = largest
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    sizes   = {l: stats.GetNumberOfPixels(l) for l in stats.GetLabels()}
    largest = sizes[1]

    dropped = sorted((s for l, s in sizes.items() if l != 1), reverse=True)
    suspicious = [s for s in dropped if s >= _SIBLING_WARN_FRAC * largest]
    print(f"  [PA] Components : {len(sizes)} (kept largest = {largest} voxels)")
    if dropped:
        print(f"  [PA] Dropped    : {len(dropped)} component(s), sizes {dropped[:5]}"
              f"{' ...' if len(dropped) > 5 else ''}")
    if suspicious:
        print(f"  [WARN] A dropped component is >= {int(_SIBLING_WARN_FRAC * 100)}% of the "
              "main one. A branch may be disconnected in the segmentation; inspect "
              "this case in Slicer.")

    keep   = sitk.BinaryThreshold(cc, 1, 1, 1, 0)              # UInt8, 1 inside
    filled = sitk.BinaryFillhole(keep)
    n_clean  = _count(filled)
    n_filled = n_clean - _count(keep)
    print(f"  [PA] Holes filled: {n_filled} voxels")
    print(f"  [PA] Clean voxels: {n_clean}")

    return filled, {
        "pa_voxels_raw":           n_raw,
        "pa_voxels_clean":         n_clean,
        "components_found":        len(sizes),
        "components_dropped_sizes": [int(s) for s in dropped[:10]],
        "possible_broken_branch":  bool(suspicious),
        "holes_filled_voxels":     int(n_filled),
    }


def _check_aux_masks(pa_clean: sitk.Image, output_dir: str) -> dict:
    """Presence, voxel count, grid match with the PA, and overlap with the PA."""
    pa_arr = sitk.GetArrayViewFromImage(pa_clean) > 0
    report = {}
    for name, fname in _AUX_MASKS.items():
        path = os.path.join(output_dir, fname)
        if not os.path.isfile(path):
            print(f"  [AUX] {name:<10}: MISSING ({fname})")
            report[name] = {"present": False}
            continue

        img    = sitk.ReadImage(path)
        voxels = _count(img)
        same   = _same_grid(pa_clean, img)
        overlap = (int(np.count_nonzero((sitk.GetArrayViewFromImage(img) > 0) & pa_arr))
                   if same else None)
        print(f"  [AUX] {name:<10}: {voxels} voxels | same grid as PA: {same}"
              f"{'' if overlap is None else f' | overlap with PA: {overlap}'}")
        if not same:
            print(f"  [WARN] {name} is not on the same grid as the PA mask. "
                  "Re-run resampling.py for this case.")
        if voxels < _MIN_VOXELS:
            print(f"  [WARN] {name} has fewer than {_MIN_VOXELS} voxels.")
        report[name] = {"present": True, "voxels": voxels,
                        "same_grid": bool(same), "overlap_with_pa_voxels": overlap}
    return report


def extract_pa(pa_path: str, output_dir: str) -> str:
    """Clean the resampled PA mask, write pa.nii.gz and pipeline_mode.json."""
    if not os.path.isfile(pa_path):
        raise FileNotFoundError(f"PA mask not found: '{pa_path}'")
    os.makedirs(output_dir, exist_ok=True)

    image = sitk.ReadImage(pa_path)
    print(f"\n[PA] File    : {pa_path}")
    print(f"[PA] Size    : {image.GetSize()}")
    print(f"[PA] Spacing : {image.GetSpacing()}")

    print("\n-- Cleaning PA mask --")
    pa_clean, pa_info = _clean_pa(image)
    pa_clean = sitk.Cast(pa_clean, sitk.sitkUInt8) * PA_LABEL
    pa_clean.CopyInformation(image)

    sx, sy, sz = image.GetSpacing()
    volume_ml  = pa_info["pa_voxels_clean"] * sx * sy * sz / 1000.0
    print(f"  [PA] Volume      : {volume_ml:.1f} mL")

    print("\n-- Checking auxiliary masks --")
    aux = _check_aux_masks(pa_clean, output_dir)

    out_path = os.path.join(output_dir, "pa.nii.gz")
    sitk.WriteImage(pa_clean, out_path)
    print(f"\nSaved cleaned PA mask -> {out_path}")

    mode_json = os.path.join(output_dir, "pipeline_mode.json")
    with open(mode_json, "w") as f:
        json.dump({
            "mode":      "pulmonary_artery",
            "pa_label":  PA_LABEL,
            "pa_volume_ml": round(volume_ml, 2),
            **pa_info,
            "aux_masks": aux,
        }, f, indent=2)
    print(f"Saved QC summary      -> {mode_json}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean the resampled pulmonary artery mask (largest component + hole filling)."
    )
    parser.add_argument("pa_path", help="Path to resampled_pa.nii.gz")
    parser.add_argument("--output-dir", required=True,
                        help="Per-case output directory (also holds the other resampled masks).")
    args = parser.parse_args()

    try:
        extract_pa(args.pa_path, args.output_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)