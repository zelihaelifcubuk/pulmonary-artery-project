"""
resampling.py  (step 1)
=======================
Resamples the CT to 1 mm isotropic, then puts every mask on that CT grid.

Two ways to run it:
    python resampling.py                          batch: first NUM_CASES cases of dataset/
    python resampling.py <case_dir> <output_dir>  one case (this is how run_pipeline.py calls it)

Outputs per case:
    resampled_ct.nii.gz, resampled_pa.nii.gz, resampled_aorta.nii.gz, resampled_rv.nii.gz,
    resampled_lung_left.nii.gz, resampled_lung_right.nii.gz
"""
import os
import sys

import SimpleITK as sitk

# ── Settings (batch mode only) ─────────────────────────────────────────────
NUM_CASES = 10          # how many cases to process: 10 = first 10, 102 = all of them

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPTS_DIR, "dataset")   # dataset/ next to this script
OUTPUTS_DIR = os.path.join(SCRIPTS_DIR, "outputs")   # outputs/ next to this script

# Masks taken from <case>/heartchambers_highres/  (output name, file name, required)
_HC_MASKS = [
    ("resampled_pa.nii.gz",    "pulmonary_artery.nii.gz",      True),
    ("resampled_aorta.nii.gz", "aorta.nii.gz",                 False),
    ("resampled_rv.nii.gz",    "heart_ventricle_right.nii.gz", False),
]

# Lung lobes taken from <case>/segmentations/, merged per side
_LUNG_LOBES = {
    "resampled_lung_left.nii.gz":  ["lung_upper_lobe_left.nii.gz",
                                    "lung_lower_lobe_left.nii.gz"],
    "resampled_lung_right.nii.gz": ["lung_upper_lobe_right.nii.gz",
                                    "lung_middle_lobe_right.nii.gz",
                                    "lung_lower_lobe_right.nii.gz"],
}


def resample_ct(ct_path: str, output_dir: str,
                new_spacing: tuple = (1.0, 1.0, 1.0)) -> str:
    """Resample CT to new_spacing (linear, -1024 HU background). Returns output path."""
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT file not found: '{ct_path}'")

    print(f"[CT]  Input   : {ct_path}")
    image = sitk.ReadImage(ct_path)
    print(f"      Size    : {image.GetSize()}  Spacing: {image.GetSpacing()}")

    new_size = tuple(
        int(round(image.GetSize()[i] * image.GetSpacing()[i] / new_spacing[i]))
        for i in range(3)
    )

    rs = sitk.ResampleImageFilter()
    rs.SetSize(new_size)
    rs.SetOutputSpacing(new_spacing)
    rs.SetOutputOrigin(image.GetOrigin())
    rs.SetOutputDirection(image.GetDirection())
    rs.SetTransform(sitk.Transform())
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(-1024.0)
    resampled = rs.Execute(image)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "resampled_ct.nii.gz")
    sitk.WriteImage(resampled, out_path)
    print(f"      New size: {resampled.GetSize()}")
    print(f"      Saved   : {out_path}")
    return out_path


def _to_ref_grid(mask: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """Nearest-neighbour resample of a mask onto the reference (CT) grid."""
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    rs.SetTransform(sitk.Transform())
    return sitk.Cast(rs.Execute(mask), sitk.sitkUInt8)


def resample_mask(mask_path: str, ref_ct_path: str, output_dir: str,
                  output_name: str, tag: str) -> str:
    """Resample one mask onto the resampled-CT grid."""
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"[{tag}] file not found: '{mask_path}'")

    print(f"\n[{tag}] Input   : {mask_path}")
    mask = sitk.ReadImage(mask_path)
    ref  = sitk.ReadImage(ref_ct_path)
    out  = _to_ref_grid(mask, ref)

    out_path = os.path.join(output_dir, output_name)
    sitk.WriteImage(out, out_path)
    print(f"      Saved   : {out_path}")
    return out_path


def resample_lung(lobe_paths: list, ref_ct_path: str, output_dir: str,
                  output_name: str, tag: str) -> str:
    """Resample lung lobes onto the CT grid and merge them into one binary mask."""
    ref    = sitk.ReadImage(ref_ct_path)
    merged = None
    print(f"\n[{tag}] Merging {len(lobe_paths)} lobes")
    for p in lobe_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[{tag}] lobe file not found: '{p}'")
        lobe   = _to_ref_grid(sitk.ReadImage(p), ref)
        lobe   = sitk.Cast(lobe > 0, sitk.sitkUInt8)
        merged = lobe if merged is None else sitk.Or(merged, lobe)

    out_path = os.path.join(output_dir, output_name)
    sitk.WriteImage(merged, out_path)
    print(f"      Saved   : {out_path}")
    return out_path


def resample_case(case_dir: str, output_dir: str,
                  new_spacing: tuple = (1.0, 1.0, 1.0)) -> dict:
    """
    Resample everything for one case folder (e.g. dataset/s0223).
    Returns {output_name: path} for all files written.
    """
    hc_dir  = os.path.join(case_dir, "heartchambers_highres")
    seg_dir = os.path.join(case_dir, "segmentations")
    outputs = {}

    # 1. CT first: it defines the grid for every mask
    ct_out = resample_ct(os.path.join(case_dir, "ct.nii.gz"), output_dir, new_spacing)
    outputs["resampled_ct.nii.gz"] = ct_out

    # 2. heartchambers_highres masks (PA required, others optional)
    for out_name, file_name, required in _HC_MASKS:
        path = os.path.join(hc_dir, file_name)
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Required mask missing: '{path}'")
            print(f"\n[WARN] Optional mask missing, skipped: {path}")
            continue
        outputs[out_name] = resample_mask(path, ct_out, output_dir,
                                          out_name, out_name.split("_")[1].upper())

    # 3. Lungs (merge lobes per side)
    for out_name, lobes in _LUNG_LOBES.items():
        paths = [os.path.join(seg_dir, f) for f in lobes]
        if not all(os.path.exists(p) for p in paths):
            print(f"\n[WARN] Lung lobe(s) missing, skipped: {out_name}")
            continue
        outputs[out_name] = resample_lung(paths, ct_out, output_dir, out_name,
                                          "LUNG-" + ("L" if "left" in out_name else "R"))
    return outputs


def list_cases(dataset_dir: str) -> list:
    """Case folders = sub-folders of dataset/ that contain ct.nii.gz (meta.csv is ignored)."""
    return sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, d, "ct.nii.gz"))
    )


if __name__ == "__main__":
    # ── Single case (called by run_pipeline.py) ────────────────────────────
    if len(sys.argv) == 3:
        case_dir, output_dir = sys.argv[1], sys.argv[2]
        os.makedirs(output_dir, exist_ok=True)
        results = resample_case(case_dir, output_dir)
        print("\n" + "=" * 60)
        print(" RESAMPLING COMPLETE.")
        for name in results:
            print(f"   {name}")
        print(f" Outputs saved to: {os.path.abspath(output_dir)}")
        print("=" * 60)
        sys.exit(0)

    if len(sys.argv) != 1:
        print("Usage: python resampling.py                          (batch, uses NUM_CASES)")
        print("       python resampling.py <case_dir> <output_dir>  (single case)")
        sys.exit(1)

    # ── Batch mode ─────────────────────────────────────────────────────────
    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] Dataset folder not found: {DATASET_DIR}")
        sys.exit(1)

    all_cases = list_cases(DATASET_DIR)
    cases = all_cases[:NUM_CASES]
    print(f"Found {len(all_cases)} case(s) in {DATASET_DIR}. Processing {len(cases)}.")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    results = {}
    for i, case in enumerate(cases, start=1):
        print("\n" + "=" * 60)
        print(f"  [{i}/{len(cases)}] CASE: {case}")
        print("=" * 60)
        out_dir = os.path.join(OUTPUTS_DIR, case)          # outputs/<case_name>/
        os.makedirs(out_dir, exist_ok=True)
        try:
            resample_case(os.path.join(DATASET_DIR, case), out_dir)
            results[case] = True
        except Exception as e:                              # one bad case won't stop the batch
            print(f"  [ERROR] {case}: {e}")
            results[case] = False

    print("\n" + "=" * 60)
    print(" RESAMPLING SUMMARY")
    print("=" * 60)
    for case, ok in results.items():
        print(f"  [{'OK    ' if ok else 'FAILED'}] {case}")
    print(f" Outputs saved to: {OUTPUTS_DIR}")
    print("=" * 60)