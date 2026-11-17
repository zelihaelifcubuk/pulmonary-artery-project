"""
check_pa_masks.py
=================
Diagnostic for the whole dataset. For each case it checks the CT and the original
pulmonary_artery mask and reports one status per case. It never stops on a bad file.

    python check_pa_masks.py          # all cases in dataset/
    python check_pa_masks.py 10       # first 10 cases
    python check_pa_masks.py s0058    # one case

Output: a table on screen and pa_mask_check.csv next to this script.
Statuses: OK, EMPTY, TINY, GEOMETRY, LOST, MISSING, CT_ERROR, PA_ERROR
Run it from the folder that contains dataset/.
"""
import csv
import os
import sys

import numpy as np
import SimpleITK as sitk

HERE        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")
PA_REL      = os.path.join("heartchambers_highres", "pulmonary_artery.nii.gz")
MIN_PA_ML   = 10.0      # smaller than this = TINY (good cases here are about 40-80 mL)


def _info(path):
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    return r.GetSize(), r.GetSpacing(), r.GetOrigin(), r.GetDirection()


def _same(a, b, tol=1e-3):
    return (tuple(a[0]) == tuple(b[0]) and np.allclose(a[1], b[1], atol=tol)
            and np.allclose(a[2], b[2], atol=tol) and np.allclose(a[3], b[3], atol=tol))


def _short(e):
    return str(e).strip().splitlines()[-1][:70] if str(e).strip() else type(e).__name__


def _other_heart_masks(case):
    """Voxel counts of aorta and right ventricle from the same heartchambers_highres model."""
    out = {}
    for name in ("aorta", "heart_ventricle_right"):
        path = os.path.join(DATASET_DIR, case, "heartchambers_highres", f"{name}.nii.gz")
        try:
            out[name] = int(np.count_nonzero(sitk.GetArrayFromImage(sitk.ReadImage(path))))
        except Exception:
            out[name] = None
    return out


def _explain(case):
    c = _other_heart_masks(case)
    if all(v is None for v in c.values()):
        return "other heart masks unreadable"
    if all((v or 0) < 100 for v in c.values()):
        return "aorta and RV also empty: heart probably not in the CT -> exclude"
    return "aorta/RV present: only the PA label failed -> regenerate it"


def check_case(case):
    """Returns dict(status, voxels, ml, same_grid, kept, note)."""
    ct_path = os.path.join(DATASET_DIR, case, "ct.nii.gz")
    pa_path = os.path.join(DATASET_DIR, case, PA_REL)
    row = dict(case=case, status="", voxels="", ml="", same_grid="", kept="", note="")

    if not os.path.isfile(pa_path):
        row.update(status="MISSING", note="PA file missing")
        return row
    try:
        ct_info = _info(ct_path)
    except Exception as e:
        row.update(status="CT_ERROR", note=f"CT unreadable: {_short(e)}")
        return row
    try:
        pa_info = _info(pa_path)
        pa      = sitk.ReadImage(pa_path)
    except Exception as e:
        row.update(status="PA_ERROR", note=f"PA unreadable: {_short(e)}")
        return row

    n_orig = int(np.count_nonzero(sitk.GetArrayFromImage(pa)))
    ml     = n_orig * float(np.prod(pa.GetSpacing())) / 1000.0
    same   = _same(ct_info, pa_info)
    row.update(voxels=n_orig, ml=round(ml, 1), same_grid=same)

    if n_orig == 0:
        row.update(status="EMPTY", note="original mask is empty; " + _explain(case))
        return row

    rs = sitk.ResampleImageFilter()
    rs.SetSize(ct_info[0]); rs.SetOutputSpacing(ct_info[1])
    rs.SetOutputOrigin(ct_info[2]); rs.SetOutputDirection(ct_info[3])
    rs.SetInterpolator(sitk.sitkNearestNeighbor); rs.SetDefaultPixelValue(0)
    rs.SetTransform(sitk.Transform())
    resampled = rs.Execute(pa)                      # keep the image alive while counting
    n_after   = int(np.count_nonzero(sitk.GetArrayFromImage(resampled)))
    ml_after  = n_after * float(np.prod(ct_info[1])) / 1000.0
    kept      = 100.0 * ml_after / ml if ml > 0 else 0.0
    row["kept"] = round(kept)

    if not same and kept < 95:
        row.update(status="GEOMETRY", note=f"grid differs from CT; {ml:.1f} mL -> {ml_after:.1f} mL")
    elif kept < 95:
        row.update(status="LOST", note="voxels lost although grids match")
    elif ml < MIN_PA_ML:
        row.update(status="TINY", note=f"only {ml:.1f} mL (< {MIN_PA_ML:g} mL); " + _explain(case))
    else:
        row.update(status="OK")
    return row


def main():
    cases = sorted(d for d in os.listdir(DATASET_DIR)
                   if os.path.isdir(os.path.join(DATASET_DIR, d)))
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        cases = [arg] if arg in cases else cases[:int(arg)]

    print(f"{'case':<8}{'status':<10}{'PA voxels':>10}{'mL':>8}{'same grid':>11}{'kept %':>8}  note")
    print("-" * 90, flush=True)
    rows = []
    for case in cases:
        try:
            row = check_case(case)
        except Exception as e:                       # never let one case stop the scan
            row = dict(case=case, status="ERROR", voxels="", ml="", same_grid="", kept="",
                       note=_short(e))
        rows.append(row)
        print(f"{row['case']:<8}{row['status']:<10}{str(row['voxels']):>10}{str(row['ml']):>8}"
              f"{str(row['same_grid']):>11}{str(row['kept']):>8}  {row['note']}", flush=True)

    out_csv = os.path.join(HERE, "pa_mask_check.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "status", "voxels", "ml", "same_grid", "kept", "note"])
        w.writeheader(); w.writerows(rows)

    print("\nSummary")
    for status in sorted({r["status"] for r in rows}):
        print(f"  {status:<9}: {sum(1 for r in rows if r['status'] == status)}")
    print(f"  total    : {len(rows)}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()