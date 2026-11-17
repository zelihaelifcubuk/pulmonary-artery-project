"""
create_pa_mask.py
=================
Runs TotalSegmentator's heartchambers_highres task on every case and makes sure the
pulmonary artery (PA) mask that comes out actually contains a PA.

Compared with the first version:
  * A case counts as done only if its PA mask is non-empty and at least MIN_PA_ML.
    Empty or tiny masks from an earlier run are retried, not skipped.
  * Every result is validated. TotalSegmentator can finish without an error and still
    write empty masks (for example when it finds no heart), so "no exception" is not success.
  * If the first attempt (MPS) fails or gives an empty PA, the case is retried on the CPU.
  * Unreadable CTs are detected up front instead of failing inside the model.
  * Every case is written to create_pa_mask_log.csv (status, device used, PA volume, note,
    TotalSegmentator version). Failed cases are classified as "heart not in the CT" or
    "only the PA is missing".

Usage:
    python create_pa_mask.py                            # default dataset folder below
    python create_pa_mask.py /path/to/dataset           # use another dataset folder
    python create_pa_mask.py --only s0250 s0419         # only these cases
    python create_pa_mask.py --retry-device none        # no CPU retry
    python create_pa_mask.py --device cpu               # run everything on the CPU
"""
import argparse
import csv
import datetime
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator

# ── Settings ───────────────────────────────────────────────────────────────
DEFAULT_DATASET_DIR = Path("/Users/elif/Desktop/pulmonary_artery/dataset")
TASK           = "heartchambers_highres"
PRIMARY_DEVICE = "mps"
RETRY_DEVICE   = "cpu"        # used when the first attempt fails; "none" disables the retry
MIN_PA_ML      = 20.0         # good cases in this dataset have about 40-165 mL
LOG_PATH       = Path(__file__).resolve().parent / "create_pa_mask_log.csv"


def mask_ml(path: Path):
    """Volume of a mask in mL; None if the file is missing or unreadable."""
    if not path.is_file():
        return None
    try:
        img = sitk.ReadImage(str(path))
        n = int(np.count_nonzero(sitk.GetArrayFromImage(img)))
        return n * float(np.prod(img.GetSpacing())) / 1000.0
    except Exception:
        return None


def ct_readable(ct_file: Path):
    try:
        r = sitk.ImageFileReader()
        r.SetFileName(str(ct_file))
        r.ReadImageInformation()
        return True, ""
    except Exception as e:
        lines = str(e).strip().splitlines()
        return False, (lines[-1] if lines else type(e).__name__)[:120]


def run_task(ct_file: Path, out_dir: Path, device: str):
    """Run TotalSegmentator. Returns an error message, or None if it finished."""
    try:
        totalsegmentator(input=str(ct_file), output=str(out_dir), task=TASK, device=device)
        return None
    except Exception as e:
        lines = str(e).strip().splitlines()
        return (lines[-1] if lines else type(e).__name__)[:120]


def classify_failure(out_dir: Path):
    aorta = mask_ml(out_dir / "aorta.nii.gz") or 0.0
    rv    = mask_ml(out_dir / "heart_ventricle_right.nii.gz") or 0.0
    if aorta < 5.0 and rv < 5.0:
        return "FAILED_NO_HEART", "aorta and RV also empty: heart probably not in the CT (exclude)"
    return "FAILED_PA_ONLY", "aorta/RV present but the PA is missing"


def process_case(subject_dir: Path, primary: str, retry, min_ml: float) -> dict:
    ct_file = subject_dir / "ct.nii.gz"
    out_dir = subject_dir / "heartchambers_highres"
    pa_file = out_dir / "pulmonary_artery.nii.gz"
    row = dict(case=subject_dir.name, status="", device="", pa_ml="", note="")

    ok, msg = ct_readable(ct_file)
    if not ok:
        row.update(status="CT_ERROR", note=f"CT unreadable: {msg}")
        return row

    ml = mask_ml(pa_file)
    if ml is not None and ml >= min_ml:
        row.update(status="SKIPPED", pa_ml=round(ml, 1), note="valid PA mask already exists")
        return row

    devices = [primary] + ([retry] if retry and retry != primary else [])
    notes = []
    for dev in devices:
        print(f"    running on {dev} ...", flush=True)
        err = run_task(ct_file, out_dir, dev)
        ml  = mask_ml(pa_file)
        if err:
            notes.append(f"{dev}: error ({err})")
        elif ml is None:
            notes.append(f"{dev}: no PA file written")
        elif ml < min_ml:
            notes.append(f"{dev}: PA only {ml:.1f} mL")
        else:
            row.update(status="OK", device=dev, pa_ml=round(ml, 1),
                       note="" if dev == primary else "; ".join(notes) + f"; fixed by {dev}")
            return row

    status, why = classify_failure(out_dir)
    row.update(status=status, pa_ml=round(ml, 1) if ml is not None else "",
               note="; ".join(notes) + "; " + why)
    return row


def main():
    parser = argparse.ArgumentParser(description="Create validated PA masks with TotalSegmentator.")
    parser.add_argument("dataset_dir", nargs="?", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--only", nargs="+", default=None, help="Process only these case names.")
    parser.add_argument("--device", default=PRIMARY_DEVICE, help="First attempt (default: mps).")
    parser.add_argument("--retry-device", default=RETRY_DEVICE,
                        help="Second attempt if the first fails (default: cpu; 'none' = no retry).")
    parser.add_argument("--min-ml", type=float, default=MIN_PA_ML,
                        help=f"Smallest PA volume counted as valid (default: {MIN_PA_ML:g} mL).")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"[ERROR] Dataset folder not found: {dataset_dir}")
        sys.exit(1)
    retry = None if args.retry_device.lower() == "none" else args.retry_device

    subjects = sorted(d for d in dataset_dir.iterdir()
                      if d.is_dir() and (d / "ct.nii.gz").exists())
    if args.only:
        subjects = [d for d in subjects if d.name in set(args.only)]
    try:
        ts_version = metadata.version("TotalSegmentator")
    except Exception:
        ts_version = "unknown"
    print(f"Found {len(subjects)} case(s) in {dataset_dir}")
    print(f"TotalSegmentator {ts_version} | task {TASK} | device {args.device}"
          f"{f' (retry: {retry})' if retry else ''} | valid PA >= {args.min_ml:g} mL\n")

    new_log = not LOG_PATH.exists()
    results = []
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "case", "status", "device",
                                               "pa_ml", "note", "totalsegmentator_version"])
        if new_log:
            writer.writeheader()
        for i, subject_dir in enumerate(subjects, start=1):
            t0 = time.time()
            print(f"[{i}/{len(subjects)}] {subject_dir.name}", flush=True)
            try:
                row = process_case(subject_dir, args.device, retry, args.min_ml)
            except Exception as e:                  # never let one case stop the run
                row = dict(case=subject_dir.name, status="ERROR", device="", pa_ml="",
                           note=str(e).strip().splitlines()[-1][:120] if str(e).strip() else "")
            results.append(row)
            print(f"    -> {row['status']}  {row['pa_ml']} mL  {row['note']}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
            writer.writerow({"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                             "totalsegmentator_version": ts_version, **row})
            f.flush()

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    for status in sorted({r["status"] for r in results}):
        names = [r["case"] for r in results if r["status"] == status]
        print(f"  {status:<16}: {len(names)}"
              f"{'  ' + ', '.join(names) if status not in ('OK', 'SKIPPED') else ''}")
    print(f"  Log: {LOG_PATH}")
    print("=" * 60)
    print("Next: run check_pa_masks.py again so pa_mask_check.csv matches the new masks.")


if __name__ == "__main__":
    main()