import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} {message}")
    sys.stdout.flush()


def is_processed(raw_path: str) -> Tuple[bool, bool]:
    base_noext = os.path.splitext(raw_path)[0]
    mzml_exists = os.path.isfile(base_noext + ".mzML") or os.path.isfile(base_noext + ".mzml")
    mgf_exists = os.path.isfile(base_noext + ".mgf") or os.path.isfile(base_noext + ".MGF")
    return mzml_exists, mgf_exists


def find_raw_directories(parent: str) -> List[str]:
    seen = set()
    dirs: List[str] = []
    for root, _, files in os.walk(parent):
        if any(f.lower().endswith(".raw") for f in files):
            if root not in seen:
                seen.add(root)
                dirs.append(root)
    dirs.sort()
    return dirs


def run_cmd(cmd: List[str], dry_run: bool = False) -> int:
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    log(f"RUN: {cmd_str}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
    return proc.returncode


def process_directory(
    subdir: str,
    scanheadsman_exe: str,
    thermo_exe: str,
    dry_run: bool,
) -> None:
    raw_files = sorted(
        os.path.join(subdir, f)
        for f in os.listdir(subdir)
        if f.lower().endswith(".raw") and os.path.isfile(os.path.join(subdir, f))
    )
    if not raw_files:
        log(f"No .raw files directly in {subdir}, skipping.")
        return

    to_process: List[str] = []
    for rawfile in raw_files:
        mzml_exists, mgf_exists = is_processed(rawfile)
        if not (mzml_exists and mgf_exists):
            to_process.append(rawfile)

    if not to_process:
        log(f"All outputs already present in {subdir}, skipping.")
        return

    # Run ScanHeadsman once per folder if any work is needed
    log(f"Running ScanHeadsman on {subdir}")
    rc = run_cmd(["mono", scanheadsman_exe, subdir], dry_run=dry_run)
    if rc != 0:
        log(f"WARNING: ScanHeadsman returned non-zero exit code {rc} for {subdir}")

    # Convert per file, idempotently
    for rawfile in to_process:
        base_noext = os.path.splitext(rawfile)[0]
        mzml_path_a = base_noext + ".mzML"
        mzml_path_b = base_noext + ".mzml"
        mgf_path_a = base_noext + ".mgf"
        mgf_path_b = base_noext + ".MGF"

        if not (os.path.isfile(mzml_path_a) or os.path.isfile(mzml_path_b)):
            log(f"Converting to mzML: {rawfile}")
            rc = run_cmd(
                ["mono", thermo_exe, f"-i={rawfile}", f"-o={subdir}", "--excludeExceptionData"],
                dry_run=dry_run,
            )
            if rc != 0:
                log(f"ERROR: ThermoRawFileParser mzML conversion failed (rc={rc}) for {rawfile}")

        if not (os.path.isfile(mgf_path_a) or os.path.isfile(mgf_path_b)):
            log(f"Converting to MGF: {rawfile}")
            rc = run_cmd(
                ["mono", thermo_exe, f"-i={rawfile}", f"-o={subdir}", "-f=0"], dry_run=dry_run
            )
            if rc != 0:
                log(f"ERROR: ThermoRawFileParser MGF conversion failed (rc={rc}) for {rawfile}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process Thermo RAW files across mixed MSV structures."
    )
    parser.add_argument(
        "--parent",
        required=True,
        help="Parent folder containing MSV directories and/or RAWs",
    )
    parser.add_argument(
        "--scanheadsman",
        required=True,
        help="Path to ScanHeadsman.exe (download from: https://github.com/caetera/ScanHeadsman)",
    )
    parser.add_argument(
        "--thermo",
        required=True,
        help="Path to ThermoRawFileParser.exe (download from: https://github.com/compomics/ThermoRawFileParser)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")

    args = parser.parse_args()

    parent = args.parent
    if not parent or not os.path.isdir(parent):
        log(f"ERROR: parent directory is invalid: '{parent}'")
        return 2

    log(f"Processing MSV folders inside: {parent}")
    raw_dirs = find_raw_directories(parent)
    if not raw_dirs:
        log("No directories containing .raw files were found.")
        return 0

    for subdir in raw_dirs:
        log(f"=== Processing subfolder: {subdir} ===")
        try:
            process_directory(
                subdir=subdir,
                scanheadsman_exe=args.scanheadsman,
                thermo_exe=args.thermo,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # pragma: no cover
            log(f"ERROR: Exception while processing {subdir}: {exc}")

    log("=== All processing complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
