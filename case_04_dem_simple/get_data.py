# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Laboratory of Intelligent Maintenance and Operations Systems (IMOS), 2025.
# Authors: Vinay Sharma and Olga Fink
# Released under the Non-Commercial License Agreement in LICENSE.txt.

#!/usr/bin/env python3
"""
download_data.py
================
Downloads, extracts, and organises DEM simulation data from:
    https://zenodo.org/records/19691595

This script will create a `data/` folder in its current location
and download all contents into it.

Usage
-----
    python download_data.py --all
    python download_data.py --homogeneous
    python download_data.py --heterogeneous gravity
    python download_data.py --all --keep-zip

Flags
-----
  --all
      Download and organise every dataset (PDF + all datasets).

  --homogeneous
      Downloads and organises:
        • 60-sphere homogeneous interaction  →  data/homogeneous/{training,validation,extrapolation}/
        • 1-sphere wall collision benchmark  →  data/homogeneous/benchmark/oblique_wall_collisions/{10_deg … 90_deg}/
        • 2-sphere oblique collision         →  data/homogeneous/benchmark/oblique_sphere_collisions/

  --heterogeneous gravity
      Downloads and organises:
        • 60-sphere gravity dataset          →  data/heterogeneous/gravity/{training,validation,extrapolation}/
        • 2073-sphere rotating cylinder      →  data/heterogeneous/gravity/rotating_cylinder/

  --keep-zip
      Keep .zip archives after extraction (deleted by default).

Final folder layout
-------------------
data/
├── download_data.py
├── Detailed_Information_on_Data_Structure.pdf          (--all only)
│
├── homogeneous/                                        (--homogeneous / --all)
│   ├── training/
│   │   ├── case_01/   …   case_05/
│   ├── validation/
│   │   └── case_06/
│   ├── extrapolation/
│   │   └── case_07/
│   └── benchmark/
│       ├── oblique_wall_collisions/
│       │   ├── 10_deg/
│       │   ├── 30_deg/
│       │   ├── 45_deg/
│       │   ├── 60_deg/
│       │   └── 90_deg/
│       └── oblique_sphere_collisions/
│           └── data_at_timestep_000.csv … (100 files)
│
# └── heterogeneous/                                      (--heterogeneous gravity / --all)
#     └── gravity/
#         ├── training/
#         │   ├── case_01/   …   case_05/
#         ├── validation/
#         │   └── case_06/
#         ├── extrapolation/
#         │   └── case_07/
#         └── rotating_cylinder/
#             └── data_at_timestep_000.csv … (2000 files)
"""
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://zenodo.org/records/19691595/files"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _dl_hook(block_num: int, block_size: int, total_size: int) -> None:
    done = block_num * block_size
    bar_len = 40
    if total_size > 0:
        pct = min(done / total_size * 100, 100)
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r    [{bar}] {pct:5.1f}%  {done / 1e6:.1f}/{total_size / 1e6:.1f} MB",
            end="",
            flush=True,
        )
    else:
        print(f"\r    {done / 1e6:.1f} MB …", end="", flush=True)


def _unzip_hook(done: int, total: int) -> None:
    pct = done / total * 100
    filled = int(40 * pct / 100)
    bar = "█" * filled + "░" * (40 - filled)
    print(f"\r    [{bar}] {pct:5.1f}%  ({done}/{total} files)", end="", flush=True)


def download(filename: str, dest_dir: Path) -> Path:
    """Download *filename* from Zenodo into *dest_dir*. Returns local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / filename

    if dest_file.exists():
        print(f"  [skip] {filename}  (already downloaded)")
        return dest_file

    url = f"{BASE_URL}/{filename}?download=1"
    print(f"  [↓]   {filename}")
    try:
        urllib.request.urlretrieve(url, dest_file, reporthook=_dl_hook)
        print()
        print(f"  [ok]  saved → {dest_file.relative_to(SCRIPT_DIR)}")
    except Exception as exc:
        print()
        dest_file.unlink(missing_ok=True)
        print(f"  [error] {filename}: {exc}", file=sys.stderr)
        sys.exit(1)
    return dest_file


def extract_zip(zip_path: Path) -> Path:
    """Extract *zip_path* into a fresh temporary directory. Returns that dir."""
    tmp = Path(tempfile.mkdtemp(prefix="dem_extract_"))
    print(f"  [unzip] {zip_path.name}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            for i, m in enumerate(members, 1):
                zf.extract(m, tmp)
                _unzip_hook(i, len(members))
        print()
    except zipfile.BadZipFile as exc:
        print()
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  [error] corrupt zip — {zip_path.name}: {exc}", file=sys.stderr)
        sys.exit(1)
    return tmp


def move(src: Path, dst: Path) -> None:
    """Move *src* to *dst*, creating parents as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    shutil.move(str(src), str(dst))


# ---------------------------------------------------------------------------
# Per-dataset download + organise functions
# ---------------------------------------------------------------------------

def _sentinel(path: Path) -> Path:
    return path.parent / f".{path.name}.done"

def _find_case_root(tmp, case_name):
    """
    Search *tmp* for a directory named *case_name* (e.g. 'CASE01') at any depth.
    Returns its parent directory (which holds all sibling CASExx folders).
    Prints a helpful tree and exits if not found.
    """
    import sys as _sys

    matches = [p for p in tmp.rglob(case_name) if p.is_dir()]
    if not matches:
        print(
            f"  [error] could not find '{case_name}' anywhere inside the extracted zip.",
            file=_sys.stderr,
        )
        print("  Extracted tree (first 30 entries):", file=_sys.stderr)
        for p in sorted(tmp.rglob("*"))[:30]:
            print(f"    {p.relative_to(tmp)}", file=_sys.stderr)
        _sys.exit(1)
    return matches[0].parent  # parent holds CASE01, CASE02, … as siblings


def get_homogeneous_main(keep_zip: bool) -> None:
    """
    Target layout:
        data/homogeneous/training/case_01  …  case_05
        data/homogeneous/validation/case_06
        data/homogeneous/extrapolation/case_07
    """
    filename = "RawData_60Spheres_Homogeneous_Interaction_Inside_Cuboidal_Enclosure.zip"
    dest_root = DATA_DIR / "homogeneous"
    sentinel = _sentinel(dest_root / "training")

    zip_path = download(filename, DATA_DIR / "_zips")

    if sentinel.exists():
        print(f"  [skip] homogeneous main dataset already organised.")
        if not keep_zip:
            zip_path.unlink(missing_ok=True)
        return

    tmp = extract_zip(zip_path)
    case_dir = _find_case_root(tmp, "CASE01")

    for i in range(1, 6):
        move(case_dir / f"CASE{i:02d}", dest_root / "training" / f"case_{i:02d}")
    move(case_dir / "CASE06", dest_root / "validation" / "case_06")
    move(case_dir / "CASE07", dest_root / "extrapolation" / "case_07")

    shutil.rmtree(tmp)
    sentinel.touch()
    print(
        f"  [ok]  homogeneous data organised under {dest_root.relative_to(SCRIPT_DIR)}/"
    )

    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"  [rm]  {filename} removed")


def get_benchmark_wall_collisions(keep_zip: bool) -> None:
    """
    Target layout:
        data/homogeneous/benchmark/oblique_wall_collisions/10_deg/
        ...
    """
    filename = "RawData_Benchmark_1Sphere_Multiple_Wall_Collision.zip"
    dest_root = DATA_DIR / "homogeneous" / "benchmark" / "oblique_wall_collisions"
    sentinel = _sentinel(dest_root)

    zip_path = download(filename, DATA_DIR / "_zips")

    if sentinel.exists():
        print(f"  [skip] oblique_wall_collisions already organised.")
        if not keep_zip:
            zip_path.unlink(missing_ok=True)
        return

    tmp = extract_zip(zip_path)

    angle_names = {"10", "30", "45", "60", "90"}
    found_root = None
    for candidate in [tmp, *tmp.iterdir()]:
        if candidate.is_dir():
            children = {p.name for p in candidate.iterdir() if p.is_dir()}
            if children & angle_names:
                found_root = candidate
                break

    if found_root is None:
        print(
            f"  [error] could not locate angle folders in {filename}", file=sys.stderr
        )
        sys.exit(1)

    dest_root.mkdir(parents=True, exist_ok=True)
    for angle in sorted(angle_names, key=int):
        src = found_root / angle
        if src.exists():
            move(src, dest_root / f"{angle}_deg")
        else:
            print(f"  [warn] angle folder {angle}° not found in zip — skipping")

    shutil.rmtree(tmp)
    sentinel.touch()
    print(
        f"  [ok]  oblique_wall_collisions organised under "
        f"{dest_root.relative_to(SCRIPT_DIR)}/"
    )

    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"  [rm]  {filename} removed")


def get_benchmark_oblique_sphere(keep_zip: bool) -> None:
    """
    Target layout:
        data/homogeneous/benchmark/oblique_sphere_collisions/data_at_timestep_000.csv …
    """
    filename = "RawData_Benchmark_2Spheres_Oblique_Collision.zip"
    dest_root = DATA_DIR / "homogeneous" / "benchmark" / "oblique_sphere_collisions"
    sentinel = _sentinel(dest_root)

    zip_path = download(filename, DATA_DIR / "_zips")

    if sentinel.exists():
        print(f"  [skip] oblique_sphere_collisions already organised.")
        if not keep_zip:
            zip_path.unlink(missing_ok=True)
        return

    tmp = extract_zip(zip_path)
    dest_root.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(tmp.rglob("*.csv"))
    if not csv_files:
        print(f"  [error] no CSV files found in {filename}", file=sys.stderr)
        sys.exit(1)

    for csv in csv_files:
        shutil.copy2(csv, dest_root / csv.name)

    shutil.rmtree(tmp)
    sentinel.touch()
    print(
        f"  [ok]  {len(csv_files)} timestep files organised under "
        f"{dest_root.relative_to(SCRIPT_DIR)}/"
    )

    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"  [rm]  {filename} removed")



def get_pdf() -> None:
    filename = "Detailed_Information_on_Data_Structure.pdf"
    download(filename, DATA_DIR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 62 - len(title))}")


def main() -> None:
    print(f"Root path: {SCRIPT_DIR}")
    print(f"Creating Data directory at: {DATA_DIR}")

    kz = False # Keep Zip is false

    print("\n=== Downloading & organising ALL datasets ===")
    section("Documentation")
    get_pdf()
    section("Homogeneous 60-sphere (training / validation / extrapolation)")
    get_homogeneous_main(kz)
    section("Benchmark — oblique wall collisions (1 sphere, 5 angles)")
    get_benchmark_wall_collisions(kz)
    section("Benchmark — oblique sphere collisions (2 spheres, oblique)")
    get_benchmark_oblique_sphere(kz)

    # Clean up empty _zips folder if all zips were deleted
    zips_dir = DATA_DIR / "_zips"
    if zips_dir.exists() and not any(zips_dir.iterdir()):
        zips_dir.rmdir()

    print("\n✓ Done.\n")


if __name__ == "__main__":
    main()