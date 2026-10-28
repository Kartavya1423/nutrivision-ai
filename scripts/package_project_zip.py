"""NutriVision AI — Project Zip Packager.

Packages the full codebase, tests, configurations, reports, notebooks,
and environment files into a clean, lightweight 'NutriVision-AI-Latest.zip'
archive without large dataset images or multi-gigabyte checkpoints.
"""

from __future__ import annotations

import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP_NAME = "NutriVision-AI-Latest.zip"
OUTPUT_ZIP_PATH = PROJECT_ROOT / OUTPUT_ZIP_NAME

# Directories to exclude entirely
EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "data",
    "models",
    "outputs",
    "runs",
    "temp_tree",
    "examples_video",
}

# File patterns or large binary extensions to exclude
EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".bin",
    ".tar",
    ".gz",
    ".onnx",
    ".mp4",
    ".avi",
    ".mov",
    ".zip",
    ".tmp",
}

# Specific files to exclude (e.g. self-zip)
EXCLUDE_FILES = {
    OUTPUT_ZIP_NAME,
    f"{OUTPUT_ZIP_NAME}.tmp.zip",
    "NutriVision-AI.zip",
    "sam2.1_t.pt",
    "yolo11n.pt",
}


def should_include(rel_path: Path) -> bool:
    """Determine if a relative path should be included in the archive."""
    parts = rel_path.parts
    # Check directory exclusion
    for excluded in EXCLUDE_DIRS:
        if excluded in parts:
            return False

    # Check file extension exclusion
    if rel_path.suffix.lower() in EXCLUDE_EXTENSIONS or ".tmp" in rel_path.name.lower():
        return False

    # Check specific file exclusion
    if rel_path.name in EXCLUDE_FILES:
        return False

    return True


def create_project_zip(output_path: Path = OUTPUT_ZIP_PATH) -> tuple[int, int]:
    """Create or update the project zip archive.
    
    Returns:
        tuple[int, int]: (number_of_files_added, zip_size_bytes)
    """
    files_added = 0
    temp_zip_path = output_path.with_suffix(".tmp.zip")

    try:
        with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for root, dirs, files in os.walk(PROJECT_ROOT):
                # Prune excluded directories in-place to prevent traversing into them
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(PROJECT_ROOT)

                    if should_include(rel_path):
                        # Archive with standard POSIX separators inside zip
                        archive_name = rel_path.as_posix()
                        zipf.write(file_path, arcname=archive_name)
                        files_added += 1

        # Atomically replace final zip
        if output_path.exists():
            output_path.unlink()
        temp_zip_path.rename(output_path)
        zip_size = output_path.stat().st_size
        return files_added, zip_size

    finally:
        if temp_zip_path.exists():
            temp_zip_path.unlink()


def sync_parent_zip(parent_zip_path: Path = PROJECT_ROOT.parent / "NutriVision-AI.zip") -> tuple[int, int] | None:
    """Synchronize parent directory archive with top-level folder prefix if it exists."""
    if not parent_zip_path.exists():
        return None

    temp_zip = parent_zip_path.with_suffix(".sync.tmp.zip")
    files_written = 0

    try:
        # Build set of workspace files
        ws_files = {}
        for root, dirs, files in os.walk(PROJECT_ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(PROJECT_ROOT)
                if should_include(rel_path):
                    ws_files[f"NutriVision-AI/{rel_path.as_posix()}"] = file_path

        # Preserve any existing special assets in parent zip (e.g. models, splits, outputs)
        preserved = {}
        with zipfile.ZipFile(parent_zip_path, "r") as bz:
            for info in bz.infolist():
                if not info.is_dir() and info.filename not in ws_files:
                    preserved[info.filename] = info

        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as nz:
            for arcname, file_path in sorted(ws_files.items()):
                nz.write(file_path, arcname=arcname)
                files_written += 1
            with zipfile.ZipFile(parent_zip_path, "r") as bz:
                for arcname, info in sorted(preserved.items()):
                    nz.writestr(info, bz.read(info.filename))
                    files_written += 1

        if parent_zip_path.exists():
            parent_zip_path.unlink()
        temp_zip.rename(parent_zip_path)
        return files_written, parent_zip_path.stat().st_size

    finally:
        if temp_zip.exists():
            temp_zip.unlink()


if __name__ == "__main__":
    start_time = datetime.now()
    count, size_bytes = create_project_zip()
    size_mb = size_bytes / (1024 * 1024)
    duration = (datetime.now() - start_time).total_seconds()

    print(f"[SUCCESS] Packaged {count} files into '{OUTPUT_ZIP_PATH}'")
    print(f"Archive Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    print(f"Time Taken: {duration:.2f}s")

    # Sync parent zip if present
    parent_res = sync_parent_zip()
    if parent_res:
        p_count, p_bytes = parent_res
        p_mb = p_bytes / (1024 * 1024)
        print(f"[SUCCESS] Synchronized {p_count} entries to '{PROJECT_ROOT.parent / 'NutriVision-AI.zip'}' ({p_mb:.2f} MB)")

