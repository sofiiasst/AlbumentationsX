#!/usr/bin/env python3
"""Create dataset ZIP from a run directory with proper path handling."""

import zipfile
from pathlib import Path
import sys


def create_dataset_zip(run_num: int):
    """Create dataset_{run_num}.zip from outputs/run_{run_num}."""
    repo_root = Path(__file__).resolve().parent
    run_dir = repo_root / "outputs" / f"run_{run_num}"
    zip_path = repo_root / "outputs" / f"dataset_{run_num}.zip"
    
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        sys.exit(1)
    
    print(f"Creating {zip_path} from {run_dir}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in run_dir.rglob('*'):
            if file_path.is_file():
                # Calculate relative path and convert backslashes to forward slashes
                rel_path = file_path.relative_to(run_dir)
                arcname = rel_path.as_posix()  # Converts to forward slashes
                
                print(f"  Adding {arcname}")
                zf.write(file_path, arcname)
    
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"✓ Successfully created {zip_path} ({zip_size_mb:.2f} MB)")


if __name__ == "__main__":
    run_num = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    create_dataset_zip(run_num)
