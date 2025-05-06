import tempfile
import shutil
import subprocess
import os
from contextlib import contextmanager

@contextmanager
def clone_repo_to_temp(repo_url: str, branch: str = "main"):
    """
    Clone a GitHub repo to a temporary directory, yield the path, and clean up after use.
    Usage:
        with clone_repo_to_temp(repo_url) as repo_path:
            # process files in repo_path
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Clone the repo (shallow clone for speed)
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", branch, repo_url, temp_dir
        ], check=True)
        yield temp_dir
    finally:
        # Clean up the temp directory
        shutil.rmtree(temp_dir, ignore_errors=True) 