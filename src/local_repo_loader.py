import tempfile
import shutil
import subprocess
import os
from contextlib import contextmanager
from typing import Tuple

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
        # Try main branch first
        try:
            subprocess.run([
                "git", "clone",
                "--depth", "1",
                "--branch", branch,
                repo_url,
                temp_dir
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            # If main fails, try master
            if branch == "main":
                subprocess.run([
                    "git", "clone",
                    "--depth", "1",
                    "--branch", "master",
                    repo_url,
                    temp_dir
                ], check=True, capture_output=True, text=True)
            else:
                raise
        
        yield temp_dir
    finally:
        # Clean up the temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

# Persistent version for session-based repo storage
def clone_repo_to_temp_persistent(repo_url: str, branch: str = "main") -> str:
    temp_dir = tempfile.mkdtemp()
    try:
        subprocess.run([
            "git", "clone",
            "--depth", "1",
            "--branch", branch,
            repo_url,
            temp_dir
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        # Try master if main fails
        if branch == "main":
            subprocess.run([
                "git", "clone",
                "--depth", "1",
                "--branch", "master",
                repo_url,
                temp_dir
            ], check=True, capture_output=True, text=True)
        else:
            raise
    return temp_dir

def get_repo_info(repo_url: str) -> Tuple[str, str]:
    """Extract owner and repository name from a GitHub URL."""
    # Remove .git if present
    repo_url = repo_url.replace(".git", "")
    # Split by / and get last two parts
    parts = repo_url.split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo 