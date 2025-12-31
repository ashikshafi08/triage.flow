"""Git repository cloning utilities - tinygrad-style rewrite (97→45 lines)"""
import tempfile, shutil, subprocess
from contextlib import contextmanager
from typing import Tuple

def _git_clone(repo_url: str, branch: str, shallow: bool, dest: str) -> None:
    """Core clone logic with main/master fallback."""
    cmd = ["git", "clone"] + (["--depth", "1"] if shallow else []) + ["--branch", branch, repo_url, dest]
    try: subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        if branch != "main": raise
        cmd[-3] = "master"  # Try master if main fails
        subprocess.run(cmd, check=True, capture_output=True, text=True)

@contextmanager
def clone_repo_to_temp(repo_url: str, branch: str = "main", shallow: bool = True):
    """Clone repo to temp dir, yield path, cleanup on exit."""
    temp_dir = tempfile.mkdtemp()
    try:
        _git_clone(repo_url, branch, shallow, temp_dir)
        yield temp_dir
    finally: shutil.rmtree(temp_dir, ignore_errors=True)

def clone_repo_to_temp_persistent(repo_url: str, branch: str = "main", shallow: bool = True) -> str:
    """Clone repo to temp dir, return path (caller must cleanup)."""
    temp_dir = tempfile.mkdtemp()
    _git_clone(repo_url, branch, shallow, temp_dir)
    return temp_dir

def unshallow_repository(repo_path: str) -> bool:
    """Convert shallow repo to full history."""
    try:
        result = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], capture_output=True, text=True, cwd=repo_path)
        if result.returncode == 0 and result.stdout.strip() == "true":
            subprocess.run(["git", "fetch", "--unshallow"], check=True, capture_output=True, text=True, cwd=repo_path)
            return True
        return False
    except subprocess.CalledProcessError: return False

def get_repo_info(repo_url: str) -> Tuple[str, str]:
    """Extract (owner, repo) from GitHub URL."""
    parts = repo_url.replace(".git", "").split("/")
    return parts[-2], parts[-1]
