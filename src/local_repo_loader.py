import tempfile
import shutil
import subprocess
import os
from contextlib import contextmanager
from typing import Tuple

@contextmanager
def clone_repo_to_temp(repo_url: str, branch: str = "main", shallow: bool = True):
    """
    Clone a GitHub repo to a temporary directory, yield the path, and clean up after use.
    Usage:
        with clone_repo_to_temp(repo_url) as repo_path:
            # process files in repo_path
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Try main branch first
        cmd = ["git", "clone"]
        if shallow:
            cmd.extend(["--depth", "1"])
        cmd.extend(["--branch", branch, repo_url, temp_dir])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            # If main fails, try master
            if branch == "main":
                cmd_master = ["git", "clone"]
                if shallow:
                    cmd_master.extend(["--depth", "1"])
                cmd_master.extend(["--branch", "master", repo_url, temp_dir])
                subprocess.run(cmd_master, check=True, capture_output=True, text=True)
            else:
                raise
        
        yield temp_dir
    finally:
        # Clean up the temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

# Persistent version for session-based repo storage
def clone_repo_to_temp_persistent(repo_url: str, branch: str = "main", shallow: bool = True) -> str:
    temp_dir = tempfile.mkdtemp()
    cmd = ["git", "clone"]
    
    if shallow:
        cmd.extend(["--depth", "1"])
    
    cmd.extend(["--branch", branch, repo_url, temp_dir])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        # Try master if main fails
        if branch == "main":
            cmd_master = ["git", "clone"]
            if shallow:
                cmd_master.extend(["--depth", "1"])
            cmd_master.extend(["--branch", "master", repo_url, temp_dir])
            subprocess.run(cmd_master, check=True, capture_output=True, text=True)
        else:
            raise
    return temp_dir

def unshallow_repository(repo_path: str) -> bool:
    """Convert a shallow repository to a full repository with complete history"""
    try:
        # Check if it's a shallow repository
        result = subprocess.run([
            "git", "rev-parse", "--is-shallow-repository"
        ], capture_output=True, text=True, cwd=repo_path)
        
        if result.returncode == 0 and result.stdout.strip() == "true":
            print(f"Repository is shallow, fetching complete history...")
            # Fetch the complete history
            subprocess.run([
                "git", "fetch", "--unshallow"
            ], check=True, capture_output=True, text=True, cwd=repo_path)
            print(f"Repository unshallowed successfully")
            return True
        else:
            print(f"Repository already has complete history")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to unshallow repository: {e}")
        return False

def get_repo_info(repo_url: str) -> Tuple[str, str]:
    """Extract owner and repository name from a GitHub URL."""
    # Remove .git if present
    repo_url = repo_url.replace(".git", "")
    # Split by / and get last two parts
    parts = repo_url.split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo 