import os
import shutil
import pathlib


def remove_pycache(directory_path: str):
    """Recursively traverses the provided directory and removes all __pycache__
    subdirectories.
    """
    for root, dirs, files in os.walk(directory_path):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            print("Removing:", pycache_path)
            shutil.rmtree(pycache_path)
        dirs[:] = [d for d in dirs if d != "__pycache__"]


def remove_dsstore(directory_path: str):
    """Recursively traverses the provided directory and removes all .DS_Store
    files.
    """
    for root, dirs, files in os.walk(directory_path):
        if ".DS_Store" in files:
            dsstore_path = os.path.join(root, ".DS_Store")
            print("Removing:", dsstore_path)
            os.remove(dsstore_path)


def remove_ruff(directory_path: str):
    """Recursively traverses the provided directory and removes all .ruff_cache
    directories.
    """
    for root, dirs, files in os.walk(directory_path):
        if ".ruff_cache" in dirs:
            pycache_path = os.path.join(root, ".ruff_cache")
            print("Removing:", pycache_path)
            shutil.rmtree(pycache_path)
        dirs[:] = [d for d in dirs if d != ".ruff_cache"]


def remove_pytest(directory_path: str):
    """Recursively traverses the provided directory and removes all .pytest_cache
    directories.
    """
    for root, dirs, files in os.walk(directory_path):
        if ".pytest_cache" in dirs:
            pycache_path = os.path.join(root, ".pytest_cache")
            print("Removing:", pycache_path)
            shutil.rmtree(pycache_path)
        dirs[:] = [d for d in dirs if d != ".pytest_cache"]


if __name__ == "__main__":
    project_dir = pathlib.Path(__file__).resolve().parent.parent
    remove_pycache(project_dir)
    remove_dsstore(project_dir)
    remove_ruff(project_dir)
    remove_pytest(project_dir)
