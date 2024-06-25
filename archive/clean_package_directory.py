import os
import shutil

def remove_pycache(directory_path: str):
    """Recursively traverses the provided directory and removes all __pycache__
    subdirectories.
    """
    for root, dirs, files in os.walk(directory_path):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            print("Removing:", pycache_path)
            shutil.rmtree(pycache_path)
        dirs[:] = [d for d in dirs if d != '__pycache__']


def remove_dsstore(directory_path: str):
    """Recursively traverses the provided directory and removes all .DS_Store
    files.
    """
    for root, dirs, files in os.walk(directory_path):
        if '.DS_Store' in files:
            dsstore_path = os.path.join(root, '.DS_Store')
            print("Removing:", dsstore_path)
            os.remove(dsstore_path)


if __name__ == '__main__':
    remove_pycache('../tabularmagic')
    remove_dsstore('../tabularmagic')




