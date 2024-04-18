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

if __name__ == '__main__':
    remove_pycache('../tabularmagic')
    
