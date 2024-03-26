import subprocess
import os
import pathlib
directory_path = str(pathlib.Path('__notebook__').parent.resolve())
import shutil


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f'Folder "{folder_path}" deleted successfully.')
        except OSError as e:
            print(f'Error: {e}')
    else:
        print(f'The folder "{folder_path}" does not exist.')


if __name__ == '__main__':

    try:
        subprocess.check_call(['pip', 'uninstall', 'tabularmagic'])
        print('Successfully uninstalled package "tabularmagic"')
    except subprocess.CalledProcessError as e:
        print(f'Failed to uninstall package "tabularmagic": {e}')

    delete_folder(f'{directory_path}/build')
    delete_folder(f'{directory_path}/dist')
    delete_folder(f'{directory_path}/tabularmagic.egg-info')



