import subprocess
import shutil
import pathlib
import os
import sys
directory_path = pathlib.Path(__file__).parent.resolve()



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

    arguments = sys.argv
    other_arguments = arguments[1:]
    if len(other_arguments) != 1:
        raise RuntimeError('Invalid command line argument. ' + \
            'Please call "python tmbuild.py install" to install, ' + \
            'or "python tmbuild.py uninstall" to uninstall.')

    if other_arguments[0] == 'install':
        try:
            subprocess.check_call(
                ['python', str(directory_path / 'setup.py'), 'sdist'])
            subprocess.check_call(['pip', 'install', f'{directory_path}'])
            print('Successfully installed tabularmagic')
        except subprocess.CalledProcessError as e:
            print(f'Failed to install tabularmagic: {e}')

    elif other_arguments[0] == 'uninstall':
        try:
            subprocess.check_call(['pip', 'uninstall', 'tabularmagic'])
            print('Successfully uninstalled package "tabularmagic"')
        except subprocess.CalledProcessError as e:
            print(f'Failed to uninstall package "tabularmagic": {e}')

        delete_folder(str(directory_path / 'build'))
        delete_folder(str(directory_path / 'dist'))
        delete_folder(str(directory_path / 'tabularmagic.egg-info'))

    else:
        raise RuntimeError('Invalid command line argument. ' + \
            'Please call "python tmbuild.py install" to install, ' + \
            'or "python tmbuild.py uninstall" to uninstall.')

