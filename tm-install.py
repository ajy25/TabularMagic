import subprocess
import pathlib
directory_path = str(pathlib.Path('__notebook__').parent.resolve())

try:
    subprocess.check_call(['python', f'{directory_path}/setup.py', 'sdist'])
    subprocess.check_call(['pip', 'install', f'{directory_path}'])
    print('Successfully installed tabularmagic')
except subprocess.CalledProcessError as e:
    print(f'Failed to install tabularmagic: {e}')

