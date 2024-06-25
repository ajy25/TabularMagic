import pathlib
from setuptools import setup, find_packages
directory_path = str(pathlib.Path('__notebook__').parent.resolve())


def parse_requirements(file_path):
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if not line.startswith('#')]
    except Exception as e:
        print(f'Error occured: {e}')
        return []
    

setup(
    name='tabularmagic',
    version='0.0.0',
    packages=find_packages(where=directory_path),
    author='Andrew Yang',
    license=pathlib.Path('LICENSE').read_text(),
    description='''TabularMagic is a wrapper of scikit-learn and statsmodels 
        algorithms for rapid exploratory statistical and machine 
        learning modeling of tabular data.''',
    long_description=pathlib.Path('README.md').read_text(),
    install_requires=parse_requirements('requirements.txt'),
    python_requires='>=3.11.5'
)





