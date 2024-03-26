import pathlib
directory_path = str(pathlib.Path('__notebook__').parent.resolve())
from setuptools import setup, find_packages


def parse_requirements(file_path):
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if not line.startswith('#')]
    except:
        return []
    

setup(
    name='tabularmagic',
    version='0.0.0',
    packages=find_packages(),
    author='Andrew Jianhua Yang',
    license=pathlib.Path('LICENSE').read_text(),
    description='''TabularMagic is a wrapper of scikit-learn and statsmodels 
        algorithms for rapid exploratory statistical and machine 
        learning modeling of tabular data.''',
    long_description=pathlib.Path('README.md').read_text(),
    install_requires=parse_requirements('requirements.txt'),
    python_requires='>=3.10'
)





