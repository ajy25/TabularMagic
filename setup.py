from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if not line.startswith('#')]

setup(
    name='tabularmagic',
    version='0.0.1',
    packages=find_packages(),
    author='Andrew Jianhua Yang',
    install_requires=parse_requirements('requirements.py')
)





