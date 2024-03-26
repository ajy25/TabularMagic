from setuptools import setup, find_packages

def parse_requirements(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if not line.startswith('#')]
    
def read_license(file_path):
    with open(file_path, 'r') as file:
        file_contents = file.read()
        return file_contents

setup(
    name='tabularmagic',
    version='0.0.1',
    packages=find_packages(),
    author='Andrew Jianhua Yang',
    license=read_license('LICENSE'),
    install_requires=parse_requirements('requirements.txt')
)







