# Description: Setup file for StrongFieldPhysics package

# Install by:
    # For production:
    #  python -m pip install StrongFieldPhysics/
    # For development (interactive installation mode):
    #  python -m pip install -e .

# Version
VERSION = '0.12.0' # MAJOR.MINOR.PATCH

# Version History:
    # 0.1.0: Initial version (2023-09-27)

# Developed by:  Aziz Alqasem (aziz_alqasem@hotmail.com)


from setuptools import setup, find_packages

MODULE_NAME = 'StrongFieldPhysics'

setup(
    name=MODULE_NAME,
    version=VERSION,
    packages=find_packages(include=['StrongFieldPhysics', 'StrongFieldPhysics.*']),
    install_requires=[
        'numpy',
        'numba',
        'scipy',
        'matplotlib',
    ]
)