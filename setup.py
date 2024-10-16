import io
import os
import re
import setuptools

# Hardcoding version instead of dynamically fetching from __init__.py
__version__ = '1.0.0'  # Update the version number if needed

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

# Reading the long description from README.md
long_description = read('README.md')

setuptools.setup(
    name='MolALKit_1_0',  # Updated package name
    version=__version__,
    python_requires='>=3.8',
    install_requires=[
        'typed-argument-parser',
        'rdkit',
        'mgktools'
    ],
    entry_points={
        'console_scripts': [
            'molalkit_run=molalkit.al.run:molalkit_run',
            'molalkit_run_from_cpt=molalkit.al.run:molalkit_run_from_cpt',
        ]
    },
    author='Yan Xiang',
    author_email='yan.xiang@duke.edu',
    description='MolALKit: A Toolkit for Active Learning in Molecular Data.',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Added for markdown support
    url='https://github.com/hrshitagowda/MolALKit_1_0',  # Updated URL to your fork
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
    package_data={'': ['data/datasets/*.csv', 'models/configs/*Config']}
)
