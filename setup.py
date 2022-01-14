import setuptools
from os import path, pardir
from distutils.command.build_ext import build_ext as DistUtilsBuildExt
from setuptools import setup, find_packages

VERSION = '0.1.0'

PKG_FOLDER = path.abspath(path.join(__file__, pardir))

with open(path.join(PKG_FOLDER, 'requirements.txt')) as req_file:
    requirements = req_file.read().splitlines()

# set a long description which is basically the README
with open(path.join(PKG_FOLDER, 'README.md')) as f:
    long_description = f.read()


setup(
    name='creed-helper',
    version=VERSION,
    packages=find_packages(),
    license='Modified Apache License 2.0',
    description='Scripts for running QC analysis on the MibiScope CAC',
    author='Angelo Lab',
    url='https://github.com/angelolab/creed-helper',
    download_url='https://github.com/angelolab/creed-helper/archive/v{}.tar.gz'.format(VERSION),
    install_requires=requirements,
    extras_require={
        'tests': ['pytest',
                  'pytest-cov',
                  'pytest-pycodestyle',
                  'testbook']
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=['License :: OSI Approved :: Apache Software License',
                 'Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.8']
)
