from setuptools import setup, find_packages
import codecs
import os.path
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='fedl',
    version=get_version("src/fedl/__init__.py"),
    author='Adam Carpenter',
    author_email="adamc@jlab.org",
    description="Code for loading field emission related data, training models, and displaying results.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=[
        'numpy>=1.21.5',
        'pandas>=1.3.5',
        'seaborn>=0.11.2',
        'scikit-learn>=1.0.2',
        'torch>=1.11.0+cu113',
    ]
)
