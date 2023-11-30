# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="amm-calc",
    version="1.0.0",
    description="amm calc library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="danielmalinovsky.github.io",
    author="Daniel Malinovsky",
    author_email="daniel.m.malinovsky@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(include=["automated_marketmaker_calc"]),
    include_package_data=True,
    install_requires=["numpy", "pandas", "math", "matplotlib.pyplot", "statistics", "IPython.display", "time", "datetime", "mpl_toolkits"]
)