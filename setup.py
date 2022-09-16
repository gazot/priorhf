from os import path
from setuptools import setup, find_packages


setup(
    name="priorhf",
    version="0.1.0",
    packages=find_packages(),
    license="MIT",
    author="Christian Gajek",
    author_email="Christian.Gajek@tum.de",
    description="Wrapper for computing priors in pyhf for Bayesian inference",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pyhf",
        "jax",
    ],
)
