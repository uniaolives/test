# setup.py

from setuptools import setup, find_packages

setup(
    name="arkhe_qutip",
    version="0.1.0",
    description="Arkhe(n) Quantum Hypergraph Library for QuTiP",
    author="Arkhe Protocol Team",
    packages=find_packages(),
    install_requires=[
        "qutip>=4.7",
        "numpy>=1.20",
        "scipy>=1.8",
        "matplotlib>=3.5",
        "networkx>=2.8",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
