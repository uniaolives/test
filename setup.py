from setuptools import setup, find_packages

setup(
    name="arkhe-qutip",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "qutip>=5.0.0",
        "qutip-qip>=0.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "grpcio>=1.60.0",
        "grpcio-tools>=1.60.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.5.0", "networkx>=2.8.0"],
        "all": ["matplotlib>=3.5.0", "networkx>=2.8.0"],
    },
    author="Arkhe(N) Team",
    description="Quantum Hypergraph Toolbox with Coherence Tracking",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/arkhe-chain/arkhe-qutip",
)
