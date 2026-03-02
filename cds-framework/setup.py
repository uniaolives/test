from setuptools import setup, find_packages

setup(
    name="cds-framework",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
    ],
    author="Architect-Omega",
    description="Anisotropic Control Dynamical System (ACDS) Framework",
    python_requires=">=3.8",
)
