from setuptools import setup, find_packages

setup(
    name="merkabah",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "qiskit",
        "numpy",
        "requests",
        "fastapi",
        "uvicorn",
        "cryptography"
    ],
)
