from setuptools import setup, find_packages

setup(
    name="merkabah-cy",
    version="0.1.0",
    package_dir={"": "src/python"},
    packages=find_packages(where="src/python"),
    install_requires=[
        "numpy",
        "torch",
        "qiskit",
        "qiskit-algorithms",
        "aiohttp",
        "aioredis",
        "cryptography",
        "pika",
        "requests",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "h5py",
        "pytorch-lightning",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
)
