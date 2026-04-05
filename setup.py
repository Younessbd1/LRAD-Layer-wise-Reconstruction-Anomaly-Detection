from setuptools import setup, find_packages

setup(
    name="lrad",
    version="0.1.0",
    description="Layer-wise Reconstruction Anomaly Detection",
    author="Youness",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "scikit-learn>=1.2",
        "pyyaml>=6.0",
    ],
)
