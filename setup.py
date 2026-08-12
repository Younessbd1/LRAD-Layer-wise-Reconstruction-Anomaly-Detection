from setuptools import find_packages, setup

setup(
    name="lrad",
    version="0.3.0",
    description=(
        "Layer-wise reconstruction anomaly detection: deep-ensemble "
        "bias/variance decomposition for OOD detection on CelebA and "
        "cold-start industrial anomaly detection on MVTec AD."
    ),
    author="BAHADDOU Youness",
    packages=find_packages(exclude=("tests", "scripts", "configs")),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "scikit-learn>=1.2",
        "scipy>=1.10",
        "pyyaml>=6.0",
        "Pillow>=9.0",
    ],
)
