"""
Setup script for LUMINARK AI Framework
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="luminark",
    version="0.1.0",
    author="LUMINARK Team",
    author_email="",
    description="A quantum-enhanced AI/ML framework with self-awareness and meta-learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/foreverforward760-crypto/LUMINARK",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "pillow>=10.0.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "quantum": [
            "qiskit>=0.45.0",
            "qiskit-aer>=0.13.0",
        ],
        "advanced": [
            "networkx>=3.1",
        ],
        "all": [
            "qiskit>=0.45.0",
            "qiskit-aer>=0.13.0",
            "networkx>=3.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "luminark-train=examples.train_mnist:main",
            "luminark-advanced=examples.train_advanced_ai:main",
        ],
    },
)
