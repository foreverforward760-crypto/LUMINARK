"""
Setup script for Mycelial Defense System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="mycelial-defense",
    version="0.1.0",
    author="Richard Leroy Stanfield Jr.",
    author_email="your.email@example.com",  # Update with actual email
    description="Bio-Inspired Active Defense for AI Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/foreverforward760-crypto/LUMINARK",
    packages=find_packages(exclude=["tests*", "examples*", "dashboard*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dashboard": [
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
            "websockets>=12.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mycelial=cli.commands:cli",
        ],
    },
    keywords=[
        "ai-safety",
        "defense",
        "security",
        "alignment",
        "bio-inspired",
        "mycelium",
        "camouflage",
    ],
    project_urls={
        "Bug Reports": "https://github.com/foreverforward760-crypto/LUMINARK/issues",
        "Source": "https://github.com/foreverforward760-crypto/LUMINARK",
    },
)
