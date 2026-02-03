#!/usr/bin/env python3
"""Setup script for PAPO package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies
core_requirements = [
    # Core build dependencies
    "setuptools>=61.0",
    "wheel", 
    "ninja",
    "packaging",
    
    # Core ML/AI packages
    "vllm==0.8.4",
    "accelerate",
    "datasets", 
    "numpy",
    "pandas",
    "peft",
    "pillow",
    "pyarrow>=15.0.0",
    "transformers==4.51.3",
    
    # Configuration and utilities
    "omegaconf",
    "codetiming", 
    "wandb",
    "tensorboard",
    
    # Math and data processing
    "mathruler",
    "pylatexenc",
    "qwen-vl-utils",
    "tensordict",
    "torchdata",
    
    # PyTorch (will be installed via specific index in install.sh)
    "torch==2.6.0",
    "torchvision", 
    "torchaudio",
]

setup(
    name="PAPO",
    version="0.1.0",
    author="PAPO Team",
    author_email="team@papo.ai",
    description="Perception-Aware Policy Optimization for Multimodal Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MikeWangWZHL/PAPO",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "ruff>=0.1.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.18",
        ],
        # Attention optimization dependencies
        "cuda-optimized": [
            "flash-attn==2.7.4.post1",
            "liger-kernel",
            "flashinfer-python",
        ],
        # All dependencies combined
        "all": [
            "flash-attn==2.7.4.post1",
            "liger-kernel",
            "flashinfer-python",
        ],
    },
    entry_points={
        "console_scripts": [
            "papo=papo.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
