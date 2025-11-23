"""
Setup script for HMTT.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hmtt",
    version="1.0.0",
    author="Ananta Team",
    description="Hybrid Math-Text Tokenizer: A discrete tokenization system for mixed NL/Math/Code content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "tokenizers>=0.13.0",
    ],
    extras_require={
        "full": [
            "regex>=2023.0.0",
            "tree-sitter>=0.20.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
