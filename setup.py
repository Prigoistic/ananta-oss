"""
Ananta: Scientific LLM Fine-tuning Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ananta",
    version="1.0.0",
    author="Ananta Team",
    description="Scientific LLM fine-tuning pipeline for mathematical reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Prigoistic/ananta-oss",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ananta-train=src.training.train_ananta:main",
            "ananta-evaluate=src.evaluation.evaluate_model:main",
            "ananta-demo=demos.app:main",
            "ananta-pipeline=src.run_pipeline:main",
        ],
    },
)
