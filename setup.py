from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neural_network_system",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive neural network training system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural_network_system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
        ],
        "distributed": [
            "horovod>=0.28.0",
            "torch-tb-profiler>=0.4.0",
        ],
        "visualization": [
            "wandb>=0.15.0",
            "graphviz>=0.20.0",
            "plotly>=5.13.0",
        ],
        "profiling": [
            "py-spy>=0.3.0",
            "memory-profiler>=0.60.0",
            "psutil>=5.9.0",
        ],
    },
) 