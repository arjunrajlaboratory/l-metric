from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="l_metric_calculator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for computing and visualizing L-metrics for gene expression data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/l-metric",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'tqdm',
        'plotly'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
