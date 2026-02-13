"""
Setup para seo_analyzer - Paquete instalable de análisis SEO.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="seo_analyzer",
    version="1.0.0",
    author="SEO Technical Team",
    author_email="seo@example.com",
    description="Módulo de análisis SEO con Polars para datos de Google Search Console",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ferfersm/seoanalizer",
    packages=['seo_analyzer'] + find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
    ],
    python_requires=">=3.11",
    install_requires=[
        "polars>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
        ],
    },
    keywords="seo google-search-console analytics polars data-analysis",
    project_urls={
        "Bug Reports": "https://github.com/ferfersm/seoanalizer/issues",
        "Source": "https://github.com/ferfersm/seoanalizer",
    },
)
