import setuptools

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setuptools.setup(
    name="capability",
    version="0.0.1rc1",
    author="Alexandre Baharov",
    author_email="alex.baharov@gmail.com",
    description="A collecton of tools for capability analysis for Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alebaho/quality-control",
    keywords=[
        "capability",
        "Cpk",
        "Cp",
        "Ppk",
        "Pp",
        "dpm",
        "yield",
        "SigmaLevel",
        ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "scipy>=1.5.2",
        "numpy>=1.19.2",
        "matplotlib>=3.3.2",
        "pandas>=1.1.2",
        "seabonr>=0.11.1",
    ],
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "*.notebooks", "*.notebooks.*"],
    python_requires='>=3.6',
)
