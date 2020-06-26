from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ntupledicts",
    version="0.9.3",
    author="Casey Pancoast",
    author_email="cqpancoast@gmail.com",
    description="Treating CMS TrackTrigger ROOT Ntuples as Python dictionaries with ML studies in mind.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cqpancoast/ntupledicts",
    license="MIT",
    keywords="cms tracktrigger track-trigger root ntuple python dictionary dict ml",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "tensorflow>=2",
        "scikit-learn>=0.22",
        "uproot>=3",
        "matplotlib>=3",
        "numpy>=1"
        ],
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics"
    ]
)
