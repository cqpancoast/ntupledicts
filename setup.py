from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ntupledicts",
    version="1.0.0",
    author="Casey Pancoast",
    description="Treating CMS TrackTrigger ROOT Ntuples as Python dictionaries with ML studies in mind.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cqpancoast/ntupledicts",
    license="MIT",
    keywords="cms tracktrigger track-trigger root ntuple python dictionary dict ml",
    packages=find_packages(),
    install_requires=["tensorflow>=2.0.0"]
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    entry_points={}
)
