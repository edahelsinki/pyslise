import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SLISE",
    version="0.0.1",
    author="Aggrathon",
    author_email="antonbjo@gmail.com",
    description="The SLISE algorithm for robust regression and explanations for black box models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aggrathon/slise-py",
    packages=setuptools.find_packages(exclude=("experiments", "tests")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
