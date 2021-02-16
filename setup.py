import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bicm",
    version="2.0",
    author="Matteo Bruno",
    author_email="matteo.bruno@imtlucca.it",
    description="Package for bipartite configuration model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mat701/BiCM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
                      "numpy>=1.14",
                      "scipy>=1.4",
                      "tqdm>=4.52.0"
                      ],
)
