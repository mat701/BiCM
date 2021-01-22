import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bicm",
    version="0.9",
    author="Matteo Bruno",
    author_email="matteo.bruno@imtlucca.it",
    description="Package for bipartite configuration model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mat701/BiCM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)
