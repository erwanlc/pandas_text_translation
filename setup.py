import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="text_translation",  # Replace with your own username
    version="0.1.0",
    author="Erwan Le Covec",
    author_email="erwan.lecovec@gmail.com",
    description="Package for text translation using transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://dev.azure.com/konecranes/kc-datascience/_git/text_translation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)