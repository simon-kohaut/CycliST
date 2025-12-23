"""Sets up the CycliST package for installation."""

# Standard library
import re
import setuptools

# Find CycliST version and author strings
with open("cyclist/__init__.py", "r", encoding="utf8") as fd:
    content = fd.read()
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE).group(1)
    author = re.search(r'^__author__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE).group(1)

# Import readme
with open("README.md", "r", encoding="utf8") as readme:
    long_description = readme.read()

setuptools.setup(
    name="cyclist",
    version=version,
    author=author,
    author_email="simon.kohaut@cs.tu-darmstadt.de",
    description="A Python package to generate the CycliST dataset for video question answering.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    package_data={
        "cyclist": ["py.typed"],  # https://www.python.org/dev/peps/pep-0561/
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: BSD-3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        # => libraries for actual functionality
        #   -> general tools
        "dataclasses; python_version < '3.7'",
        "types-dataclasses; python_version < '3.7'",
        "typing-extensions; python_version < '3.8'",
        #   -> generic scientific
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "seaborn",
        "opencv-python",
        # => testing and code quality
        #   -> static code analysis
        "black",
        "ruff",
        #   -> dynamic code analysis
        "hypothesis",
        "pytest",
        "pytest-cov",
    ],
)
