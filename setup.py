# -*- coding: utf-8 -*-
"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call
import shlex
import sys
from warnings import warn

py_version = sys.version_info
if py_version.major < 3:
    raise RuntimeError("reVX is not compatible with python 2!")
elif py_version.minor < 6:
    warn("You will the get best results by running reVX with python >= 3.6")

try:
    from pypandoc import convert_text
except ImportError:
    convert_text = lambda string, *args, **kwargs: string

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", encoding="utf-8") as readme_file:
    readme = convert_text(readme_file.read(), "rst", format="md")

with open(os.path.join(here, "NaTGenPD", "version.py"), encoding="utf-8") as f:
    version = f.read()

version = version.split('=')[-1].strip().strip('"').strip("'")


class PostDevelopCommand(develop):
    """
    Class to run post setup commands
    """

    def run(self):
        """
        Run method that tries to install pre-commit hooks
        """
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as e:
            warn("Unable to run 'pre-commit install': {}"
                 .format(e))

        develop.run(self)


test_requires = ["pytest", ]

numpy_dependency = "numpy>=1.15.0"
pandas_dependency = "pandas>=0.23.0"
click_dependency = "click>=7.0"
scipy_dependency = "scipy>=1.2.1"

setup(
    name="NaTGenPD",
    version=version,
    description="National Thermal Generation Performace Database",
    long_description=readme,
    author="Michael Rossol",
    author_email="michael.rossol@nrel.gov",
    url="https://github.com/NREL/NaTGenPD",
    packages=find_packages(),
    package_dir={"NaTGenPD": "NaTGenPD"},
    entry_points={
        "console_scripts": ["NaTGenPD=NaTGenPD.cli:main", ],
    },
    include_package_data=True,
    license="BSD license",
    zip_safe=False,
    keywords="NaTGenPD",
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Modelers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    test_suite="tests",
    install_requires=["click", "h5py", "numpy", "pandas", "scipy",
                      "scikit-learn"],
    extras_require={
        "test": test_requires,
        "dev": test_requires + ["pypandoc", "flake8", "pre-commit", "pylint"],
    },
    cmdclass={"develop": PostDevelopCommand},
)
