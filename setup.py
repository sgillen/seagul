from setuptools import setup, find_packages
from Cython.Build import cythonize


setup(name="seagul", version="0.0.1", packages=[package for package in find_packages()])

setup(
    ext_modules = cythonize(["seagul/integrationx.pyx", "seagul/notebooks/scratch/pyx_profile.pyx", "seagul/envs/simple_nonlinear/linear_zx.pyx"])
)
