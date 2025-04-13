from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='genrobo3d',
    version='0.1.1',
    description='Generalizable robotic manipulation with VLLMs',
    author='Shizhe Chen',
    author_email='cshizhe@gmail.com',
    packages=find_packages(),
)