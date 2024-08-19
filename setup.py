from setuptools import find_packages
from distutils.core import setup

setup(
    name='aerial_gym',
    version='1.0.0',
    author='Mihir Kulkarni',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='mihir.kulkarni@ntnu.no',
    description='Isaac Gym environments for Aerial Robots',
    install_requires=['isaacgym',
                      'matplotlib',
                      'numpy',
                      'torch',
                      'pytorch3d']
)

setup(
    name='pysot',
    version='1.0.0',
    author='ZhiyuanChen',
    license="Apache License Version 2.0",
    packages=find_packages(),
    author_email='dont know',
    description='SenseTime Research platform for single object tracking, implementing algorithms like SiamRPN and SiamMask. ',
    install_requires=['numpy',
                      'torch']
)