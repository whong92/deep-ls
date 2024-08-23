from setuptools import setup, Extension
import numpy as np

module = Extension('vrpstate', sources=['deepls/vrpstate.cpp'], include_dirs=[np.get_include()])


setup(
    name='deep-ls-tsp',
    version='0.0.1',
    packages=['tests', 'deepls'],
    url='',
    license='',
    author='Wai Hong Ong',
    author_email='samuelong168@gmail.com',
    description='',
    ext_modules=[module]
)
