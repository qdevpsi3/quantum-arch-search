from setuptools import find_packages, setup

setup(name='qas_gym',
      version='0.0.1',
      install_requires=['gym', 'cirq', 'numpy'],
      packages=find_packages('src'),
      package_dir={'': 'src'})
