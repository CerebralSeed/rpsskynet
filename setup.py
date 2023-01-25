from setuptools import setup, find_packages

exec(open('rpsskynet/version.py').read())

setup(
  name = 'rpsskynet',
  packages = find_packages(),
  version = __version__,
  license='MIT',
  description = 'Rock Paper Scissors Skynet - Pytorch',
  author = 'Jeremiah Johnson',
  author_email = 'j.johnson.bbt@gmail.com',
  url = 'https://github.com/CerebralSeed/rpsskynet',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'interactive models',
    'rock paper scissors',
    'pytorch'

  ],
  install_requires=[
    'torch',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
