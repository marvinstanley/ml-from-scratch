from setuptools import setup, find_packages

setup(name = 'ml_scratch',
      version = '0.0.1',
      description = "Basic Machine Learning module for personal learning purpose",
      url = "https://github.com/marvinstanley/ml-from-scratch",
      author = "Stanley Marvin",
      author_email = 'stanleymarvin1999@yahoo.com',
      keywords='ml machine learning machinelearning basic',
      packages = find_packages(),
      package_data={'ml_scratch':['*']},
      zip_safe = False,
      install_requires = [
              'numpy'],
       )
