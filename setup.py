from setuptools import setup

setup(name='conformal',
      version='0.1',
      url='https://github.com/koulanurag/conformal',
      description='It is used for conformal prediction',
      author='Anurag Koul',
      author_email='koulanurag@gmail.com',
      license='MIT',
      packages=['conformal'],
      py_modules=['conformal'],
      install_requires=['numpy', 'matplotlib'],
      keywords=['deep-learning', 'conformal-prediction', 'machine-learning'],
      zip_safe=False)
