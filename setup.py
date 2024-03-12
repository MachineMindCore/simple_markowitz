from setuptools import setup, find_packages

setup(
    name='simple_markowitz',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'yfinance',
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'openpyxl'
    ],
    author='MachineMindCore',
    author_email='machine.mind.core@gmail.com',
    description='A package for analyzing stock portfolios',
    url='https://github.com/MachineMindCore/simple_markowitz',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)