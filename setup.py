from setuptools import setup, find_packages

setup(
    name="reservoir_computing",
    version="0.35",
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>1.19.5',
        'scikit_learn>=1.4',
        'scipy',
        'requests'
    ],
    author="Filippo Maria Bianchi",
    author_email="filippombianchi@gmail.com",
    description="Library for implementing reservoir computing models (Echo State Networks) for multivariate time series classification, clustering, and forecasting.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)