from setuptools import setup, find_packages

setup(
    name="cell_type_analyzer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'Pillow',
        'torch',
        'SimpleITK',
        'scikit-learn',
        'transformers',
        'umap-learn',
        'scipy',
        'seaborn',
        'openpyxl',
        'requests'
    ],
    python_requires='>=3.7',
)