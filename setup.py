from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='tempEgo3D',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23.1',
        'scipy>=1.10.1',

    ],
    entry_points={
        'console_scripts': [],
    },
    url='https://github.com/samuelLovett/tempEgo3D.git',
    author='Samuel Lovett',
    author_email='samuellovett@cmail.carleton.ca',
    description="A package for estimating 3D ego-motion velocity using millimetre-wave (mmWave) radar",
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)