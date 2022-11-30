import sys

from setuptools import setup, find_packages

MIN_PYTHON_VERSION = (3, 9)

if sys.version_info[:2] < MIN_PYTHON_VERSION:
    raise RuntimeError('Python version required = {}.{}'.format(MIN_PYTHON_VERSION[0], MIN_PYTHON_VERSION[1]))

import ocean

REQUIRED_PACKAGES = [

]

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Natural Language :: Russian
Natural Language :: English
Intended Audience :: Developers
Intended Audience :: Education
Intended Audience :: End Users/Desktop
Intended Audience :: Science/Research
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3 :: Only
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Image Processing
Topic :: Scientific/Engineering :: Image Recognition
Topic :: Software Development
Topic :: Software Development :: Libraries
Topic :: Software Development :: Libraries :: Python Modules
Topic :: Software Development :: Documentation
Topic :: Software Development :: Documentation :: Sphinx
Topic :: Software Development :: Sound/Audio
Topic :: Software Development :: Sound/Audio :: Analysis
Topic :: Software Development :: Sound/Audio :: Speech
Topic :: Software Development :: Libraries
Topic :: Software Development :: Python Modules
Topic :: Software Development :: Localization
Topic :: Software Development :: Utilities
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Framework :: Jupyter
Framework :: Jupyter :: JupyterLab :: 4
Framework :: Sphinx
"""

with open('README.md', 'r') as fh:
    long_description = fh.read()

    setup(
        name = ocean.__name__,
        packages = find_packages(),
        license = ocean.__license__,
        version = ocean.__release__,
        author = ocean.__author__en__,
        author_email = ocean.__email__,
        maintainer = ocean.__maintainer__en__,
        maintainer_email = ocean.__maintainer_email__,
        url = ocean.__uri__,
        description = ocean.__summary__,
        long_description = long_description,
        long_description_content_type = 'text/markdown',
        install_requires=REQUIRED_PACKAGES,
        keywords = ['big5', 'MachineLearning', 'Statistics', 'ComputerVision', 'ArtificialIntelligence',
                    'Preprocessing'],
        include_package_data = True,
        classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
        python_requires = '>=3.9, <4',
        entry_points = {
            'console_scripts': [],
        },
        project_urls = {
            'Bug Reports': 'https://github.com/DmitryRyumin/ocean/issues',
            'Documentation': 'https://ocean.readthedocs.io',
            'Source Code': 'https://github.com/DmitryRyumin/ocean/tree/main/ocean',
            'Download': 'https://github.com/DmitryRyumin/ocean/tags',
        },
    )
