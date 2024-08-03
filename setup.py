import sys

from setuptools import setup, find_packages

MIN_PYTHON_VERSION = (3, 9)

if sys.version_info[:2] < MIN_PYTHON_VERSION:
    raise RuntimeError("Python version required = {}.{}".format(MIN_PYTHON_VERSION[0], MIN_PYTHON_VERSION[1]))

import oceanai

REQUIRED_PACKAGES = [
    "ipython >= 8.18.1",
    "jupyterlab == 3.5.0",
    "tensorflow >= 2.15.0",
    "keras >= 2.11.0",
    "Keras-Applications>=1.0.8",
    "numpy >= 1.23.5",
    "scipy >= 1.9.3",
    "pandas >= 1.5.2",
    "requests >= 2.28.1",
    "opensmile >= 2.4.1",
    "librosa >= 0.9.2",
    "audioread >= 3.0.0",
    "scikit-learn >= 1.1.3",
    "opencv-contrib-python >= 4.6.0.66",
    "mediapipe >= 0.9.0",
    "liwc >= 0.5.0",
    "transformers >= 4.36.0",
    "sentencepiece >= 0.1.99",
    "torch == 2.0.1",
    "torchaudio == 2.0.2",
    "sacremoses >= 0.0.1",
    "gradio == 4.40.0",
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
Topic :: Documentation
Topic :: Documentation :: Sphinx
Topic :: Multimedia :: Sound/Audio
Topic :: Multimedia :: Sound/Audio :: Analysis
Topic :: Multimedia :: Sound/Audio :: Speech
Topic :: Software Development :: Libraries
Topic :: Software Development :: Libraries :: Python Modules
Topic :: Software Development :: Localization
Topic :: Utilities
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Framework :: Jupyter
Framework :: Jupyter :: JupyterLab :: 4
Framework :: Sphinx
"""

with open("README.md", "r") as fh:
    long_description = fh.read()

    setup(
        name=oceanai.__name__,
        packages=find_packages(),
        license=oceanai.__license__,
        version=oceanai.__release__,
        author=oceanai.__author__en__,
        author_email=oceanai.__email__,
        maintainer=oceanai.__maintainer__en__,
        maintainer_email=oceanai.__maintainer_email__,
        url=oceanai.__uri__,
        description=oceanai.__summary__,
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=REQUIRED_PACKAGES,
        keywords=[
            "OCEAN-AI",
            "MachineLearning",
            "Statistics",
            "ComputerVision",
            "ArtificialIntelligence",
            "Preprocessing",
        ],
        include_package_data=True,
        classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
        python_requires=">=3.9, <4",
        entry_points={
            "console_scripts": [],
        },
        project_urls={
            "Bug Reports": "https://github.com/DmitryRyumin/oceanai/issues",
            "Documentation": "https://oceanai.readthedocs.io",
            "Source Code": "https://github.com/DmitryRyumin/oceanai/tree/main/oceanai",
            "Download": "https://github.com/DmitryRyumin/oceanai/tags",
        },
    )
