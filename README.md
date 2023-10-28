# [OCEANAI](https://oceanai.readthedocs.io/en/latest/)

[![SAI](./docs/source/_static/badges/SAI_badge_flat.svg)](https://sai.itmo.ru/)
[![ITMO](./docs/source/_static/badges/ITMO_badge_flat.svg)](https://en.itmo.ru/en/)

![PyPI](https://img.shields.io/pypi/v/oceanai)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oceanai)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/oceanai)
![GitHub repo size](https://img.shields.io/github/repo-size/dmitryryumin/oceanai)
![PyPI - Status](https://img.shields.io/pypi/status/oceanai)
![PyPI - License](https://img.shields.io/pypi/l/oceanai)
![GitHub top language](https://img.shields.io/github/languages/top/dmitryryumin/oceanai)
![Documentation Status](https://readthedocs.org/projects/oceanai/badge/?version=latest)

---

| [Documentation in Russian](https://oceanai.readthedocs.io/ru/latest/index.html) |
|---------------------------------------------------------------------------------|

---

<h4 align="center"><span style="color:#EC256F;">Description</span></h4>

---

> **[OCEANAI](https://oceanai.readthedocs.io/en/latest/)** is an open-source library consisting of a set of algorithms for intellectual analysis of human behavior based on multimodal data for automatic personal traits assessment to performance of professional duties. The library evaluates 5 traits: **O**penness to experience, **C**onscientiousness, **E**xtraversion, **A**greeableness, **N**euroticism.

---

**[OCEANAI](https://oceanai.readthedocs.io/en/latest/)** includes three main algorithms:

1. Audio Information Analysis Algorithm (AIA).
2. Video Information Analysis Algorithm (VIA).
3. Multimodal Fusion Algorithm (MF).

The AIA and VIA algorithms implement the functions of strong artificial intelligence (AI) in terms of complexing acoustic and visual features built on different principles (expert and neural network), i.e. these algorithms implement the approaches of composite (hybrid) AI. The necessary pre-processing is carried out in the algorithms audio and video information, the calculation of visual and acoustic features and the issuance of prediction personality traits based on them.

The MF algorithm is a link between two information analysis algorithms (AIA and VIA). This algorithm performs a weighted neural network combination of prediction personality traits obtained using the AIA and VIA algorithms.

In addition to the main task - unimodal and multimodal personality traits assessment, the features implemented in **[OCEANAI](https://oceanai.readthedocs.io/en/latest/)** will allow researchers to solve other problems of analyzing human behavior, for example, recognizing his affective states.

To install the library, you should refer to the **[Installation and Update](https://oceanai.readthedocs.io/en/latest/user_guide/installation.html#id2)**.

To work with audio information, you should refer to the **[Audio information processing](https://oceanai.readthedocs.io/en/latest/user_guide/samples/audio.html)**.

To work with video information, you should refer to the **[Video information processing](https://oceanai.readthedocs.io/en/latest/user_guide/samples/video.html)**.

To work with audio-visual information, you should refer to the **[Multimodal information processing](https://oceanai.readthedocs.io/en/latest/user_guide/samples/multimodal.html)**.

The library solves practical tasks:

1. **[Ranking of potential candidates by professional responsibilities](https://oceanai.readthedocs.io/en/latest/user_guide/notebooks/Pipeline_practical_task_1.html)**.
2. **[Predicting consumer preferences for industrial goods](https://oceanai.readthedocs.io/en/latest/user_guide/notebooks/Pipeline_practical_task_2.html)**.

**[OCEANAI](https://oceanai.readthedocs.io/en/latest/)** uses the latest open-source libraries for audio and video processing: **[librosa](https://librosa.org/)**, **[openSMILE](https://audeering.github.io/opensmile-python/)**, **[openCV](https://pypi.org/project/opencv-python/)**, **[mediapipe](https://google.github.io/mediapipe/getting_started/python)**.

**[OCEANAI](https://oceanai.readthedocs.io/en/latest/)** is written in the **[python programming language](https://www.python.org/)**. Neural network models are implemented and trained using an open-source library code **[TensorFlow](https://www.tensorflow.org/)**.

---

## Research data

The **[OCEANAI](https://oceanai.readthedocs.io/en/latest/)** library was tested on two corpora:

1) The publicly available and large-scale **[First Empressions V2 corpus](https://chalearnlap.cvc.uab.cat/dataset/24/description/)**.
2) On the first publicly available Russian-language **[Multimodal Personality Traits Assessment (MuPTA) corpus](https://hci.nw.ru/en/pages/mupta-corpus)**.

---

| [Development team](https://oceanai.readthedocs.io/en/latest/about.html) |
|-------------------------------------------------------------------------|

---

## Certificate of state registration of a computer program

**[Library of algorithms for intelligent analysis of human behavior based on multimodal data, providing human's personality traits assessment to perform professional duties (OCEAN-AI)](https://new.fips.ru/registers-doc-view/fips_servlet?DB=EVM&DocNumber=2023613724&TypeFile=html)**

---

## Supported by

The study is supported by the [Research Center Strong Artificial Intelligence in Industry](https://sai.itmo.ru/)
of [ITMO University](https://en.itmo.ru/) as part of the plan of the center's program: Development and testing of an experimental prototype of a library of strong AI algorithms in terms of hybrid decision making based on the interaction of AI and decision maker based on models of professional behavior and cognitive processes of decision maker in poorly formalized tasks with high uncertainty.
