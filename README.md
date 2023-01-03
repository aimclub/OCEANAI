# [OCEANAI](https://github.com/DmitryRyumin/ocean)

![PyPI](https://img.shields.io/pypi/v/oceanai)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oceanai)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/oceanai)
![GitHub repo size](https://img.shields.io/github/repo-size/dmitryryumin/oceanai)
![PyPI - Status](https://img.shields.io/pypi/status/oceanai)
![PyPI - License](https://img.shields.io/pypi/l/oceanai)
![GitHub top language](https://img.shields.io/github/languages/top/dmitryryumin/oceanai)
![Documentation Status](https://readthedocs.org/projects/oceanai/badge/?version=latest)

---

| [Documentation in Russian](https://github.com/DmitryRyumin/oceanai/blob/main/README_RU.md) |
|--------------------------------------------------------------------------------------------|

---

<h4 align="center"><span style="color:#EC256F;">Description</span></h4>

---

>  **[OCEANAI](https://github.com/DmitryRyumin/oceanai)** is an open-source library consisting of a set of algorithms for intellectual analysis of human behavior based on multimodal data for automatic personal traits assessment to performance of professional duties. The library evaluates 5 traits: **O**penness to experience, **C**onscientiousness, **E**xtraversion, **A**greeableness, **N**euroticism.

---

**[OCEANAI](https://github.com/DmitryRyumin/oceanai)** includes three main algorithms:

1. Audio Information Analysis Algorithm (AIA).
2. Video Information Analysis Algorithm (VIA).
3. Multimodal Fusion Algorithm (MF).

The AIA and VIA algorithms implement the functions of strong artificial intelligence (AI) in terms of complexing acoustic and visual features built on different principles (expert and neural network), i.e. these algorithms implement the approaches of composite (hybrid) AI. The necessary pre-processing is carried out in the algorithms audio and video information, the calculation of visual and acoustic features and the issuance of prediction personality traits based on them.

The MF algorithm is a link between two information analysis algorithms (AIA and VIA). This algorithm performs a weighted neural network combination of prediction personality traits obtained using the AIA and VIA algorithms.

In addition to the main task - unimodal and multimodal personality traits assessment, the features implemented in **[OCEANAI](https://github.com/DmitryRyumin/oceanai)** will allow researchers to solve other problems of analyzing human behavior, for example, recognizing his affective states.

To install the library, you should refer to the **[Installation and Update](https://oceanai.readthedocs.io/en/latest/user_guide/installation.html#id2)**.

To work with audio information, you should refer to the **[Audio information processing](https://oceanai.readthedocs.io/en/latest/user_guide/samples/audio.html)**.

To work with video information, you should refer to the **[Video information processing](https://oceanai.readthedocs.io/en/latest/user_guide/samples/video.html)**.

To work with audio-visual information, you should refer to the **[Multimodal information processing](https://oceanai.readthedocs.io/en/latest/user_guide/samples/multimodal.html)**.

The library solves practical tasks:

1. **[Ranking of potential candidates by professional responsibilities](https://oceanai.readthedocs.io/en/latest/user_guide/notebooks/Pipeline_practical_task_1.html)**.
2. **[Predicting consumer preferences for industrial goods](https://oceanai.readthedocs.io/en/latest/user_guide/notebooks/Pipeline_practical_task_2.html)**.

**[OCEANAI](https://github.com/DmitryRyumin/oceanai)** uses the latest open-source libraries for audio and video processing: **[librosa](https://librosa.org/)**, **[openSMILE](https://audeering.github.io/opensmile-python/)**, **[openCV](https://pypi.org/project/opencv-python/)**, **[mediapipe](https://google.github.io/mediapipe/getting_started/python)**.

**[OCEANAI](https://github.com/DmitryRyumin/oceanai)** is written in the **[python programming language](https://www.python.org/)**. Neural network models are implemented and trained using an open-source library code **[TensorFlow](https://www.tensorflow.org/)**.
