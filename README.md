# [OCEAN-AI](https://oceanai.readthedocs.io/en/latest/)

<p align="center">
    <img src="https://raw.githubusercontent.com/aimclub/OCEANAI/main/docs/source/_static/logo.svg" alt="Logo" width="40%">
<p>

---

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
[![App](https://img.shields.io/badge/🤗-DEMO--OCEANAI-FFD21F.svg)]([https://huggingface.co/spaces/ElenaRyumina/AVCER](https://huggingface.co/spaces/ElenaRyumina/OCEANAI))

---

| [Documentation in Russian](https://oceanai.readthedocs.io/ru/latest/index.html) |
|---------------------------------------------------------------------------------|

---

<h4 align="center"><span style="color:#EC256F;">Description</span></h4>

---

> **[OCEAN-AI](https://oceanai.readthedocs.io/en/latest/)** is an open-source library consisting of a set of algorithms for intellectual analysis of human behavior based on multimodal data for automatic personality traits (PT) assessment. The library evaluates five PT: **O**penness to experience, **C**onscientiousness, **E**xtraversion, **A**greeableness, Non-**N**euroticism.

<p align="center">
    <img src="https://raw.githubusercontent.com/aimclub/OCEANAI/main/docs/source/_static/Pipeline_OCEANAI.en.svg" alt="Pipeline">
<p>

---

**[OCEAN-AI](https://oceanai.readthedocs.io/en/latest/)** includes three main algorithms:

1. Audio Information Analysis Algorithm (AIA).
2. Video Information Analysis Algorithm (VIA).
3. Text Information Analysis Algorithm (TIA).
4. Multimodal Information Fusion Algorithm (MIF).

The AIA, VIA and TIA algorithms implement the functions of strong artificial intelligence (AI) in terms of complexing acoustic, visual and linguistic features built on different principles (hand-crafted and deep features), i.e. these algorithms implement the approaches of composite (hybrid) AI. The necessary pre-processing of audio, video and text information, the calculation of visual, acoustic and linguistic features and the output of predictions of personality traits based on them are carried out in the algorithms.

The MIF algorithm is a combination of three information analysis algorithms (AIA, VIA and TIA). This algorithm performs feature-level fusion obtained by the AIA, VIA and TIA algorithms.

In addition to the main task - unimodal and multimodal personality traits assessment, the features implemented in **[OCEAN-AI](https://oceanai.readthedocs.io/en/latest/)** will allow researchers to solve other problems of analyzing human behavior, for example, affective state recognition.

To install the library, you should refer to the **[Installation and Update](https://oceanai.readthedocs.io/en/latest/user_guide/installation.html#id2)**.

To work with audio information, you should refer to the **[Audio information processing](https://oceanai.readthedocs.io/en/latest/user_guide/samples/audio.html)**.

To work with video information, you should refer to the **[Video information processing](https://oceanai.readthedocs.io/en/latest/user_guide/samples/video.html)**.

To work with text information, you should refer to the **[Text information processing](https://oceanai.readthedocs.io/en/latest/user_guide/samples/text.html)**.

To work with multimodal information, you should refer to the **[Multimodal information processing](https://oceanai.readthedocs.io/en/latest/user_guide/samples/multimodal.html)**.

The library solves practical tasks:

1. **[Ranking of potential candidates by professional responsibilities](https://oceanai.readthedocs.io/en/latest/user_guide/notebooks/Pipeline_practical_task_1.html)**.
2. **[Predicting consumer preferences for industrial goods](https://oceanai.readthedocs.io/en/latest/user_guide/notebooks/Pipeline_practical_task_2.html)**.
3. **[Forming effective work teams](https://oceanai.readthedocs.io/ru/latest/user_guide/notebooks/Pipeline_practical_task_3.html)**.

**[OCEAN-AI](https://oceanai.readthedocs.io/en/latest/)** uses the latest open-source libraries for audio, video and text processing: **[librosa](https://librosa.org/)**, **[openSMILE](https://audeering.github.io/opensmile-python/)**, **[openCV](https://pypi.org/project/opencv-python/)**, **[mediapipe](https://google.github.io/mediapipe/getting_started/python)**, **[transformers](https://pypi.org/project/transformers)**.

**[OCEAN-AI](https://oceanai.readthedocs.io/en/latest/)** is written in the **[python programming language](https://www.python.org/)**. Neural network models are implemented and trained using an open-source library code **[TensorFlow](https://www.tensorflow.org/)**.

---

## Research data

The **[OCEAN-AI](https://oceanai.readthedocs.io/en/latest/)** library was tested on two corpora:

1) The publicly available and large-scale **[First Impressions V2 corpus](https://chalearnlap.cvc.uab.cat/dataset/24/description/)**.
2) On the first publicly available Russian-language **[Multimodal Personality Traits Assessment (MuPTA) corpus](https://hci.nw.ru/en/pages/mupta-corpus)**.

---

| [Development team](https://oceanai.readthedocs.io/en/latest/about.html) |
|-------------------------------------------------------------------------|

---

## Certificate of state registration of a computer program

**[Library of algorithms for intelligent analysis of human behavior based on multimodal data, providing human's personality traits assessment to perform professional duties (OCEAN-AI)](https://new.fips.ru/registers-doc-view/fips_servlet?DB=EVM&DocNumber=2023613724&TypeFile=html)**

## Certificate of state registration of a database

**[MuPTA - Multimodal Personality Traits Assessment Corpus](https://new.fips.ru/registers-doc-view/fips_servlet?DB=DB&DocNumber=2023624011&TypeFile=html)**

---

## Publications

### Journals

```bibtex
@article{ryumina24_prl,
    author = {Ryumina, Elena and Markitantov, Maxim and Ryumin, Dmitry and Karpov, Alexey},
    title = {{Gated Siamese Fusion Network based on Multimodal Deep and Hand-Crafted Features for Personality Traits Assessment}},
    volume = {185},
    pages = {45--51},
    journal = {Pattern Recognition Letters},
    year = {2024},
    issn = {0167--8655},
    doi = {10.1016/j.patrec.2024.07.004},
    url = {https://www.sciencedirect.com/science/article/pii/S0167865524002071},
}
```

```bibtex
@article{ryumina24_eswa,
    author = {Elena Ryumina and Maxim Markitantov and Dmitry Ryumin and Alexey Karpov},
    title = {OCEAN-AI Framework with EmoFormer Cross-Hemiface Attention Approach for Personality Traits Assessment},
    journal = {Expert Systems with Applications},
    volume = {239},
    pages = {122441},
    year = {2024},
    doi = {https://doi.org/10.1016/j.eswa.2023.122441},
}
```

```bibtex
@article{ryumina22_neurocomputing,
    author = {Elena Ryumina and Denis Dresvyanskiy and Alexey Karpov},
    title = {In Search of a Robust Facial Expressions Recognition Model: A Large-Scale Visual Cross-Corpus Study},
    journal = {Neurocomputing},
    volume = {514},
    pages = {435--450},
    year = {2022},
    doi = {https://doi.org/10.1016/j.neucom.2022.10.013},
}
```

### Conferences

```bibtex
@inproceedings{ryumina24_interspeech,
    author = {Elena Ryumina and Dmitry Ryumin and and Alexey Karpov},
    title = {OCEAN-AI: Open Multimodal Framework for Personality Traits Assessment and HR-Processes Automatization},
    year = {2024},
    booktitle = {INTERSPEECH},
    pages = {in press},
    doi = {in press},
}
```

```bibtex
@inproceedings{ryumina23_interspeech,
    author = {Elena Ryumina and Dmitry Ryumin and Maxim Markitantov and Heysem Kaya and Alexey Karpov},
    title = {Multimodal Personality Traits Assessment (MuPTA) Corpus: The Impact of Spontaneous and Read Speech},
    year = {2023},
    booktitle = {INTERSPEECH},
    pages = {4049--4053},
    doi = {https://doi.org/10.21437/Interspeech.2023-1686},
}
```

---

## Supported by

The study is supported by the [Research Center Strong Artificial Intelligence in Industry](https://sai.itmo.ru/)
of [ITMO University](https://en.itmo.ru/) as part of the plan of the center's program: Development and testing of an experimental prototype of a library of strong AI algorithms in terms of hybrid decision making based on the interaction of AI and decision maker based on models of professional behavior and cognitive processes of decision maker in poorly formalized tasks with high uncertainty.
