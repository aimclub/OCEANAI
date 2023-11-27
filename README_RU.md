# [OCEANAI](https://oceanai.readthedocs.io/ru/latest/)

[![SAI](./docs/source/_static/badges/SAI_badge_flat.svg)](https://sai.itmo.ru/)
[![ITMO](./docs/source/_static/badges/ITMO_badge_flat_rus.svg)](https://itmo.ru/ru/)

![PyPI](https://img.shields.io/pypi/v/oceanai)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oceanai)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/oceanai)
![GitHub repo size](https://img.shields.io/github/repo-size/dmitryryumin/oceanai)
![PyPI - Status](https://img.shields.io/pypi/status/oceanai)
![PyPI - License](https://img.shields.io/pypi/l/oceanai)
![GitHub top language](https://img.shields.io/github/languages/top/dmitryryumin/oceanai)
![Documentation Status](https://readthedocs.org/projects/oceanai/badge/?version=latest)

---

| [Документация на английском](https://oceanai.readthedocs.io/en/latest/index.html) |
|-----------------------------------------------------------------------------------|

---

<h4 align="center"><span style="color:#EC256F;">Описание</span></h4>

> **[OCEANAI](https://oceanai.readthedocs.io/ru/latest/)** - библиотека с открытым исходным кодом, состоящая из набора алгоритмов интеллектуального анализа поведения человека на основе его мультимодальных данных для автоматического оценивания уровня отдельных персональных качеств личности человека для выполнения профессиональных обязанностей. Библиотека оценивает 5 качеств: Открытость опыту (**O**penness), Добросовестность (**C**onscientiousness), Экстраверсия (**E**xtraversion), Доброжелательность (**A**greeableness), Нейротизм/невротизм (**N**euroticism).

---

**[OCEANAI](https://oceanai.readthedocs.io/ru/latest/)** включает четыре основных алгоритма:

1. Алгоритм анализа аудиоинформации (ААИ).
2. Алгоритм анализа видеоинформации (АВИ).
3. Алгоритм анализа текстовой информации (АТИ).
4. Алгоритм мультимодального объединения информации (МОИ).

Алгоритмы ААИ, АВИ и АТИ реализуют функции сильного искусственного интеллекта (ИИ) в части комплексирования акустических, визуальных и текстовых признаков, построенных на различных принципах (экспертных и нейросетевых), т.е. данные алгоритмы реализуют подходы композитного (гибридного) ИИ. В алгоритмах осуществляется необходимая предобработка аудио-, видео- и текстовой информации, вычисление акустических, визуальных и текстовых признаков и выдача гипотез предсказаний по ним.

Алгоритм МОИ является связующим звеном трех алгоритмов анализа информации (ААИ, АВИ и АТИ). Данный алгоритм выполняет
нейросетевое объединение признаков полученных с помощью алгоритмов ААИ, АВИ и АТИ.

Помимо основной задачи - мультимодального оценивания персональных качеств личности человека, реализованные в **[OCEANAI](https://oceanai.readthedocs.io/ru/latest/)** признаки позволят исследователям решать другие задачи анализа поведения человека, например распознавание его аффективных состояний.

Для установки библиотеки необходимо обратиться к **[Установка и обновление](https://oceanai.readthedocs.io/ru/latest/user_guide/installation.html)**.

Для работы со аудиоинформацией следует обратиться к **[Аудиообработка информации](https://oceanai.readthedocs.io/ru/latest/user_guide/samples/audio.html)**.

Для работы со видеоинформацией следует обратиться к **[Видеообработка информации](https://oceanai.readthedocs.io/ru/latest/user_guide/samples/video.html)**.

Для работы с текстовой информацией следует обратиться к **[Текстовая обработка информации](https://oceanai.readthedocs.io/ru/latest/user_guide/samples/text.html)**.

Для работы с мультимодальной информацией следует обращаться к **[Мультимодальная обработка информации](https://oceanai.readthedocs.io/ru/latest/user_guide/samples/multimodal.html)**.

Библиотека решает практические задачи:

1. **[Ранжирование потенциальных кандидатов по профессиональным обязанностям](https://oceanai.readthedocs.io/ru/latest/user_guide/notebooks/Pipeline_practical_task_1.html)**.
2. **[Прогнозирование потребительских предпочтений на промышленные товары](https://oceanai.readthedocs.io/ru/latest/user_guide/notebooks/Pipeline_practical_task_2.html)**.

**[OCEANAI](https://oceanai.readthedocs.io/ru/latest/)** использует самые актуальные библиотеки с открытым исходным кодом для обработки аудио и видеоинформации: **[librosa](https://librosa.org/)**,
**[openSMILE](https://audeering.github.io/opensmile-python/)**,
**[openCV](https://pypi.org/project/opencv-python/)**,
**[mediapipe](https://google.github.io/mediapipe/getting_started/python)**.

**[OCEANAI](https://github.com/DmitryRyumin/oceanai)** написана на языке программирования
**[python](https://www.python.org/)**. Нейросетевые модели
реализованы и обучены с использованием библиотеки с открытым исходным кодом
**[TensorFlow](https://www.tensorflow.org/)**.

---

## Исследовательские данные

Библиотека **[OCEANAI](https://oceanai.readthedocs.io/ru/latest/)** была апробирована на двух корпусах:

1. Общедоступном и крупномаштабном корпусе **[First Empressions V2](https://chalearnlap.cvc.uab.cat/dataset/24/description/)**.
2. На первом общедоступном рускоязычном мультимодальном корпусе для оценки персональных качеств - **[Multimodal Personality Traits Assessment (MuPTA) Corpus](https://hci.nw.ru/ru/pages/mupta-corpus)**.

---

| [Команда разработчиков](https://oceanai.readthedocs.io/ru/latest/about.html) |
|------------------------------------------------------------------------------|

---

## Свидетельство о государственной регистрации программы для ЭВМ

**[Библиотека алгоритмов интеллектуального анализа поведения человека на основе его мультимодальных данных, обеспечивающих оценивание уровня отдельных персональных качеств личности человека для выполнения профессиональных обязанностей (OCEAN-AI)](https://new.fips.ru/registers-doc-view/fips_servlet?DB=EVM&DocNumber=2023613724&TypeFile=html)**

---

## Поддержка

Исследование проводится при поддержке [Исследовательского центра сильного искусственного интеллекта в промышленности](https://sai.itmo.ru/) [Университета ИТМО](https://itmo.ru) в рамках мероприятия программы центра: Разработка и испытания экспериментального образца библиотеки алгоритмов сильного ИИ в части гибридного принятия решений на базе взаимодействия ИИ и ЛПР на основе моделей профессионального поведения и когнитивных процессов ЛПР в трудно формализуемых задачах с высокой неопределенностью.
