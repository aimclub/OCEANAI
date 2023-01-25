# [OCEANAI](https://github.com/DmitryRyumin/oceanai/blob/main/README_RU.md)

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

> **[OCEANAI](https://github.com/DmitryRyumin/oceanai)** - библиотека с открытым исходным кодом, состоящая из набора алгоритмов интеллектуального анализа поведения человека на основе его мультимодальных данных для автоматического оценивания уровня отдельных персональных качеств личности человека для выполнения профессиональных обязанностей. Библиотека оценивает 5 качеств: Открытость опыту (**O**penness), Добросовестность (**C**onscientiousness), Экстраверсия (**E**xtraversion), Доброжелательность (**A**greeableness), Нейротизм/невротизм (**N**euroticism).

---

**[OCEANAI](https://github.com/DmitryRyumin/oceanai)** включает три основных алгоритма:

1. Алгоритм анализа аудиоинформации (ААИ).
2. Алгоритм анализа видеоинформации (АВИ).
3. Алгоритм мультимодального объединения информации (МОИ).

Алгоритмы ААИ и АВИ реализуют функции сильного искусственного интеллекта (ИИ) в части комплексирования акустических и визуальных признаков, построенных на различных принципах (экспертных и нейросетевых), т.е. данные алгоритмы реализуют подходы композитного (гибридного) ИИ. В алгоритмах осуществляется необходимая предобработка аудио- и видеоинформации, вычисление визуальных и акустических признаков и выдача гипотез предсказаний по ним.

Алгоритм МОИ является связующим звеном двух алгоритмов анализа информации (ААИ и АВИ). Данный алгоритм выполняет
взвешенное нейросетевое объединение гипотез предсказаний полученных с помощью алгоритмов ААИ и АВИ.

Помимо основной задачи - мультимодального оценивания персональных качеств личности человека, реализованные в **[OCEANAI](https://github.com/DmitryRyumin/oceanai)** признаки позволят исследователям решать другие задачи анализа поведения человека, например распознавание его аффективных состояний.

Для установки библиотеки необходимо обратиться к **[Установка и обновление](https://oceanai.readthedocs.io/ru/latest/user_guide/installation.html)**.

Для работы со аудиоинформацией следует обратиться к **[Аудиообработка информации](https://oceanai.readthedocs.io/ru/latest/user_guide/samples/audio.html)**.

Для работы со видеооинформацией следует обратиться к **[Видеообработка информации](https://oceanai.readthedocs.io/ru/latest/user_guide/samples/video.html)**.

Для работы с аудиовизуальной информацией следует обращаться к **[Мультимодальная обработка информации](https://oceanai.readthedocs.io/ru/latest/user_guide/samples/multimodal.html)**.

Библиотека решает практические задачи:

1. **[Ранжирование потенциальных кандидатов по профессиональным обязанностям](https://oceanai.readthedocs.io/ru/latest/user_guide/notebooks/Pipeline_practical_task_1.html)**.
2. **[Прогнозирование потребительских предпочтений на промышленные товары](https://oceanai.readthedocs.io/ru/latest/user_guide/notebooks/Pipeline_practical_task_2.html)**.

**[OCEANAI](https://github.com/DmitryRyumin/oceanai)** использует самые актуальные библиотеки с открытым исходным кодом для обработки аудио и видеоинформации: **[librosa](https://librosa.org/)**,
**[openSMILE](https://audeering.github.io/opensmile-python/)**,
**[openCV](https://pypi.org/project/opencv-python/)**,
**[mediapipe](https://google.github.io/mediapipe/getting_started/python)**.

**[OCEANAI](https://github.com/DmitryRyumin/oceanai)** написана на языке программирования
**[python](https://www.python.org/)**. Нейросетевые модели
реализованы и обучены с использованием библиотеки с открытым исходным кодом
**[TensorFlow](https://www.tensorflow.org/)**.

---

| [Команда разработчиков](https://oceanai.readthedocs.io/ru/latest/about.html) |
|------------------------------------------------------------------------------|

---

## Поддержка
Разработка поддерживается исследовательским центром [**«Сильный искусственный интеллект в промышленности»**](<https://sai.itmo.ru/>) [**Университета ИТМО**](https://itmo.ru).

<h4 align="center">
    <a href="https://sai.itmo.ru/" target="_blank">
        <img src="https://raw.githubusercontent.com/DmitryRyumin/OCEANAI/main/docs/source/_static/AIM-Strong_Sign_Norm-01_Colors.svg" alt="AI Center" width="60%"/>
    </a>
</h4>


