.. meta::
   :description: Персональные качества личности человека
   :keywords: oceanai, machine learning, statistics, computer vision, artificial intelligence, preprocessing

Документация библиотеки алгоритмов сильного искусственного интеллекта - ``OCEANAI``
===================================================================================

`OCEANAI <https://oceanai.readthedocs.io/ru/latest/>`_ - библиотека с открытым исходным кодом, состоящая из набора алгоритмов интеллектуального анализа поведения человека на основе его мультимодальных данных для автоматического оценивания уровня отдельных персональных качеств личности человека для выполнения профессиональных обязанностей. Библиотека оценивает 5 качеств: Открытость опыту (**O**\ p\ enness), Добросовестность (**C**\ o\ nscientiousness), Экстраверсия (**E**\ x\ traversion), Доброжелательность (**A**\ g\ reeableness), Нейротизм/невротизм (**N**\ e\ uroticism).

.. image:: https://img.shields.io/pypi/v/oceanai
   :alt: PyPI (image)
.. image:: https://img.shields.io/pypi/pyversions/oceanai
   :alt: PyPI - Python Version (image)
.. image:: https://img.shields.io/pypi/implementation/oceanai
   :alt: PyPI - Implementation (image)
.. image:: https://img.shields.io/github/repo-size/dmitryryumin/oceanai
   :alt: GitHub repo size (image)
.. image:: https://img.shields.io/pypi/status/oceanai
   :alt: PyPI - Status (image)
.. image:: https://img.shields.io/pypi/l/oceanai
   :alt: PyPI - License (image)
.. image:: https://img.shields.io/github/languages/top/dmitryryumin/oceanai
   :alt: GitHub top language (image)
.. image:: https://readthedocs.org/projects/oceanai/badge/?version=latest
   :target: https://oceanai.readthedocs.io/ru/latest/?badge=latest
   :alt: Documentation Status (image)

-----

`OCEANAI <https://oceanai.readthedocs.io/ru/latest/>`_ включает три основных алгоритма:

#. Алгоритм анализа аудиоинформации (ААИ).
#. Алгоритм анализа видеоинформации (АВИ).
#. Алгоритм мультимодального объединения информации (МОИ).

Алгоритмы ААИ и АВИ реализуют функции сильного искусственного интеллекта (ИИ) в части комплексирования акустических и визуальных признаков, построенных на различных принципах (экспертных и нейросетевых), т.е. данные алгоритмы реализуют подходы композитного (гибридного) ИИ. В алгоритмах осуществляется необходимая предобработка аудио- и видеоинформации, вычисление визуальных и акустических признаков и выдача гипотез предсказаний по ним.

Алгоритм МОИ является связующим звеном двух алгоритмов анализа информации (ААИ и АВИ). Данный алгоритм выполняет
взвешенное нейросетевое объединение гипотез предсказаний полученных с помощью алгоритмов ААИ и АВИ.

Помимо основной задачи - мультимодального оценивания персональных качеств личности человека, реализованные в
`OCEANAI <https://oceanai.readthedocs.io/ru/latest/>`_ признаки позволят исследователям решать другие задачи анализа поведения
человека, например распознавание его аффективных состояний.

`OCEANAI <https://oceanai.readthedocs.io/ru/latest/>`_ использует самые актуальные библиотеки с открытым исходным кодом
для обработки аудио и видеоинформации: `librosa <https://librosa.org/>`_,
`openSMILE <https://audeering.github.io/opensmile-python/>`_,
`openCV <https://pypi.org/project/opencv-python/>`_,
`mediapipe <https://google.github.io/mediapipe/getting_started/python>`_.

`OCEANAI <https://oceanai.readthedocs.io/ru/latest/>`_ написана на языке программирования
`python <https://www.python.org/>`_. Нейросетевые модели
реализованы и обучены с использованием библиотеки с открытым исходным кодом
`TensorFlow <https://www.tensorflow.org/>`_.

-----

Свидетельство о государственной регистрации программы для ЭВМ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Библиотека алгоритмов интеллектуального анализа поведения человека на основе его мультимодальных данных, обеспечивающих оценивание уровня отдельных персональных качеств личности человека для выполнения профессиональных обязанностей (OCEAN-AI) <https://new.fips.ru/registers-doc-view/fips_servlet?DB=EVM&DocNumber=2023613724&TypeFile=html>`_

-----

Разработка поддерживается исследовательским центром
`«Сильный искусственный интеллект в промышленности» <https://sai.itmo.ru/>`_ `Университета ИТМО <https://itmo.ru>`_.

.. figure:: _static/AIM-Strong_Sign_Norm-01_Colors.svg
   :scale: 100 %
   :align: center
   :alt: Сильный искусственный интеллект в промышленности (Университет ИТМО)
   :target: https://sai.itmo.ru

-----

.. toctree::
   :maxdepth: 1
   :caption: Содержание:

   user_guide/index
   modules/index
   modules/class_diagram
   about
   faq

..
   .. sidebar-links::
      :caption: Дополнительные ссылки:
      :github:
      :pypi: oceanai

Индексация
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
