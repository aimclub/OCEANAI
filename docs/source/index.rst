.. meta::
   :description: Персональные качества личности человека
   :keywords: oceanai, machine learning, statistics, computer vision, artificial intelligence, preprocessing

Документация библиотеки алгоритмов сильного искусственного интеллекта - ``OCEAN-AI``
====================================================================================

`OCEAN-AI <https://oceanai.readthedocs.io/ru/latest/>`_ - библиотека с открытым исходным кодом, состоящая из набора алгоритмов интеллектуального анализа поведения человека на основе его мультимодальных данных для автоматического оценивания уровня отдельных персональных качеств личности человека (ПКЛЧ). Библиотека оценивает 5 ПКЛЧ: Открытость опыту (**O**\ p\ enness), Добросовестность (**C**\ o\ nscientiousness), Экстраверсия (**E**\ x\ traversion), Доброжелательность (**A**\ g\ reeableness), Эмоциональная стабильность (Non-**N**\ e\ uroticism).

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

-----

.. figure:: _static/Pipeline_OCEANAI.svg
   :scale: 80 %
   :align: center
   :alt: Функциональная схема библиотеки OCEAN-AI

-----

`OCEAN-AI <https://oceanai.readthedocs.io/ru/latest/>`_ включает четыре основных алгоритма:

#. Алгоритм анализа аудиоинформации (ААИ).
#. Алгоритм анализа видеоинформации (АВИ).
#. Алгоритм анализа текстовой информации (АТИ).
#. Алгоритм мультимодального объединения информации (МОИ).

Алгоритмы ААИ, АВИ и АТИ реализуют функции сильного искусственного интеллекта (ИИ) в части комплексирования акустических, визуальных и текстовых признаков, построенных на различных принципах (экспертных и нейросетевых), т.е. данные алгоритмы реализуют подходы композитного (гибридного) ИИ. В алгоритмах осуществляется необходимая предобработка аудио-, видео- и текстовой информации, вычисление акустических, визуальных и текстовых признаков и выдача гипотез предсказаний по ним.

Алгоритм МОИ является связующим звеном трех алгоритмов анализа информации (ААИ, АВИ и АТИ). Данный алгоритм выполняет
нейросетевое объединение признаков полученных с помощью алгоритмов ААИ, АВИ и АТИ.

`OCEAN-AI <https://oceanai.readthedocs.io/ru/latest/>`_ предоставляет примеры решения прикладных задач на основе полученных гипотез предсказаний оценок ПКЛЧ:

#. Ранжирование потенциальных кандидатов для выполнения профессиональных обязанностей:
    #. по группам профессий;
    #. по профессиональным навыкам.
#. Прогнозирование потребительских предпочтений по выбору промышленных потребительских товаров:
    #. на примере характеристик автомобиля;
    #. на примере категорий применения мобильного устройства.
#. Формирование эффективных рабочих коллективов:
    #. поиск подходящего младшего коллеги;
    #. поиск подходящего старшего коллеги.

Помимо основной задачи - мультимодального оценивания персональных качеств личности человека, реализованные в
`OCEAN-AI <https://oceanai.readthedocs.io/ru/latest/>`_ признаки позволят исследователям решать другие задачи анализа поведения
человека, например распознавание его аффективных состояний.

`OCEAN-AI <https://oceanai.readthedocs.io/ru/latest/>`_ использует самые актуальные библиотеки с открытым исходным кодом
для обработки аудио-, видео- и текстовой информации: `librosa <https://librosa.org/>`_,
`openSMILE <https://audeering.github.io/opensmile-python/>`_,
`openCV <https://pypi.org/project/opencv-python/>`_,
`mediapipe <https://google.github.io/mediapipe/getting_started/python>`_, `transformers <https://pypi.org/project/transformers/>`_.

`OCEAN-AI <https://oceanai.readthedocs.io/ru/latest/>`_ написана на языке программирования
`python <https://www.python.org/>`_. Нейросетевые модели
реализованы и обучены с использованием библиотеки с открытым исходным кодом
`TensorFlow <https://www.tensorflow.org/>`_.

-----

Исследовательские данные
~~~~~~~~~~~~~~~~~~~~~~~~

Библиотека `OCEAN-AI <https://oceanai.readthedocs.io/ru/latest/>`_ была апробирована на двух корпусах:

#. Общедоступном и крупномаштабном корпусе `First Impressions V2 <https://chalearnlap.cvc.uab.cat/dataset/24/description/>`_.
#. На первом общедоступном рускоязычном мультимодальном корпусе для оценки персональных качеств - `Multimodal Personality Traits Assessment (MuPTA) Corpus <https://hci.nw.ru/ru/pages/mupta-corpus>`_.

-----

Свидетельство о государственной регистрации программы для ЭВМ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Библиотека алгоритмов интеллектуального анализа поведения человека на основе его мультимодальных данных, обеспечивающих оценивание уровня отдельных персональных качеств личности человека для выполнения профессиональных обязанностей (OCEAN-AI) <https://new.fips.ru/registers-doc-view/fips_servlet?DB=EVM&DocNumber=2023613724&TypeFile=html>`_

Свидетельство о государственной регистрации базы данных
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Корпус для мультимодального оценивания персональных качеств личности человека (MuPTA - Multimodal Personality Traits Assessment Corpus) <https://new.fips.ru/registers-doc-view/fips_servlet?DB=DB&DocNumber=2023624011&TypeFile=html>`_

-----

Публикации
~~~~~~~~~~

Журналы
-------

.. code-block:: bibtex

    @article{ryumina24_prl,
        author = {Ryumina, Elena and Markitantov, Maxim and Ryumin, Dmitry and Karpov, Alexey},
        title = {{Gated Siamese Fusion Network based on Multimodal Deep and Hand-Crafted Features for Personality Traits Assessment}},
        volume = {185},
        pages = {45--51},
        journal = {Pattern Recognition Letters},
        year = {2024},
        issn = {0167-8655},
        doi = {10.1016/j.patrec.2024.07.004},
        url = {https://www.sciencedirect.com/science/article/pii/S0167865524002071},
    }

.. code-block:: bibtex

    @article{ryumina24_eswa,
        author = {Elena Ryumina and Maxim Markitantov and Dmitry Ryumin and Alexey Karpov},
        title = {OCEAN-AI Framework with EmoFormer Cross-Hemiface Attention Approach for Personality Traits Assessment},
        journal = {Expert Systems with Applications},
        volume = {239},
        pages = {122441},
        year = {2024},
        doi = {https://doi.org/10.1016/j.eswa.2023.122441},
    }

.. code-block:: bibtex

    @article{ryumina22_neurocomputing,
        author = {Elena Ryumina and Denis Dresvyanskiy and Alexey Karpov},
        title = {In Search of a Robust Facial Expressions Recognition Model: A Large-Scale Visual Cross-Corpus Study},
        journal = {Neurocomputing},
        volume = {514},
        pages = {435--450},
        year = {2022},
        doi = {https://doi.org/10.1016/j.neucom.2022.10.013},
    }

Конференции
-----------

.. code-block:: bibtex

    @inproceedings{ryumina24_interspeech,
        author = {Elena Ryumina and Dmitry Ryumin and and Alexey Karpov},
        title = {OCEAN-AI: Open Multimodal Framework for Personality Traits Assessment and HR-Processes Automatization},
        year = {2024},
        booktitle = {INTERSPEECH},
        pages = {in press},
        doi = {in press},
    }

.. code-block:: bibtex

    @inproceedings{ryumina23_interspeech,
        author = {Elena Ryumina and Dmitry Ryumin and Maxim Markitantov and Heysem Kaya and Alexey Karpov},
        title = {Multimodal Personality Traits Assessment (MuPTA) Corpus: The Impact of Spontaneous and Read Speech},
        year = {2023},
        booktitle = {INTERSPEECH},
        pages = {4049--4053},
        doi = {https://doi.org/10.21437/Interspeech.2023-1686},
    }

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
