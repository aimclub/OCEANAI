#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ядро
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings
for warn in [UserWarning, FutureWarning]: warnings.filterwarnings('ignore', category = warn)

from dataclasses import dataclass # Класс данных

import os                 # Взаимодействие с файловой системой
import sys                # Доступ к некоторым переменным и функциям Python
import re                 # Регулярные выражения
import time               # Работа со временем
import numpy as np        # Научные вычисления
import scipy
import pandas as pd       # Обработка и анализ данных
import opensmile          # Анализ, обработка и классификация звука
import jupyterlab as jlab # Интерактивная среда разработки для работы с блокнотами, кодом и данными
import requests           # Отправка HTTP запросов
import librosa            # Обработка аудио
import audioread          # Декодирование звука
import sklearn            # Машинное обучение и интеллектуальный анализ данных
import cv2                # Алгоритмы компьютерного зрения
import mediapipe as mp    # Набор нейросетевых моделей и решений для компьютерного зрения
import IPython
import logging
import pymediainfo        # Получение meta данных из медиафайлов
import urllib.error       # Обработка ошибок URL
import math

from datetime import datetime # Работа со временем
from typing import List, Dict, Tuple, Union, Optional, Iterable # Типы данных

from IPython import get_ipython
from IPython.display import Markdown, display, clear_output

# Персональные
import oceanai                                     # oceanai - персональные качества личности человека
from oceanai.modules.core.settings import Settings # Глобальный файл настроек

# Порог регистрации сообщений TensorFlow
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf # Машинное обучение от Google
import keras

# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class CoreMessages(Settings):
    """Класс для сообщений

    Args:
        lang (str): Смотреть :attr:`~oceanai.modules.core.language.Language.lang`
        color_simple (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.color_simple`
        color_info (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.color_info`
        color_err (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.color_err`
        color_true (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.color_true`
        bold_text (bool): Смотреть :attr:`~oceanai.modules.core.settings.Settings.bold_text`
        num_to_df_display (int): Смотреть :attr:`~oceanai.modules.core.settings.Settings.num_to_df_display`
        text_runtime (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.text_runtime`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__() # Выполнение конструктора из суперкласса

        self._trac_file: str = self._('Файл')
        self._trac_line: str = self._('Линия')
        self._trac_method: str = self._('Метод')
        self._trac_type_err: str = self._('Тип ошибки')

        self._sec: str = self._('сек.')

        self._folder_not_found: str = self._oh + self._('директория "{}" не найдена ...')
        self._files_not_found: str = self._oh + self._('в указанной директории необходимые файлы не найдены ...')
        self._file_not_found: str = self._oh + self._('файл "{}" не найден ...')
        self._directory_inst_file: str = self._oh + self._('вместо файла передана директория "{}" ...')
        self._no_acoustic_signal: str = self._oh + self._('файл "{}" не содержит акустического сигнала ...')
        self._url_error_log: str = self._oh + self._('не удалось сохранить LOG файл{} ...')
        self._url_error_code_log: str = self._(' (ошибка {})')

        self._mul: str = '&#10005;' # Знак умножения

        self._get_acoustic_feature_stat: str = '{}' * 4 + self._(
            'Статистика извлеченных признаков из акустического сигнала:'
            '{}Общее количество сегментов с:'
            '{}1. экспертными признаками: {}'
            '{}2. лог мел-спектрограммами: {}'
            '{}Размерность матрицы экспертных признаков одного сегмента: '
            '{} ') + self._mul + ' {}' + \
            self._('{}Размерность тензора с лог мел-спектрограммами одного сегмента:') + \
            '{} ' + self._mul + ' {} ' + self._mul + ' {}'

        self._get_visual_feature_stat: str = '{}' * 4 + self._(
            'Статистика извлеченных признаков из визуального сигнала:'
            '{}Общее количество сегментов с:'
            '{}1. экспертными признаками: {}'
            '{}2. нейросетевыми признаками: {}'
            '{}Размерность матрицы экспертных признаков одного сегмента: '
            '{} ') + self._mul + ' {}' + \
            self._('{}Размерность тензора с нейросетевыми признаками одного сегмента:') + \
            '{} ' + self._mul + ' {} ' + \
            self._('{}Понижение кадровой частоты: с') + '{} ' + self._('до') + ' {} '

        self._curr_progress_union_predictions: str = '{} ' + self._from_precent + ' {} ({}%) ... {} ...'

        self._sum_ranking_exceeded: str = self._oh + self._(
            'сумма весов для ранжирования персональных качеств должна быть равна 100 ...'
        )

        self._dataframe_empty: str = self._oh + self._('DataFrame с данными пуст ...')

# ######################################################################################################################
# Ядро модулей
# ######################################################################################################################
@dataclass
class Core(CoreMessages):
    """Класс-ядро модулей

    Args:
        lang (str): Смотреть :attr:`~oceanai.modules.core.language.Language.lang`
        color_simple (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.color_simple`
        color_info (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.color_info`
        color_err (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.color_err`
        color_true (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.color_true`
        bold_text (bool): Смотреть :attr:`~oceanai.modules.core.settings.Settings.bold_text`
        num_to_df_display (int): Смотреть :attr:`~oceanai.modules.core.settings.Settings.num_to_df_display`
        text_runtime (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.text_runtime`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__() # Выполнение конструктора из суперкласса

        self._start_time: Union[int, float] = -1 # Старт времени выполнения
        self._runtime: Union[int, float] = -1 # Время выполнения

        self._notebook_history_output: List[str] = [] # История вывода сообщений в ячейке Jupyter

        self._df_pkgs: pd.DataFrame = pd.DataFrame() # DataFrame c версиями установленных библиотек

        # Персональные качества личности человека (Порядок только такой)
        self._b5: Dict[str, Tuple[str, ...]] = {
            'en': ('openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'),
            'ru': (self._('открытость опыту'), self._('добросовестность'), self._('экстраверсия'),
                   self._('доброжелательность'), self._('нейротизм'))
        }

        # Веса для нейросетевых архитектур
        self._weights_for_big5: Dict[str, Dict] = {
            'audio': {
                'hc': {
                    'sberdisk': 'https://download.sberdisk.ru/download/file/400635799?token=MMRrak8fMsyzxLE&filename=weights_2022-05-05_11-27-55.h5',
                },
                'nn': {
                    'sberdisk': 'https://download.sberdisk.ru/download/file/400635678?token=W6LCtD33FQHnYEz&filename=weights_2022-05-03_07-46-14.h5',
                },
                'b5': {
                    'openness': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/405035301?token=443WRA9MFWqWBAE&filename=weights_2022-06-15_16-16-20.h5',
                    },
                    'conscientiousness': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/405034601?token=eDG28m3H6c8bWoE&filename=weights_2022-06-15_16-21-57.h5',
                    },
                    'extraversion': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/405034830?token=3daBSTYnmZaesee&filename=weights_2022-06-15_16-26-41.h5',
                    },
                    'agreeableness': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/405034397?token=52ZPHMjb4CFmdYa&filename=weights_2022-06-15_16-32-51.h5',
                    },
                    'neuroticism': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/405035156?token=q8CZJ99rZqcNxkM&filename=weights_2022-06-15_16-37-46.h5',
                    },
                },
            },
            'video': {
                'hc': {
                    'sberdisk': 'https://download.sberdisk.ru/download/file/412059444?token=JXerCfAjJZg6crD&filename=weights_2022-08-27_18-53-35.h5',
                },
                'nn': {
                    'sberdisk': 'https://download.sberdisk.ru/download/file/412059478?token=85KeW6q4QKy6kP8&filename=weights_2022-03-22_16-31-48.h5',
                },
                'fe': {
                    'sberdisk': 'https://download.sberdisk.ru/download/file/414207833?token=ygzxWEkndjSMnEL&filename=weights_2022-11-01_12-27-07.h5'
                },
                'b5': {
                    'openness': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/415127050?token=rfpy9TLdbeXtiN7&filename=weights_2022-06-15_16-46-30.h5',
                    },
                    'conscientiousness': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/415126986?token=PnjzaHaR3wPg2RT&filename=weights_2022-06-15_16-48-50.h5',
                    },
                    'extraversion': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/415127012?token=s5aTwbt8DBkt7G4&filename=weights_2022-06-15_16-54-06.h5',
                    },
                    'agreeableness': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/415126845?token=joN7TMHk59Gffsf&filename=weights_2022-06-15_17-02-03.h5',
                    },
                    'neuroticism': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/415127032?token=NEBSsE7mjyjen3o&filename=weights_2022-06-15_17-06-15.h5',
                    },
                },
            },
            'av': {
                'b5': {
                    'openness': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/425515223?token=Btoo2flpzGewhry&filename=weights_2022-08-28_11-14-35.h5',
                    },
                    'conscientiousness': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/425515311?token=bUjloqk32e4wasj&filename=weights_2022-08-28_11-08-10.h5',
                    },
                    'extraversion': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/425515337?token=IC44ovFFcnj4DZl&filename=weights_2022-08-28_11-17-57.h5',
                    },
                    'agreeableness': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/425515346?token=gFjvtM2HIabtsvc&filename=weights_2022-08-28_11-25-11.h5',
                    },
                    'neuroticism': {
                        'sberdisk': 'https://download.sberdisk.ru/download/file/425515375?token=pPpzOQC9z6WMzNt&filename=weights_2022-06-14_21-44-09.h5',
                    },
                },
            },
        }

        # Верные предсказания для подсчета точности
        self._true_traits: Dict[str, str] = {
            'sberdisk': 'https://download.sberdisk.ru/download/file/410305241?token=TFePK6w5CW6ADnq&filename=data_true_traits.csv',
        }

        self._df_files: pd.DataFrame = pd.DataFrame() # DataFrame с данными
        self._df_files_ranking: pd.DataFrame = pd.DataFrame() # DataFrame с ранжированными данными
        self._dict_of_files: Dict[str, List[Union[int, str, float]]] = {} # Словарь для DataFrame с данными

        self._df_accuracy: pd.DataFrame = pd.DataFrame() # DataFrame с результатами вычисления точности
        # Словарь для DataFrame с результатами вычисления точности
        self._dict_of_accuracy: Dict[str, List[Union[int, float]]] = {}

        self._keys_id: str = 'ID' # Идентификатор
        self._keys_score: str = 'Candidate score' # Комплексная оценка кандидатов

        self._ext_for_logs: str = '.csv' # Расширение для сохранения lOG файлов

        # Тип файла с META информацией
        self._type_meta_info: Dict[str, List[str]] = {
            'Video': ['format', 'duration', 'other_width', 'other_height', 'other_display_aspect_ratio',
                      'minimum_frame_rate', 'frame_rate', 'maximum_frame_rate', 'other_bit_rate', 'encoded_date']
        }

        # ----------------------- Только для внутреннего использования внутри класса

        self.__tab: str = '&nbsp;' * 4 # Табуляция (в виде пробелов)

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_notebook_(self) -> bool:
        """Получение результата определения запуска библиотеки в Jupyter или аналогах

        Returns:
            bool: **True** если библиотека запущена в Jupyter или аналогах, в обратном случае **False**

        .. dropdown:: Пример

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                print(core.is_notebook_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                True
        """

        return self.__is_notebook()

    # Получение времени выполнения
    @property
    def runtime_(self):
        """Получение времени выполнения

        Returns:
            Union[int, float]: Время выполнения

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._r_start()
                for cnt in range(0, 10000000): res = cnt * 2
                core._r_end(out = False)

                print(core.runtime_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                0.838

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                print(core.runtime_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                -1

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._r_start()
                for cnt in range(0, 10000000): res = cnt * 2

                print(core.runtime_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                -1
        """

        return self._runtime

    @property
    def dict_of_files_(self) -> Dict[str, List[Union[int, str, float]]]:
        """Получение словаря для DataFrame с данными

        .. hint:: На основе данного словаря формируется DataFrame с данными ``df_files_``

        Returns:
            Dict[str, List[Union[int, str, float]]]: Словарь для DataFrame с данными

        .. dropdown:: Пример

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                len(core.dict_of_files_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                0
        """

        return self._dict_of_files

    @property
    def dict_of_accuracy_(self) -> Dict[str, List[Union[int, float]]]:
        """Получение словаря для DataFrame с результатами вычисления точности

        .. hint:: На основе данного словаря формируется DataFrame с данными ``df_accuracy_``

        Returns:
            Dict[str, List[Union[int, float]]]: Словарь для DataFrame с результатами вычисления точности

        .. dropdown:: Пример

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                len(core.dict_of_accuracy_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                0
        """

        return self._dict_of_accuracy

    @property
    def df_pkgs_(self) -> pd.DataFrame:
        """Получение DataFrame c версиями установленных библиотек

        Returns:
            pd.DataFrame: **DataFrame** c версиями установленных библиотек

        .. dropdown:: Пример

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core.libs_vers(out = False, runtime = True, run = True)
                core.df_pkgs_

            .. output-cell::
                :execution-count: 1
                :linenos:

                |----|--------------|---------|
                |    | Package      | Version |
                |----|--------------|---------|
                | 1  | TensorFlow   | 2.11.0  |
                | 2  | Keras        | 2.11.0  |
                | 3  | OpenCV       | 4.6.0   |
                | 4  | MediaPipe    | 0.9.0   |
                | 5  | NumPy        | 1.23.5  |
                | 6  | SciPy        | 1.9.3   |
                | 7  | Pandas       | 1.5.2   |
                | 8  | Scikit-learn | 1.1.3   |
                | 9  | OpenSmile    | 2.4.1   |
                | 10 | Librosa      | 0.9.2   |
                | 11 | AudioRead    | 3.0.0   |
                | 12 | IPython      | 8.7.0   |
                | 13 | PyMediaInfo  | 6.0.1   |
                | 14 | Requests     | 2.28.1  |
                | 15 | JupyterLab   | 3.5.0   |
                |----|--------------|---------|
        """

        return self._df_pkgs

    @property
    def df_files_(self) -> pd.DataFrame:
        """Получение DataFrame c данными

        Returns:
            pd.DataFrame: **DataFrame** c данными

        .. dropdown:: Пример

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                len(core.df_files_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                0
        """

        return self._df_files

    @property
    def df_files_ranking_(self) -> pd.DataFrame:
        """Получение DataFrame c ранжированными данными

        Returns:
            pd.DataFrame: **DataFrame** c данными

        .. dropdown:: Пример

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                len(core.df_files_ranking_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                0
        """

        return self._df_files_ranking

    @property
    def df_accuracy_(self) -> pd.DataFrame:
        """Получение DataFrame с результатами вычисления точности

        Returns:
            pd.DataFrame: **DataFrame** с результатами вычисления точности

        .. dropdown:: Пример

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                len(core.df_accuracy_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                0
        """

        return self._df_accuracy

    @property
    def weights_for_big5_(self) -> Dict[str, Dict]:
        """Получение весов для нейросетевых архитектур

        Returns:
            Dict: Словарь с весами для нейросетевых архитектур

        .. dropdown:: Пример
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core()
                core.weights_for_big5_

            .. output-cell::
                :execution-count: 1
                :linenos:
                :tab-width: 12

                {
                    'audio': {
                        'hc': {
                            'sberdisk': 'https://download.sberdisk.ru/download/file/400635799?token=MMRrak8fMsyzxLE&filename=weights_2022-05-05_11-27-55.h5',
                        },
                        'nn': {
                            'sberdisk': 'https://download.sberdisk.ru/download/file/400635678?token=W6LCtD33FQHnYEz&filename=weights_2022-05-03_07-46-14.h5',
                        },
                        'b5': {
                            'openness': {
                                'sberdisk': 'https://download.sberdisk.ru/download/file/405035301?token=443WRA9MFWqWBAE&filename=weights_2022-06-15_16-16-20.h5',
                            },
                            'conscientiousness': {
                                'sberdisk': 'https://download.sberdisk.ru/download/file/405034601?token=eDG28m3H6c8bWoE&filename=weights_2022-06-15_16-21-57.h5',
                            },
                            'extraversion': {
                                'sberdisk': 'https://download.sberdisk.ru/download/file/405034830?token=3daBSTYnmZaesee&filename=weights_2022-06-15_16-26-41.h5',
                            },
                            'agreeableness': {
                                'sberdisk': 'https://download.sberdisk.ru/download/file/405034397?token=52ZPHMjb4CFmdYa&filename=weights_2022-06-15_16-32-51.h5',
                            },
                            'neuroticism': {
                                'sberdisk': 'https://download.sberdisk.ru/download/file/405035156?token=q8CZJ99rZqcNxkM&filename=weights_2022-06-15_16-37-46.h5',
                            },
                        },
                    },
                    'video': {
                        'hc': {
                            'sberdisk': 'https://download.sberdisk.ru/download/file/412059444?token=JXerCfAjJZg6crD&filename=weights_2022-08-27_18-53-35.h5',
                        },
                        'nn': {
                            'sberdisk': 'https://download.sberdisk.ru/download/file/412059478?token=85KeW6q4QKy6kP8&filename=weights_2022-03-22_16-31-48.h5',
                        },
                        'fe': {
                            'sberdisk': 'https://download.sberdisk.ru/download/file/414207833?token=ygzxWEkndjSMnEL&filename=weights_2022-11-01_12-27-07.h5'
                        },
                        'b5': {
                            'openness': {
                                'sberdisk': 'https://download.sberdisk.ru/download/file/415127050?token=rfpy9TLdbeXtiN7&filename=weights_2022-06-15_16-46-30.h5',
                            },
                            'conscientiousness': {
                                'sberdisk': 'https://download.sberdisk.ru/download/file/415126986?token=PnjzaHaR3wPg2RT&filename=weights_2022-06-15_16-48-50.h5',
                            },
                            'extraversion': {
                                'sberdisk': 'https://download.sberdisk.ru/download/file/415127012?token=s5aTwbt8DBkt7G4&filename=weights_2022-06-15_16-54-06.h5',
                            },
                            'agreeableness': {
                                'sberdisk': 'https://download.sberdisk.ru/download/file/415126845?token=joN7TMHk59Gffsf&filename=weights_2022-06-15_17-02-03.h5',
                            },
                            'neuroticism': {
                                'sberdisk': 'https://download.sberdisk.ru/download/file/415127032?token=NEBSsE7mjyjen3o&filename=weights_2022-06-15_17-06-15.h5',
                            }
                        }
                    }
                }
        """

        return self._weights_for_big5

    @property
    def true_traits_(self) -> Dict[str, str]:
        """Получение путей к верным предсказаниям для подсчета точности

        Returns:
            Dict: Словарь с путями к верным предсказаниям для подсчета точности

        .. dropdown:: Пример
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core()
                core.true_traits_

            .. output-cell::
                :execution-count: 1
                :linenos:
                :tab-width: 12

                {
                    'sberdisk': 'https://download.sberdisk.ru/download/file/410305241?token=TFePK6w5CW6ADnq&filename=data_true_traits.csv'
                }
        """

        return self._true_traits

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (сообщения)
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _traceback() -> Dict:
        """Трассировка исключений

        .. note::
            protected (защищенный метод)

        Returns:
            Dict: Словарь с описанием исключения

        .. dropdown:: Пример
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                import pprint
                from oceanai.modules.core.core import Core

                core = Core()

                try: raise Exception
                except:
                    pp = pprint.PrettyPrinter(compact = True)
                    pp.pprint(core._traceback())

            .. output-cell::
                :execution-count: 1
                :linenos:
                :tab-width: 8

                {
                    'filename': '/var/folders/gw/w3k5kxtx0s3_nqdqw94zr8yh0000gn/T/ipykernel_22253/4179594971.py',
                    'lineno': 6,
                    'name': '<cell line: 6>',
                    'type': 'Exception'
                }
        """

        exc_type, exc_value, exc_traceback = sys.exc_info() # Получение информации об ошибке

        _trac = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__
        }

        return _trac

    def _notebook_display_markdown(self, message: str, last: bool = False, out: bool = True) -> None:
        """Отображение сообщения

        .. note::
            protected (защищенный метод)

        Args:
            message (str): Сообщение
            last (bool): Замена последнего сообщения
            out (bool): Отображение

        Returns:
            None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core._notebook_display_markdown('Сообщение')

            .. output-cell::
                :execution-count: 1
                :linenos:

                Сообщение

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core._notebook_display_markdown(1)

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-14 15:52:03] Неверные типы или значения аргументов в "Core._notebook_display_markdown" ...
        """

        if self.is_notebook_ is True:
            try:
                # Проверка аргументов
                if type(message) is not str or not message: raise TypeError
            except TypeError:
                self._inv_args(__class__.__name__, self._notebook_display_markdown.__name__, out = out); return None

            if type(last) is not bool: last = False

            self._add_notebook_history_output(message, last) # Добавление истории вывода сообщений в ячейке Jupyter

            if type(out) is not bool: out = True

            if out is True: display(Markdown(message)) # Отображение

    def _metadata_info(self, last: bool = False, out: bool = True) -> None:
        """Информация об библиотеке

        .. note::
            protected (защищенный метод)

        Args:
            last (bool): Замена последнего сообщения
            out (bool): Отображение

        Returns:
            None

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core._metadata_info(last = False, out = True)

            .. output-cell::
                :execution-count: 1
                :linenos:
                :tab-width: 8

                [2022-10-14 13:05:54] oceanai - персональные качества личности человека:
                    Авторы:
                        Рюмина Елена [ryumina_ev@mail.ru]
                        Рюмин Дмитрий [dl_03.03.1991@mail.ru]
                        Карпов Алексей [karpov@iias.spb.su]
                    Сопровождающие:
                        Рюмина Елена [ryumina_ev@mail.ru]
                        Рюмин Дмитрий [dl_03.03.1991@mail.ru]
                    Версия: 1.0.0-a7
                    Лицензия: GPLv3

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core._metadata_info(last = 1, out = [])

            .. output-cell::
                :execution-count: 2
                :linenos:
                :tab-width: 8

                [2022-10-14 13:05:54] oceanai - персональные качества личности человека:
                    Авторы:
                        Рюмина Елена [ryumina_ev@mail.ru]
                        Рюмин Дмитрий [dl_03.03.1991@mail.ru]
                        Карпов Алексей [karpov@iias.spb.su]
                    Сопровождающие:
                        Рюмина Елена [ryumina_ev@mail.ru]
                        Рюмин Дмитрий [dl_03.03.1991@mail.ru]
                    Версия: 1.0.0-a7
                    Лицензия: GPLv3
        """

        if self.is_notebook_ is True:
            tab = self.__tab

            b = '**' if self.bold_text_ is True else ''
            cr = self.color_simple_

            generate_name_with_email = lambda list1, list2: ''.join(
                map(str, map(
                    lambda l1, l2: f'<br /><span style=\"color:{cr}\">{tab * 2}{l1} [<u>{l2}</u>]</span>',
                    list1.split(', '), list2.split(', ')
                ))
            )

            author = generate_name_with_email(
                oceanai.__author__ru__ if self.lang_ == 'ru' else oceanai.__author__en__, oceanai.__email__
            )
            maintainer = generate_name_with_email(
                oceanai.__maintainer__ru__ \
                    if self.lang_ == 'ru' else oceanai.__maintainer__en__, oceanai.__maintainer_email__
            )

            # Отображение сообщения
            self._notebook_display_markdown(('{}' * 8).format(
                f'<span style=\"color:{self.color_simple_}\">{b}[</span><span style=\"color:{self.color_info_}\">',
                datetime.now().strftime(self._format_time),
                f'</span><span style=\"color:{self.color_simple_}\">]</span> ',
                f'<span style=\"color:{self.color_simple_}\">{self._metadata[0]}:</span>{b}',
                f'<br /><span style=\"color:{cr}\">{tab}{self._metadata[1]}:</span>{author}',
                f'<br /><span style=\"color:{cr}\">{tab}{self._metadata[2]}:</span>{maintainer}',
                f'<br /><span style=\"color:{cr}\">{tab}{self._metadata[3]}: <u>{oceanai.__release__}</u></span>',
                f'<br /><span style=\"color:{cr}\">{tab}{self._metadata[4]}: <u>{oceanai.__license__}</u></span></p>'
            ), last, out)

    def _inv_args(self, class_name: str, build_name: str, last: bool = False, out: bool = True) -> None:
        """Сообщение об указании неверных типов аргументов

        .. note::
            protected (защищенный метод)

        Args:
            class_name (str): Имя класса
            build_name (str): Имя метода/функции
            last (bool): Замена последнего сообщения
            out (bool): Отображение

        Returns:
            None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core._inv_args(
                    Core.__name__, core._info.__name__,
                    last = False, out = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-14 11:58:04] Неверные типы или значения аргументов в "Core._info" ...

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core._inv_args(1, '', last = False, out = True)

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-14 11:58:04] Неверные типы или значения аргументов в "Core._inv_args" ...
        """

        if self.is_notebook_ is True:
            try:
                # Проверка аргументов
                if type(class_name) is not str or not class_name or type(build_name) is not str or not build_name:
                    raise TypeError
            except TypeError: class_name, build_name = __class__.__name__, self._inv_args.__name__

            inv_args = self._invalid_arguments.format(class_name + '.' + build_name)

            if len(inv_args) == 0: inv_args = self._invalid_arguments_empty

            b = '**' if self.bold_text_ is True else ''

            # Отображение сообщения
            self._notebook_display_markdown('{}[{}{}{}] {}{}'.format(
                f'<span style=\"color:{self.color_simple_}\">{b}',
                f'</span><span style=\"color:{self.color_err_}\">',
                datetime.now().strftime(self._format_time),
                f'</span><span style=\"color:{self.color_simple_}\">', inv_args, f'{b}</span>'
            ), last, out)

    def _info(self, message: str, last: bool = False, out: bool = True) -> None:
        """Информационное сообщение

        .. note::
            protected (защищенный метод)

        Args:
            message (str): Сообщение
            last (bool): Замена последнего сообщения
            out (bool): Отображение

        Returns:
            None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._info(
                    message = 'Информационное сообщение 1',
                    last = False, out = True
                )

                core.color_simple_ = '#FFF'
                core.color_info_ = '#0B45B9'
                core.bold_text_ = False

                core._info(
                    message = 'Информационное сообщение 2',
                    last = True, out = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-14 11:35:00] Информационное сообщение 1
                [2022-10-14 11:35:00] Информационное сообщение 2

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._info(
                    message = '',
                    last = False, out = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-14 11:43:00] Неверные типы или значения аргументов в "Core._info" ...
        """

        if self.is_notebook_ is True:
            try:
                # Проверка аргументов
                if type(message) is not str or not message: raise TypeError
            except TypeError: self._inv_args(__class__.__name__, self._info.__name__, out = out); return None

            b = '**' if self.bold_text_ is True else ''

            # Отображение сообщения
            self._notebook_display_markdown(('{}' * 4).format(
                f'<span style=\"color:{self.color_simple_}\">{b}[</span><span style=\"color:{self.color_info_}\">',
                datetime.now().strftime(self._format_time),
                f'</span><span style=\"color:{self.color_simple_}\">]</span> ',
                f'<span style=\"color:{self.color_simple_}\">{message}</span>{b} '
            ), last, out)

    def _info_wrapper(self, message: str) -> str:
        """Обернутое информационное сообщение

        .. note::
            protected (защищенный метод)

        Args:
            message (str): Сообщение

        Returns:
            str: Обернутое информационное сообщение

        .. dropdown:: Пример
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                print(core._info_wrapper('Обернутое информационное сообщение 1'))

                core.color_info_ = '#0B45B9'
                print(core._info_wrapper('Обернутое информационное сообщение 2'))

            .. output-cell::
                :execution-count: 1
                :linenos:

                <span style="color:#1776D2">Обернутое информационное сообщение 1</span>
                <span style="color:#0B45B9">Обернутое информационное сообщение 2</span>
        """

        if self.is_notebook_ is True:
            return ('{}' * 3).format(f'<span style=\"color:{self.color_info_}\">', message, f'</span>')

    # Положительная информация
    def _info_true(self, message: str, last: bool = False, out: bool = True) -> None:
        """Положительная информация

        .. note::
            protected (защищенный метод)

        Args:
            message (str): Сообщение
            last (bool): Замена последнего сообщения
            out (bool): Отображение

        Returns:
            None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._info_true(
                    message = 'Информационное положительное сообщение 1',
                    last = False, out = True
                )

                core.color_true_ = '#008001'
                core.bold_text_ = False

                core._info_true(
                    message = 'Информационное положительное сообщение 2',
                    last = True, out = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                Информационное положительное сообщение 1

                Информационное положительное сообщение 2

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._info_true(
                    message = '',
                    last = False, out = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-22 16:46:56] Неверные типы или значения аргументов в "Core._info_true" ...
        """

        if self.is_notebook_ is True:
            try:
                # Проверка аргументов
                if type(message) is not str or not message: raise TypeError
            except TypeError: self._inv_args(__class__.__name__, self._info_true.__name__, out = out); return None

            b = '**' if self.bold_text_ is True else ''

            # Отображение сообщения
            self._notebook_display_markdown(
                '{}'.format(f'<span style=\"color:{self.color_true_}\">{b}{message}{b}</span>'), last, out)

    def _bold_wrapper(self, message: str) -> str:
        """Обернутое сообщение с жирным начертанием

        .. note::
            protected (защищенный метод)

        Args:
            message (str): Сообщение

        Returns:
            str: Обернутое сообщение с жирным начертанием

        .. dropdown:: Пример
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core(bold_text = False)
                print(core._bold_wrapper(
                    'Обернутое сообщение без жирного начертания'
                ))

                core.bold_text = True
                print(core._bold_wrapper(
                    'Обернутое сообщение с жирным начертанием'
                ))

            .. output-cell::
                :execution-count: 1
                :linenos:

                <span style="color:#666">Обернутое сообщение без жирного начертания</span>
                <span style="color:#666">**Обернутое сообщение с жирным начертанием**</span>
        """

        if self.is_notebook_ is True:
            b = '**' if self.bold_text_ is True else ''

            return ('{}' * 3).format(f'<span style=\"color:{self.color_simple_}\">{b}', message, f'{b}</span>')

    def _error(self, message: str, last: bool = False, out: bool = True) -> None:
        """Сообщение об ошибке

        .. note::
            protected (защищенный метод)

        Args:
            message (str): Сообщение
            last (bool): Замена последнего сообщения
            out (bool): Отображение

        Returns:
            None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._error(
                    message = 'Сообщение об ошибке 1',
                    last = False, out = True
                )

                core.color_simple_ = '#FFF'
                core.color_err_ = 'FF0000'
                core.bold_text_ = False

                core._error(
                    message = 'Сообщение об ошибке 2',
                    last = True, out = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-12 15:21:00] Сообщение об ошибке 1
                [2022-10-12 15:21:00] Сообщение об ошибке 2

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._error(
                    message = '',
                    last = False, out = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-12 17:06:04] Неверные типы или значения аргументов в "Core._error" ...
        """

        if self.is_notebook_ is True:
            try:
                # Проверка аргументов
                if type(message) is not str or not message: raise TypeError
            except TypeError: self._inv_args(__class__.__name__, self._error.__name__, out = out); return None


            b = '**' if self.bold_text_ is True else ''

            # Отображение сообщения
            self._notebook_display_markdown('{}[{}{}{}] {}{}'.format(
                f'<span style=\"color:{self.color_simple_}\">{b}', f'</span><span style=\"color:{self.color_err_}\">',
                datetime.now().strftime(self._format_time),
                f'</span><span style=\"color:{self.color_simple_}\">', message, f'{b}</span>'
            ), last, out)

    def _other_error(self, message: str, last: bool = False, out: bool = True) -> None:
        """Сообщение об прочей ошибке

        .. note::
            protected (защищенный метод)

        Args:
            message (str): Сообщение
            last (bool): Замена последнего сообщения
            out (bool): Отображение

        Returns:
            None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core()

                try: raise Exception
                except:
                    core._other_error(
                        message = 'Сообщение об ошибке 1',
                        last = False, out = True
                    )

                core.color_simple_ = '#FFF'
                core.color_err_ = 'FF0000'
                core.bold_text_ = False

                try: raise Exception
                except:
                    core._other_error(
                        message = 'Сообщение об ошибке 2',
                        last = True, out = True
                    )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-14 16:25:11] Сообщение об ошибке 1

                    Файл: /var/folders/gw/w3k5kxtx0s3_nqdqw94zr8yh0000gn/T/ipykernel_20011/333478077.py
                    Линия: 5
                    Метод: <cell line: 5>
                    Тип ошибки: Exception

                [2022-10-14 16:25:11] Сообщение об ошибке 2

                    Файл: /var/folders/gw/w3k5kxtx0s3_nqdqw94zr8yh0000gn/T/ipykernel_20011/333478077.py
                    Линия: 16
                    Метод: <cell line: 16>
                    Тип ошибки: Exception

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core()

                try: raise Exception
                except:
                    core._other_error(
                        message = '',
                        last = False, out = True
                    )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-14 16:25:11] Неверные типы или значения аргументов в "Core._other_error" ...
        """

        if self.is_notebook_ is True:
            try:
                # Проверка аргументов
                if type(message) is not str or not message: raise TypeError
            except TypeError: self._inv_args(__class__.__name__, self._other_error.__name__, out = out); return None

            trac = self._traceback() # Трассировка исключений

            b = '**' if self.bold_text_ is True else ''
            cr = self.color_simple_

            # Отображение сообщения
            self._notebook_display_markdown(('{}' * 8).format(
                f'<span style=\"color:{cr}\">{b}[</span><span style=\"color:{self.color_err_}\">',
                datetime.now().strftime(self._format_time),
                f'</span><span style=\"color:{cr}\">]</span> ',
                f'<span style=\"color:{cr}\">{message}</span>{b}',
                f'<p><span style=\"color:{cr}\">{self.__tab}{self._trac_file}: <u>{trac["filename"]}</u></span>',
                f'<br /><span style=\"color:{cr}\">{self.__tab}{self._trac_line}: <u>{trac["lineno"]}</u></span>',
                f'<br /><span style=\"color:{cr}\">{self.__tab}{self._trac_method}: <u>{trac["name"]}</u></span>',
                f'<br /><span style=\"color:{cr}\">{self.__tab}{self._trac_type_err}: <u>{trac["type"]}</u></span></p>'
            ), last, out)

    def _error_wrapper(self, message: str) -> str:
        """Обернутое сообщение об ошибке

        .. note::
            protected (защищенный метод)

        Args:
            message (str): Сообщение

        Returns:
            str: Обернутое сообщение об ошибке

        .. dropdown:: Пример

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                print(core._error_wrapper(
                    'Обернутое сообщение об ошибке 1'
                ))

                core.color_err_ = '#FF4545'
                print(core._error_wrapper(
                    'Обернутое сообщение об ошибке 2'
                ))

            .. output-cell::
                :execution-count: 1
                :linenos:

                <span style="color:#FF0000">Обернутое сообщение об ошибке 1</span>
                <span style="color:#FF4545">Обернутое сообщение об ошибке 2</span>
        """

        if self.is_notebook_ is True:
            return ('{}' * 3).format(f'<span style=\"color:{self.color_err_}\">', message, f'</span>')

    def _stat_acoustic_features(self, last: bool = False, out: bool = True,
                                **kwargs: Union[int, Tuple[int], tf.TensorShape]) -> None:
        """Сообщение со статистикой извлеченных признаков из акустического сигнала

        .. note::
            protected (защищенный метод)

        Args:
            last (bool): Замена последнего сообщения
            out (bool): Отображение
            **kwargs (Union[int, Tuple[int], tf.TensorShape]): Дополнительные именованные аргументы

        Returns:
            None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core(
                    color_simple = '#FFF',
                    color_info = '#1776D2',
                    bold_text = True,
                )

                core._stat_acoustic_features(
                    last = False, out = True,
                    len_hc_features = 12,
                    len_melspectrogram_features = 12,
                    shape_hc_features = [196, 25],
                    shape_melspectrogram_features = [224, 224, 3],
                )

            .. output-cell::
                :execution-count: 1
                :linenos:
                :tab-width: 8

                [2022-10-14 17:59:20] Статистика извлеченных признаков из акустического сигнала:
                    Общее количество сегментов с:
                        1. экспертными признаками: 12
                        2. лог мел-спектрограммами: 12
                    Размерность матрицы экспертных признаков одного сегмента: 196 ✕ 25
                    Размерность тензора с лог мел-спектрограммами одного сегмента: 224 ✕ 224 ✕ 3

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core(
                    color_simple = '#FFF',
                    color_info = '#1776D2',
                    bold_text = True,
                )

                core._stat_acoustic_features(
                    last = False, out = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-14 17:59:21] Неверные типы или значения аргументов в "Core._stat_acoustic_features" ...
        """

        if self.is_notebook_ is True:
            tab = self.__tab

            b = '**' if self.bold_text_ is True else ''
            cr = self.color_simple_

            try:
                # Отображение сообщения
                self._notebook_display_markdown(self._get_acoustic_feature_stat.format(
                    f'<span style=\"color:{cr}\">{b}[</span><span style=\"color:{self.color_info_}\">',
                    datetime.now().strftime(self._format_time),
                    f'</span><span style=\"color:{cr}\">]</span> ', f'<span style=\"color:{cr}\">',
                    f'</span>{b}<br /><span style=\"color:{cr}\">{tab}',
                    f'</span><br /><span style=\"color:{cr}\">{tab * 2}', f'<u>{kwargs["len_hc_features"]}</u></span>',
                    f'<br /><span style=\"color:{cr}\">{tab * 2}',
                    f'<u>{kwargs["len_melspectrogram_features"]}</u></span>',
                    f'<br /><span style=\"color:{cr}\">{tab}',
                    f'<u>{kwargs["shape_hc_features"][0]}</u>', f'<u>{kwargs["shape_hc_features"][1]}</u></span>',
                    f'<br /><span style=\"color:{cr}\">{tab}',
                    f' <u>{kwargs["shape_melspectrogram_features"][0]}</u>',
                    f'<u>{kwargs["shape_melspectrogram_features"][1]}</u>',
                    f'<u>{kwargs["shape_melspectrogram_features"][2]}</u></span>',
                ), last, out)
            except KeyError:
                self._inv_args(__class__.__name__, self._stat_acoustic_features.__name__, out = out); return None

    def _stat_visual_features(self, last: bool = False, out: bool = True,
                              **kwargs: Union[int, Tuple[int], tf.TensorShape]) -> None:
        """Сообщение c статистикой извлеченных признаков из визуального сигнала

        .. note::
            protected (защищенный метод)

        Args:
            last (bool): Замена последнего сообщения
            out (bool): Отображение
            **kwargs (Union[int, Tuple[int], tf.TensorShape]): Дополнительные именованные аргументы

        Returns:
            None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core(
                    color_simple = '#FFF',
                    color_info = '#1776D2',
                    bold_text = True,
                )

                core._stat_visual_features(
                    last = False, out = True,
                    len_hc_features = 23,
                    len_nn_features = 23,
                    shape_hc_features = [10, 115],
                    shape_nn_features = [10, 512],
                    fps_before = 30,
                    fps_after = 10
                )

            .. output-cell::
                :execution-count: 1
                :linenos:
                :tab-width: 8

                [2022-11-03 16:18:40] Статистика извлеченных признаков из визуального сигнала:
                    Общее количество сегментов с:
                        1. экспертными признаками: 23
                        2. нейросетевыми признаками: 23
                    Размерность матрицы экспертных признаков одного сегмента: 10 ✕ 115
                    Размерность тензора с нейросетевыми признаками одного сегмента: 10 ✕ 512
                    Понижение кадровой частоты: с 30 до 10

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core(
                    color_simple = '#FFF',
                    color_info = '#1776D2',
                    bold_text = True,
                )

                core._stat_visual_features(
                    last = False, out = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-11-03 16:19:35] Неверные типы или значения аргументов в "Core._stat_visual_features" ...
        """

        if self.is_notebook_ is True:
            tab = self.__tab

            b = '**' if self.bold_text_ is True else ''
            cr = self.color_simple_

            try:
                # Отображение сообщения
                self._notebook_display_markdown(self._get_visual_feature_stat.format(
                    f'<span style=\"color:{cr}\">{b}[</span><span style=\"color:{self.color_info_}\">',
                    datetime.now().strftime(self._format_time),
                    f'</span><span style=\"color:{cr}\">]</span> ', f'<span style=\"color:{cr}\">',
                    f'</span>{b}<br /><span style=\"color:{cr}\">{tab}',
                    f'</span><br /><span style=\"color:{cr}\">{tab * 2}', f'<u>{kwargs["len_hc_features"]}</u></span>',
                    f'<br /><span style=\"color:{cr}\">{tab * 2}',
                    f'<u>{kwargs["len_nn_features"]}</u></span>',
                    f'<br /><span style=\"color:{cr}\">{tab}',
                    f'<u>{kwargs["shape_hc_features"][0]}</u>', f'<u>{kwargs["shape_hc_features"][1]}</u></span>',
                    f'<br /><span style=\"color:{cr}\">{tab}',
                    f' <u>{kwargs["shape_nn_features"][0]}</u>', f'<u>{kwargs["shape_nn_features"][1]}</u></span>',
                    f'<br /><span style=\"color:{cr}\">{tab}',
                    f' <u>{kwargs["fps_before"]}</u>', f'<u>{kwargs["fps_after"]}</u></span>',
                ), last, out)
            except KeyError:
                self._inv_args(__class__.__name__, self._stat_visual_features.__name__, out = out); return None

    def _r_start(self) -> None:
        """Начало отсчета времени выполнения

        .. note::
            protected (защищенный метод)

        .. hint:: Работает в связке с ``_r_end()``

        Returns:
            None

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._r_start()
                for cnt in range(0, 10000000): res = cnt * 2
                core._r_end()

            .. output-cell::
                :execution-count: 1
                :linenos:

                --- Время выполнения: 0.819 сек. ---

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                for cnt in range(0, 10000000): res = cnt * 2
                core._r_end()

            .. output-cell::
                :execution-count: 1
                :linenos:

                --- Время выполнения: 1665756222.704 сек. ---
        """

        self._runtime = self._start_time = -1 # Сброс значений

        self._start_time = time.time() # Отсчет времени выполнения

    def _r_end(self, last: bool = False, out: bool = True) -> None:
        """Конец отсчета времени выполнения

        .. note::
            protected (защищенный метод)

        .. hint:: Работает в связке с ``_r_start()``

        Args:
            last (bool): Замена последнего сообщения
            out (bool): Отображение

        Returns:
            None

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._r_start()
                for cnt in range(0, 10000000): res = cnt * 2
                core._r_end()

            .. output-cell::
                :execution-count: 1
                :linenos:

                --- Время выполнения: 0.819 сек. ---

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                for cnt in range(0, 10000000): res = cnt * 2
                core._r_end()

            .. output-cell::
                :execution-count: 1
                :linenos:

                --- Время выполнения: 1665756222.704 сек. ---
        """

        self._runtime = round(time.time() - self._start_time, 3) # Время выполнения

        t = '--- {}: {} {} ---'.format(self.text_runtime_, self._runtime, self._sec)

        if self.is_notebook_ is True:
            b = '**' if self.bold_text_ is True else ''

            # Отображение сообщения
            self._notebook_display_markdown(
                '{}'.format(f'<span style=\"color:{self.color_simple_}\">{b}{t}{b}</span>'), last, out)

    def _progressbar(
            self, message: str, progress: str, clear_out: bool = True, last: bool = False, out: bool = True
    ) -> None:
        """Индикатор выполнения

        .. note::
            protected (защищенный метод)

        Args:
            message (str): Сообщение
            progress (str): Индикатор выполнения
            clear_out (bool): Очистка области вывода
            last (bool): Замена последнего сообщения
            out (bool): Отображение

        Returns:
            None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core()

                for cnt in range(1, 4):
                    core._progressbar(
                        message = 'Цикл действий',
                        progress = 'Итерация ' + str(cnt),
                        clear_out = False,
                        last = False, out = True
                    )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-14 16:52:20] Цикл действий

                    Итерация 1

                [2022-10-14 16:52:20] Цикл действий

                    Итерация 2

                [2022-10-14 16:52:20] Цикл действий

                    Итерация 3

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core()

                for cnt in range(1, 4):
                    core._progressbar(
                        message = 'Цикл действий',
                        progress = 'Итерация ' + str(cnt),
                        clear_out = True,
                        last = True, out = True
                    )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-14 16:52:20] Цикл действий

                    Итерация 3

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core()

                for cnt in range(1, 4):
                    core._progressbar(
                        message = 1,
                        progress = 2,
                        clear_out = True,
                        last = False, out = True
                    )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-14 16:52:38] Неверные типы или значения аргументов в "Core._progressbar" ...
        """

        if self.is_notebook_ is True:
            if clear_out is True: clear_output(True)

            try:
                # Проверка аргументов
                if type(message) is not str or not message or type(progress) is not str or not progress: raise TypeError
            except TypeError: self._inv_args(__class__.__name__, self._progressbar.__name__, out = out); return None

            b = '**' if self.bold_text is True else ''
            tab = self.__tab

            # Отображение сообщения
            self._notebook_display_markdown(('{}' * 5).format(
                f'<span style=\"color:{self.color_simple_}\">{b}[</span><span style=\"color:{self.color_info_}\">',
                datetime.now().strftime(self._format_time),
                f'</span><span style=\"color:{self.color_simple_}\">]</span> ',
                f'<span style=\"color:{self.color_simple_}\">{message}</span>{b}',
                f'<p><span style=\"color:{self.color_simple_}\">{tab}{progress}</span></p>'
            ), last, out)

    def _progressbar_union_predictions(
        self, message: str, item: int, info: str, len_paths: int, clear_out: bool = True, last: bool = False,
            out: bool = True
    ) -> None:
        """Индикатор выполнения получения прогнозов по аудио

        .. note::
            private (приватный метод)

        Args:
            message (str): Сообщение
            item (int): Номер видеофайла
            info (str): Локальный путь
            len_paths (int): Количество видеофайлов
            clear_out (bool): Очистка области вывода
            last (bool): Замена последнего сообщения
            out (bool): Отображение

        Returns:
            None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core()

                l = range(1, 4, 1)

                for progress in l:
                    core._progressbar_union_predictions(
                        message = 'Цикл действий',
                        item = progress,
                        info = 'Путь к файлу',
                        len_paths = len(l),
                        clear_out = False,
                        last = False, out = True
                    )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-20 16:51:49] Цикл действий

                    1 из 3 (33.33%) ... Путь к файлу ...

                [2022-10-20 16:51:49] Цикл действий

                    2 из 3 (66.67%) ... Путь к файлу ...

                [2022-10-20 16:51:49] Цикл действий

                    3 из 3 (100.0%) ... Путь к файлу ...

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core()

                l = range(1, 4, 1)

                for progress in l:
                    core._progressbar_union_predictions(
                        message = 'Цикл действий',
                        item = progress,
                        info = 'Путь к файлу',
                        len_paths = len(l),
                        clear_out = True,
                        last = True, out = True
                    )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-20 16:51:55] Цикл действий

                    3 из 3 (100.0%) ... Путь к файлу ...

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core()

                l = range(1, 4, 1)

                for progress in l:
                    core._progressbar_union_predictions(
                        message = 1,
                        item = progress,
                        info = 'Путь к файлу',
                        len_paths = len(l),
                        clear_out = True,
                        last = False, out = True
                    )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-20 16:55:15] Неверные типы или значения аргументов в "Audio._progressbar_union_predictions" ...
        """

        if self.is_notebook_ is True:
            if clear_out is False and last is True: clear_out, last = last, clear_out
            elif clear_out is False and last is False: clear_out = True
            if clear_out is True: clear_output(True)

            try:
                # Проверка аргументов
                if (type(message) is not str or not message or type(item) is not int or type(len_paths) is not int
                    or type(info) is not str or not info): raise TypeError
            except TypeError:
                self._inv_args(__class__.__name__, self._progressbar_union_predictions.__name__, out = out)
                return None

            self._progressbar(
                message,
                self._curr_progress_union_predictions.format(
                    item, len_paths, round(item * 100 / len_paths, 2), info
                ),
                clear_out = clear_out,
                last = last, out = False
            )
            if out: self.show_notebook_history_output()

    def _clear_notebook_history_output(self) -> None:
        """Очистка истории вывода сообщений в ячейке Jupyter

        .. note::
            protected (защищенный метод)

        Returns:
            None

        .. dropdown:: Пример

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._add_notebook_history_output(
                    message = 'Сообщение 1', last = False
                )
                core._add_notebook_history_output(
                    message = 'Сообщение 2', last = False
                )

                core._clear_notebook_history_output()

                core.show_notebook_history_output()

            .. output-cell::
                :execution-count: 1
                :linenos:


        """

        self._notebook_history_output.clear() # Очистка истории вывода сообщений в ячейке Jupyter

    def _add_notebook_history_output(self, message: str, last: bool = False) -> None:
        """Добавление истории вывода сообщений в ячейке Jupyter

        .. note::
            protected (защищенный метод)

        Args:
            message (str): Сообщение
            last (bool): Замена последнего сообщения

        Returns:
            None

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._add_notebook_history_output(
                    message = 'Сообщение 1', last = False
                )
                core._add_notebook_history_output(
                    message = 'Сообщение 2', last = False
                )
                core._add_notebook_history_output(
                    message = 'Замена последнего сообщения', last = True
                )

                core.show_notebook_history_output()

            .. output-cell::
                :execution-count: 1
                :linenos:

                Сообщение 1
                Замена последнего сообщения

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core()

                for message, last in zip(
                    [
                        'Сообщение 1',
                        'Сообщение 2',
                        'Замена последнего сообщения'
                    ],
                    [False, False, True]
                ):
                    core._add_notebook_history_output(
                        message = message, last = last
                    )

                core.show_notebook_history_output()

            .. output-cell::
                :execution-count: 2
                :linenos:

                Сообщение 1
                Замена последнего сообщения
        """

        if last is True:
            try: self._notebook_history_output[-1] = message
            except Exception: pass
            else: return None

        self._notebook_history_output.append(message)

    def _del_last_el_notebook_history_output(self) -> None:
        """Удаление последнего сообщения из истории вывода сообщений в ячейке Jupyter

        .. note::
            protected (защищенный метод)

        Returns:
            None

        .. dropdown:: Пример

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._add_notebook_history_output(
                    message = 'Сообщение 1', last = False
                )
                core._add_notebook_history_output(
                    message = 'Сообщение 2', last = False
                )

                core._del_last_el_notebook_history_output()

                core.show_notebook_history_output()

            .. output-cell::
                :execution-count: 1
                :linenos:

                Сообщение 1
        """

        try: last_el = self._notebook_history_output.pop()
        except Exception: pass

    def _add_last_el_notebook_history_output(self, message: str) -> None:
        """Добавление текста к последнему сообщению из истории вывода сообщений в ячейке Jupyter

        .. note::
            protected (защищенный метод)

        Args:
            message (str): Сообщение

        Returns:
            None

        .. dropdown:: Пример

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._add_last_el_notebook_history_output(message = '...')

                core._add_notebook_history_output(
                    message = 'Сообщение 1', last = False
                )
                core._add_last_el_notebook_history_output(message = '...')

                core.show_notebook_history_output()

            .. output-cell::
                :execution-count: 1
                :linenos:

                ...
                Сообщение 1 ...
        """

        try: self._notebook_history_output[-1] += ' ' + message
        except Exception: self._add_notebook_history_output(message = message, last = False)

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __is_notebook() -> bool:
        """Определение запуска библиотеки в Jupyter или аналогах

        .. note::
            private (приватный метод)

        Returns:
            bool: **True** если библиотека запущена в Jupyter или аналогах, в обратном случае **False**

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core._Core__is_notebook()

            .. output-cell::
                :execution-count: 1
                :linenos:

                True

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.core import Core

                Core._Core__is_notebook()

            .. output-cell::
                :execution-count: 2
                :linenos:

                True
        """

        try:
            # Определение режима запуска библиотеки
            shell = get_ipython().__class__.__name__
        except (NameError, Exception): return False # Запуск в Python
        else:
            if shell == 'ZMQInteractiveShell' or shell == 'Shell': return True
            elif shell == 'TerminalInteractiveShell': return False
            else: return False

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------
    def _get_paths(self, path: Iterable, depth: int = 1, out: bool = True) -> Union[List[str], bool]:
        """Получение директорий где хранятся данные

        .. note::
            protected (защищенный метод)

        Args:
            path (Iterable): Директория набора данных
            depth (int): Глубина иерархии для извлечения классов
            out (bool): Отображение

        Returns:
            Union[List[str], bool]: **False** если проверка аргументов не удалась или список с директориями

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                core = Core()
                core._get_paths(
                    path = '/Users/dl/GitHub/oceanai/oceanai/dataset',
                    depth = 1, out = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [
                    '/Users/dl/GitHub/oceanai/oceanai/dataset/test80_01',
                    '/Users/dl/GitHub/oceanai/oceanai/dataset/1',
                    '/Users/dl/GitHub/oceanai/oceanai/dataset/test80_17'
                ]

            :bdg-danger:`Ошибки` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core._get_paths(
                    path = '',
                    depth = 1, out = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-12 16:36:16] Неверные типы или значения аргументов в "Core._get_paths" ...
                False

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core._get_paths(
                    path = '/Users/dl/GitHub/oceanai/oceanai/folder',
                    depth = 1, out = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-13 18:37:11] Ой! Что-то пошло не так ... директория "/Users/dl/GitHub/oceanai/oceanai/folder" не найдена ...

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/core/core.py
                    Линия: 1023
                    Метод: _get_paths
                    Тип ошибки: FileNotFoundError

                False
        """

        try:
            # Проверка аргументов
            if (not isinstance(path, Iterable) or not path or type(depth) is not int or depth < 1
                    or type(out) is not bool):
                raise TypeError
        except TypeError: self._inv_args(__class__.__name__, self._get_paths.__name__, out = out); return False
        else:
            if type(path) is not list: path = [path]

            new_path = [] # Список с директориями

            # Проход по всем директориям набора данных
            for curr_path in path:
                try: scandir = os.scandir(os.path.normpath(str(curr_path)))
                except FileNotFoundError:
                    self._other_error(self._folder_not_found.format(self._info_wrapper(str(curr_path))), out = out)
                    return False
                except Exception: self._other_error(self._unknown_err, out = out); return False
                else:
                    for f in scandir:
                        if f.is_dir() and not f.name.startswith('.'):
                            ignore = False # По умолчанию не игнорировать директорию
                            if depth == 1:
                                for curr_dir in self.ignore_dirs_:
                                    if type(curr_dir) is not str: continue
                                    # Игнорировать директорию
                                    if re.search('^' + curr_dir, f.name) is not None: ignore = True

                            if ignore is False: new_path.append(f.path)

            # Рекурсивный переход на следующий уровень иерархии
            if depth != 1 and len(new_path) > 0: return self._get_paths(new_path, depth - 1)

            return new_path # Список с директориями

    def _append_to_list_of_files(self, path: str, preds: List[Optional[float]], out: bool = True) -> bool:
        """Добавление значений в словарь для DataFrame c данными

        .. note::
            protected (защищенный метод)

        Args:
            path (str): Путь к файлу
            preds (List[Optional[float]]): Предсказания персональных качеств
            out (bool): Отображение

        Returns:
            bool: **True** если значения в словарь для DataFrame были добавлены, в обратном случае **False**

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core.keys_dataset_ = ['P', 'O', 'C', 'E', 'A', 'N']

                core._append_to_list_of_files(
                    path = './6V807Mf_gHM.003.mp4',
                    preds = [0.5, 0.6, 0.2, 0.1, 0.8],
                    out = True
                )

                core._append_to_list_of_files(
                    path = './6V807Mf_gHM.004.mp4',
                    preds = [0.4, 0.5, 0.1, 0, 0.7],
                    out = True
                )

                core.dict_of_files_

            .. output-cell::
                :execution-count: 1
                :linenos:

                {
                    'P': ['./6V807Mf_gHM.003.mp4', './6V807Mf_gHM.004.mp4'],
                    'O': [0.5, 0.4],
                    'C': [0.6, 0.5],
                    'E': [0.2, 0.1],
                    'A': [0.1, 0],
                    'N': [0.8, 0.7]
                }

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core.keys_dataset_ = ['P', 'O', 'C', 'E', 'A', 'N']

                core._append_to_list_of_files(
                    path = './6V807Mf_gHM.003.mp4',
                    preds = [0.5, 0.6, 0.2, 0.1, 0.8],
                    out = True
                )

                core.keys_dataset_ = ['P2', 'O2', 'C2', 'E2', 'A2', 'N2']

                core._append_to_list_of_files(
                    path = './6V807Mf_gHM.004.mp4',
                    preds = [0.4, 0.5, 0.1, 0, 0.7],
                    out = True
                )

                core.dict_of_files_

            :bdg-light:`-- 2 --`

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-13 18:22:40] Ой! Что-то пошло не так ... смотрите настройки ядра и цепочку выполнения действий ...

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/core/core.py
                    Линия: 1105
                    Метод: _append_to_list_of_files
                    Тип ошибки: KeyError

                {
                    'P': ['./6V807Mf_gHM.003.mp4'],
                    'O': [0.5],
                    'C': [0.6],
                    'E': [0.2],
                    'A': [0.1],
                    'N': [0.8]
                }
        """

        try:
            if len(self._dict_of_files.keys()) != len(self.keys_dataset_):
                # Словарь для DataFrame набора данных с данными
                self._dict_of_files = dict(zip(self.keys_dataset_, [[] for _ in range(0, len(self.keys_dataset_))]))

            self._dict_of_files[self.keys_dataset_[0]].append(path)

            for i in range(len(preds)): self._dict_of_files[self.keys_dataset_[i + 1]].append(preds[i])
        except (IndexError, KeyError): self._other_error(self._som_ww, out = out); return False
        except Exception: self._other_error(self._unknown_err, out = out); return False
        else: return True

    def _append_to_list_of_accuracy(self, preds: List[Optional[float]], out: bool = True) -> bool:
        """Добавление значений в словарь для DataFrame с результатами вычисления точности

        .. note::
            protected (защищенный метод)

        Args:
            preds (List[Optional[float]]): Предсказания персональных качеств
            out (bool): Отображение

        Returns:
            bool: **True** если значения в словарь для DataFrame были добавлены, в обратном случае **False**

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core.keys_dataset_ = ['O', 'C', 'E', 'A', 'N']

                core._append_to_list_of_accuracy(
                    preds = [0.5, 0.6, 0.2, 0.1, 0.8],
                    out = True
                )

                core._append_to_list_of_accuracy(
                    preds = [0.4, 0.5, 0.1, 0, 0.7],
                    out = True
                )

                core.dict_of_accuracy_

            .. output-cell::
                :execution-count: 1
                :linenos:

                {
                    'O': [0.5, 0.4],
                    'C': [0.6, 0.5],
                    'E': [0.2, 0.1],
                    'A': [0.1, 0],
                    'N': [0.8, 0.7]
                }

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core.keys_dataset_ = ['O', 'C', 'E', 'A', 'N']

                core._append_to_list_of_accuracy(
                    preds = [0.5, 0.6, 0.2, 0.1, 0.8],
                    out = True
                )

                core.keys_dataset_ = ['O2', 'C2', 'E2', 'A2', 'N2']

                core._append_to_list_of_accuracy(
                    preds = [0.4, 0.5, 0.1, 0, 0.7],
                    out = True
                )

                core.dict_of_accuracy_

            :bdg-light:`-- 2 --`

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-12-03 23:08:15] Ой! Что-то пошло не так ... смотрите настройки ядра и цепочку выполнения действий ...

                    Файл: /Users/dl/GitHub/OCEANAI.private/oceanai/modules/core/core.py
                    Линия: 2669
                    Метод: _append_to_list_of_accuracy
                    Тип ошибки: KeyError

                {
                    'O': [0.5, 0.4],
                    'C': [0.6, 0.5],
                    'E': [0.2, 0.1],
                    'A': [0.1, 0],
                    'N': [0.8, 0.7]
                }
        """

        try:
            if len(self._dict_of_accuracy.keys()) != len(self.keys_dataset_[1:]):
                # Словарь для DataFrame набора данных с результатами вычисления точности
                self._dict_of_accuracy = dict(
                    zip(self.keys_dataset_[1:], [[] for _ in range(0, len(self.keys_dataset_[1:]))])
                )

            for i in range(len(preds)): self._dict_of_accuracy[self.keys_dataset_[i + 1]].append(preds[i])
        except (IndexError, KeyError): self._other_error(self._som_ww, out = out); return False
        except Exception: self._other_error(self._unknown_err, out = out); return False
        else: return True

    def _create_folder_for_logs(self, out: bool = True):
        """Создание директории для сохранения LOG файлов

        .. note::
            protected (защищенный метод)

        Args:
            out (bool): Отображение

        Returns:
            bool: **True** если директория создана или существует, в обратном случае **False**

        .. dropdown:: Пример
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core.path_to_logs_ = './logs'

                core._create_folder_for_logs(out = True)

            .. output-cell::
                :execution-count: 1
                :linenos:

                true
        """

        if type(out) is not bool: out = True

        try:
            if not os.path.exists(self.path_to_logs_): os.makedirs(self.path_to_logs_)
        except (FileNotFoundError, TypeError): self._other_error(self._som_ww, out = out); return False
        except Exception: self._other_error(self._unknown_err, out = out); return False
        else: return True

    def _save_logs(self, df: pd.DataFrame, name: str, out: bool = True) -> bool:
        """Сохранение LOG файла

        .. note::
            protected (защищенный метод)

        Args:
            df (pd.DataFrame):  DataFrame который будет сохранен в LOG файл
            name (str): Имя LOG файла
            out (bool): Отображение

        Returns:
            bool: **True** если LOG файл сохранен, в обратном случае **False**

        .. dropdown:: Пример
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                import pandas as pd
                from oceanai.modules.core.core import Core

                df = pd.DataFrame.from_dict(
                    data = {'Test': [1, 2, 3]}
                )

                core = Core()

                core.path_to_logs_ = './logs'

                core._save_logs(
                    df = df, name = 'test', out = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                True
        """

        try:
            # Проверка аргументов
            if type(df) is not pd.DataFrame or type(name) is not str or not name or type(out) is not bool:
                raise TypeError
        except TypeError: self._inv_args(__class__.__name__, self._save_logs.__name__, out = out); return False
        else:
            # Создание директории для сохранения LOG файлов
            if self._create_folder_for_logs() is True:
                # Сохранение LOG файла
                try:
                    df.to_csv(os.path.join(self.path_to_logs_, name + self._ext_for_logs))
                except urllib.error.HTTPError as e:
                    self._other_error(self._url_error_log.format(self._url_error_code_log.format(
                        self._error_wrapper(str(e.code)))), out = out)
                except urllib.error.URLError: self._other_error(self._url_error_log.format(''), out = out)
                except Exception: self._other_error(self._unknown_err, out = out); return False
                else: return True

            return False

    def _round_math(self, val: Union[int, float], out: bool = True) -> Union[int, bool]:
        """Округление чисел по математическому закону

        .. note::
            protected (защищенный метод)

        Args:
            val (Union[int, float]): Число для округления
            out (bool): Отображение

        Returns:
             Union[int, bool]: Округленное число если ошибок не выявлено, в обратном случае **False**

        .. dropdown:: Пример
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._round_math(4.5)

            .. output-cell::
                :execution-count: 1
                :linenos:

                5

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()

                core._round_math(-2.5)

            .. output-cell::
                :execution-count: 1
                :linenos:

                -3

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.core.core import Core

                core = Core()

                core._round_math('')

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-11-03 15:52:30] Неверные типы или значения аргументов в "Core._round_math" ...

                False
        """

        try:
            # Проверка аргументов
            if ((type(val) is not int and type(val) is not float and type(val) is not np.float64)
                or type(out) is not bool): raise TypeError
        except TypeError: self._inv_args(__class__.__name__, self._round_math.__name__, out = out); return False
        else:
            modf = math.modf(val)

            if modf[0] >= 0.5: res = modf[1] + 1
            else:
                if modf[0] <= -0.5: res = modf[1] - 1
                else: res = math.ceil(modf[1])

            return int(res)

    def _candidate_ranking(
        self, weigths_openness: int = 0, weigths_conscientiousness: int = 0, weigths_extraversion: int = 0,
        weigths_agreeableness: int = 0, weigths_neuroticism: int = 0, out: bool = True
    ) -> pd.DataFrame:
        """Ранжирование кандидатов

        .. note::
            protected (защищенный метод)

        Args:
            weigths_openness (int): Вес для ранжирования персонального качества (открытость опыту)
            weigths_conscientiousness (int): Вес для ранжирования персонального качества (добросовестность)
            weigths_extraversion (int): Вес для ранжирования персонального качества (экстраверсия)
            weigths_agreeableness (int): Вес для ранжирования персонального качества (доброжелательность)
            weigths_neuroticism (int): Вес для ранжирования персонального качества (нейротизм)
            out (bool): Отображение

        Returns:
             pd.DataFrame: **DataFrame** c ранжированными данными
        """

        # Сброс
        self._df_files_ranking = pd.DataFrame() # Пустой DataFrame с ранжированными данными

        try:
            # Проверка аргументов
            if (type(weigths_openness) is not int or not (0 <= weigths_openness <= 100)
                or type(weigths_conscientiousness) is not int or not (0 <= weigths_conscientiousness <= 100)
                or type(weigths_extraversion) is not int or not (0 <= weigths_extraversion <= 100)
                or type(weigths_agreeableness) is not int or not (0 <= weigths_agreeableness <= 100)
                or type(weigths_neuroticism) is not int or not (0 <= weigths_neuroticism <= 100)
                or type(out) is not bool): raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self._candidate_ranking.__name__, out = out)
            return self._df_files_ranking
        else:
            try:
                if sum([
                    weigths_openness, weigths_conscientiousness, weigths_extraversion,
                    weigths_agreeableness, weigths_neuroticism
                ]) != 100: raise TypeError
            except TypeError: self._other_error(self._sum_ranking_exceeded, out = out); return self._df_files_ranking
            except Exception: self._other_error(self._unknown_err, out = out); return self._df_files_ranking
            else:
                try:
                    if len(self._df_files) == 0: raise TypeError

                except TypeError: self._other_error(self._dataframe_empty, out = out); return self._df_files_ranking
                except Exception: self._other_error(self._unknown_err, out = out); return self._df_files_ranking
                else:
                    try:
                        df_files = self._df_files[self.keys_dataset_[1:]]
                        traits_sum = np.sum(df_files.values * [
                            weigths_openness, weigths_conscientiousness, weigths_extraversion,
                            weigths_agreeableness, weigths_neuroticism
                        ], axis = 1)
                        self._df_files_ranking = self._df_files
                        self._df_files_ranking[self._keys_score] = traits_sum
                        self._df_files_ranking = self._df_files_ranking.sort_values(
                            by = self._keys_score, ascending = False
                        )
                    except Exception: self._other_error(self._unknown_err, out = out); return self._df_files_ranking
                    else: return self._df_files_ranking

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def show_notebook_history_output(self) -> None:
        """Отображение истории вывода сообщений в ячейке Jupyter

        Returns:
            None

        .. dropdown:: Пример

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core._info(
                    message = 'Информационное сообщение',
                    last = False, out = False
                )

                core.show_notebook_history_output()

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-15 18:27:46] Информационное сообщение
        """

        if self.is_notebook_ is True and len(self._notebook_history_output) > 0:
            # Отображение
            for e in self._notebook_history_output: display(e if isinstance(e, pd.DataFrame) else Markdown(e))

    def libs_vers(self, out: bool = True, runtime: bool = True, run: bool = True) -> None:
        """Получение и отображение версий установленных библиотек

        Args:
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core.libs_vers(out = True, runtime = True, run = True)

            .. output-cell::
                :execution-count: 1
                :linenos:

                |----|--------------|---------|
                |    | Package      | Version |
                |----|--------------|---------|
                | 1  | TensorFlow   | 2.11.0  |
                | 2  | Keras        | 2.11.0  |
                | 3  | OpenCV       | 4.6.0   |
                | 4  | MediaPipe    | 0.9.0   |
                | 5  | NumPy        | 1.23.5  |
                | 6  | SciPy        | 1.9.3   |
                | 7  | Pandas       | 1.5.2   |
                | 8  | Scikit-learn | 1.1.3   |
                | 9  | OpenSmile    | 2.4.1   |
                | 10 | Librosa      | 0.9.2   |
                | 11 | AudioRead    | 3.0.0   |
                | 12 | IPython      | 8.7.0   |
                | 13 | PyMediaInfo  | 6.0.1   |
                | 14 | Requests     | 2.28.1  |
                | 15 | JupyterLab   | 3.5.0   |
                |----|--------------|---------|
                --- Время выполнения: 0.005 сек. ---

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core.libs_vers(out = True, runtime = True, run = False)

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-15 18:17:27] Выполнение заблокировано пользователем ...

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.core import Core

                core = Core()
                core.libs_vers(out = True, runtime = True, run = 1)

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-15 18:18:51] Неверные типы или значения аргументов в "Core.libs_vers" ...
        """

        self._clear_notebook_history_output() # Очистка истории вывода сообщений в ячейке Jupyter

        # Сброс
        self._df_pkgs = pd.DataFrame() # Пустой DataFrame

        if type(out) is not bool: out = True

        try:
            # Проверка аргументов
            if type(runtime) is not bool or type(run) is not bool: raise TypeError
        except TypeError: self._inv_args(__class__.__name__, self.libs_vers.__name__, out = out)
        else:
            # Блокировка выполнения
            if run is False: self._error(self._lock_user, out = out); return None

            if runtime: self._r_start()

            pkgs = {
                'Package': [
                    'TensorFlow', 'Keras', 'OpenCV', 'MediaPipe', 'NumPy', 'SciPy', 'Pandas', 'Scikit-learn',
                    'OpenSmile', 'Librosa', 'AudioRead', 'IPython', 'PyMediaInfo', 'Requests', 'JupyterLab'
                ],
                'Version': [i.__version__ for i in [
                    tf, keras, cv2, mp, np, scipy, pd, sklearn, opensmile, librosa, audioread, IPython, pymediainfo,
                    requests, jlab
                ]]
            }

            self._df_pkgs = pd.DataFrame(data = pkgs) # Версии используемых библиотек
            self._df_pkgs.index += 1

            # Отображение
            if self.is_notebook_ is True and out is True: display(self._df_pkgs)

            if runtime: self._r_end(out = out)