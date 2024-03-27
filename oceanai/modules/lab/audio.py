#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Аудио
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################

import warnings

# Подавление Warning
for warn in [UserWarning, FutureWarning]:
    warnings.filterwarnings("ignore", category=warn)

from dataclasses import dataclass  # Класс данных

import os  # Взаимодействие с файловой системой
import logging
import requests  # Отправка HTTP запросов
import numpy as np  # Научные вычисления
import pandas as pd  # Обработка и анализ данных
import opensmile  # Анализ, обработка и классификация звука
import librosa  # Обработка аудио
import audioread  # Декодирование звука
import math
import gradio

from urllib.parse import urlparse
from urllib.error import URLError
from pathlib import Path  # Работа с путями в файловой системе
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from datetime import datetime  # Работа со временем
import subprocess

from typing import Dict, List, Tuple, Union, Optional, Callable  # Типы данных

from IPython.display import clear_output

# Персональные
from oceanai.modules.lab.download import Download  # Загрузка файлов
from oceanai.modules.core.exceptions import IsSmallWindowSizeError

# Порог регистрации сообщений TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # Машинное обучение от Google
import keras

from tensorflow.keras.applications import VGG16

# ######################################################################################################################
# Настройки необходимых инструментов
# ######################################################################################################################
pd.set_option("display.max_columns", None)  # Максимальное количество отображаемых столбцов
pd.set_option("display.max_rows", None)  # Максимальное количество отображаемых строк


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class AudioMessages(Download):
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
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._audio_modality: str = self._(" (аудио модальность) ...")
        self._formation_audio_model_hc: str = self._formation_model_hc + self._audio_modality
        self._formation_audio_model_nn: str = self._formation_model_nn + self._audio_modality
        self._formation_audio_models_b5: str = self._formation_models_b5 + self._audio_modality

        self._load_audio_model_weights_hc: str = self._load_model_weights_hc + self._audio_modality
        self._load_audio_model_weights_nn: str = self._load_model_weights_nn + self._audio_modality
        self._load_audio_models_weights_b5: str = self._load_models_weights_b5 + self._audio_modality

        self._get_acoustic_feature_info: str = self._(
            "Извлечение признаков (экспертных и лог мел-спектрограмм) из " "акустического сигнала ..."
        )
        self._get_acoustic_feature_hc_error: str = self._oh + self._(
            "экспертные признаки из акустического сигнала не " "извлечены ..."
        )
        self._get_acoustic_feature_spec_error: str = self._oh + self._(
            "лог мел-спектрограммы из акустического сигнала " "не извлечены ..."
        )

        self._window_small_size_error: str = self._oh + self._(
            "указан слишком маленький размер ({}) окна сегмента " "сигнала ..."
        )

        self._model_audio_hc_not_formation: str = self._model_hc_not_formation + self._audio_modality
        self._model_audio_nn_not_formation: str = self._model_nn_not_formation + self._audio_modality
        self._models_audio_not_formation: str = self._models_not_formation + self._audio_modality

        self._concat_audio_pred_error: str = self._concat_pred_error + self._audio_modality
        self._norm_audio_pred_error: str = self._norm_pred_error + self._audio_modality


# ######################################################################################################################
# Аудио
# ######################################################################################################################
@dataclass
class Audio(AudioMessages):
    """Класс для обработки аудио

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
        super().__post_init__()  # Выполнение конструктора из суперкласса

        # Нейросетевая модель **tf.keras.Model** для получения оценок по экспертным признакам
        self._audio_model_hc: Optional[tf.keras.Model] = None
        # Нейросетевая модель **tf.keras.Model** для получения оценок по нейросетевым признакам
        self._audio_model_nn: Optional[tf.keras.Model] = None
        # Нейросетевые модели **tf.keras.Model** для получения результатов оценки персональных качеств
        self._audio_models_b5: Dict[str, Optional[tf.keras.Model]] = dict(
            zip(self._b5["en"], [None] * len(self._b5["en"]))
        )

        self._smile: opensmile.core.smile.Smile = self.__smile()  # Извлечение функций OpenSmile

        # ----------------------- Только для внутреннего использования внутри класса

        # Настройки для спектрограммы
        self.__pl: List[Union[int, str, bool, float, None]] = [
            2048,
            512,
            None,
            True,
            "reflect",
            2.0,
            128,
            "slaney",
            True,
            None,
        ]
        self.__len_paths: int = 0  # Количество искомых файлов
        self.__local_path: Union[Callable[[str], str], None] = None  # Локальный путь

        # Ключи для точности
        self.__df_accuracy_index: List[str] = ["MAE", "Accuracy"]
        self.__df_accuracy_index_name: str = "Metrics"
        self.__df_accuracy_mean: str = "Mean"

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def audio_model_hc_(self) -> Optional[tf.keras.Model]:
        """Получение нейросетевой модели **tf.keras.Model** для получения оценок по экспертным признакам

        Returns:
            Optional[tf.keras.Model]: Нейросетевая модель **tf.keras.Model** или None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.load_audio_model_hc(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

                audio.audio_model_hc_

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-17 13:54:35] Формирование нейросетевой архитектуры модели для получения оценок по экспертным признакам (аудио модальность) ...

                --- Время выполнения: 0.509 сек. ---

                <tf.keras.Model at 0x13dd600a0>

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.audio_model_hc_

            .. output-cell::
                :execution-count: 2
                :linenos:


        """

        return self._audio_model_hc

    @property
    def audio_model_nn_(self) -> Optional[tf.keras.Model]:
        """Получение нейросетевой модели **tf.keras.Model** для получения оценок по нейросетевым признакам

        Returns:
            Optional[tf.keras.Model]: Нейросетевая модель **tf.keras.Model** или None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.load_audio_model_nn(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

                audio.audio_model_nn_

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-17 13:58:29] Формирование нейросетевой архитектуры для получения оценок по нейросетевым признакам ...

                --- Время выполнения: 0.444 сек. ---

                <tf.keras.Model at 0x13db97760>

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.audio_model_nn_

            .. output-cell::
                :execution-count: 2
                :linenos:


        """

        return self._audio_model_nn

    @property
    def audio_models_b5_(self) -> Dict[str, Optional[tf.keras.Model]]:
        """Получение нейросетевых моделей **tf.keras.Model** для получения результатов оценки персональных качеств

        Returns:
            Dict: Словарь с нейросетевыми моделями **tf.keras.Model**

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.load_audio_models_b5(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

                audio.audio_models_b5_

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-19 15:45:35] Формирование нейросетевых архитектур моделей для получения результатов оценки
                персональных качеств (аудио модальность) ...

                --- Время выполнения: 0.07 сек. ---

                {
                    'openness': <tf.keras.Model at 0x1481e03a0>,
                    'conscientiousness': <tf.keras.Model at 0x147d13520>,
                    'extraversion': <tf.keras.Model at 0x1481edfa0>,
                    'agreeableness': <tf.keras.Model at 0x1481cfc40>,
                    'non_neuroticism': <tf.keras.Model at 0x1481cffd0>
                }

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.audio_models_b5_

            .. output-cell::
                :execution-count: 2
                :linenos:
                :tab-width: 8

                {
                    'openness': None,
                    'conscientiousness': None,
                    'extraversion': None,
                    'agreeableness': None,
                    'non_neuroticism': None
                }
        """

        return self._audio_models_b5

    @property
    def smile_(self) -> opensmile.core.smile.Smile:
        """Получение функций OpenSmile

        Returns:
            opensmile.core.smile.Smile: Извлеченные функции OpenSmile

        .. dropdown:: Пример
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()
                audio.smile_

            .. output-cell::
                :execution-count: 1
                :linenos:
                :tab-width: 8

                {
                    '$opensmile.core.smile.Smile': {
                        'feature_set': 'eGeMAPSv02',
                        'feature_level': 'LowLevelDescriptors',
                        'options': {},
                        'sampling_rate': None,
                        'channels': [0],
                        'mixdown': False,
                        'resample': False
                    }
                }
        """

        return self._smile

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)
    # ------------------------------------------------------------------------------------------------------------------

    def __load_model_weights(
        self,
        url: str,
        force_reload: bool = True,
        info_text: str = "",
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> bool:
        """Загрузка весов нейросетевой модели

        .. note::
            private (приватный метод)

        Args:
            url (str): Полный путь к файлу с весами нейросетевой модели
            force_reload (bool): Принудительная загрузка файла с весами нейросетевой модели из сети
            info_text (str): Текст для информационного сообщения
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если веса нейросетевой модели загружены, в обратном случае **False**

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.path_to_save_ = './models'
                audio.chunk_size_ = 2000000

                audio._Audio__load_model_weights(
                    url = 'https://download.sberdisk.ru/download/file/400635799?token=MMRrak8fMsyzxLE&filename=weights_2022-05-05_11-27-55.h5',
                    force_reload = True,
                    info_text = 'Загрузка весов нейросетевой модели',
                    out = True, runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-17 12:21:48] Загрузка весов нейросетевой модели

                [2022-10-17 12:21:48] Загрузка файла "weights_2022-05-05_11-27-55.h5" (100.0%) ...

                --- Время выполнения: 0.439 сек. ---

                True

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.path_to_save_ = './models'
                audio.chunk_size_ = 2000000

                audio._Audio__load_model_weights(
                    url = './models/weights_2022-05-05_11-27-55.h5',
                    force_reload = True,
                    info_text = 'Загрузка весов нейросетевой модели',
                    out = True, runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-17 12:21:50] Загрузка весов нейросетевой модели

                --- Время выполнения: 0.002 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.path_to_save_ = './models'
                audio.chunk_size_ = 2000000

                audio._Audio__load_model_weights(
                    url = 'https://download.sberdisk.ru/download/file/400635799?token=MMRrak8fMsyzxLE&filename=weights_2022-05-05_11-27-55.h5',
                    force_reload = True, info_text = '',
                    out = True, runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-17 12:21:57] Неверные типы или значения аргументов в "Audio.__load_model_weights" ...

                False
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        try:
            # Проверка аргументов
            if (
                type(url) is not str
                or not url
                or type(force_reload) is not bool
                or type(info_text) is not str
                or not info_text
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__load_model_weights.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(info_text, last=False, out=out)

            sections = urlparse(url)  # Парсинг URL адреса

            try:
                # URL файл невалидный
                if sections.scheme == "":
                    raise requests.exceptions.InvalidURL
            except requests.exceptions.InvalidURL:
                url = os.path.normpath(url)

                try:
                    if os.path.isfile(url) is False:
                        raise FileNotFoundError  # Не файл
                except FileNotFoundError:
                    self._other_error(self._load_model_weights_error, out=out)
                    return False
                except Exception:
                    self._other_error(self._unknown_err, out=out)
                    return False
                else:
                    self._url_last_filename = url
                    return True
            else:
                try:
                    if force_reload is False:
                        clear_output(True)
                    # Загрузка файла из URL
                    res_download_file_from_url = self._download_file_from_url(
                        url=url, force_reload=force_reload, runtime=False, out=out, run=True
                    )
                except Exception:
                    self._other_error(self._unknown_err, out=out)
                    return False
                else:
                    # Файл загружен
                    if res_download_file_from_url != 200:
                        return False

                    return True
            finally:
                if runtime:
                    self._r_end(out=out)

    @staticmethod
    def __smile() -> opensmile.core.smile.Smile:
        """Извлечение функций OpenSmile

        .. note::
            private (приватный метод)

        Returns:
             opensmile.core.smile.Smile: Извлеченные функции OpenSmile

        .. dropdown:: Пример
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()
                audio._Audio__smile()

            .. output-cell::
                :execution-count: 1
                :linenos:
                :tab-width: 8

                {
                    '$opensmile.core.smile.Smile': {
                        'feature_set': 'eGeMAPSv02',
                        'feature_level': 'LowLevelDescriptors',
                        'options': {},
                        'sampling_rate': None,
                        'channels': [0],
                        'mixdown': False,
                        'resample': False
                    }
                }
        """

        return opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

    def __norm_pred(self, pred_data: np.ndarray, len_spec: int = 16, out: bool = True) -> np.ndarray:
        """Нормализация оценок по экспертным и нейросетевым признакам

        .. note::
            private (приватный метод)

        Args:
            pred_data (np.ndarray): Оценки
            len_spec (int): Максимальный размер вектора оценок
            out (bool): Отображение

        Returns:
            np.ndarray: Нормализованные оценки по экспертным и нейросетевым признакам

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                import numpy as np
                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                arr = np.array([
                    [0.64113516, 0.6217892, 0.54451424, 0.6144415, 0.59334993],
                    [0.6652424, 0.63606125, 0.572305, 0.63169795, 0.612515]
                ])

                audio._Audio__norm_pred(
                    pred_data = arr,
                    len_spec = 4,
                    out = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:
                :tab-width: 8

                array([
                    [0.64113516, 0.6217892 , 0.54451424, 0.6144415 , 0.59334993],
                    [0.6652424 , 0.63606125, 0.572305  , 0.63169795, 0.612515],
                    [0.65318878, 0.62892523, 0.55840962, 0.62306972, 0.60293247],
                    [0.65318878, 0.62892523, 0.55840962, 0.62306972, 0.60293247]
                ])

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                import numpy as np
                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                arr = np.array([])

                audio._Audio__norm_pred(
                    pred_data = arr,
                    len_spec = 4,
                    out = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-20 22:03:17] Неверные типы или значения аргументов в "Audio.__norm_pred" ...

                array([], dtype=float64)
        """

        try:
            # Проверка аргументов
            if (
                type(pred_data) is not np.ndarray
                or len(pred_data) < 1
                or type(len_spec) is not int
                or len_spec < 1
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__norm_pred.__name__, out=out)
            return np.array([])
        else:
            try:
                if pred_data.shape[0] < len_spec:
                    return np.pad(pred_data, ((0, len_spec - pred_data.shape[0]), (0, 0)), "mean")
                return pred_data[:len_spec]
            except ValueError:
                self._other_error(self._norm_audio_pred_error, last=False, out=out)
                return np.array([])
            except Exception:
                self._other_error(self._unknown_err, out=out)
                return np.array([])

    def __concat_pred(
        self, pred_hc: np.ndarray, pred_melspectrogram: np.ndarray, out: bool = True
    ) -> List[Optional[np.ndarray]]:
        """Конкатенация оценок по экспертным и нейросетевым признакам

        .. note::
            private (приватный метод)

        Args:
            pred_hc (np.ndarray): Оценки по экспертным признакам
            pred_melspectrogram (np.ndarray): Оценки по нейросетевым признакам
            out (bool): Отображение

        Returns:
            List[Optional[np.ndarray]]: Конкатенированные оценки по экспертным и нейросетевым признакам

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                import numpy as np
                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                arr_hc = np.array([
                    [0.64113516, 0.6217892, 0.54451424, 0.6144415, 0.59334993],
                    [0.6652424, 0.63606125, 0.572305, 0.63169795, 0.612515]
                ])

                arr_melspectrogram = np.array([
                    [0.56030345, 0.7488746, 0.44648764, 0.59893465, 0.5701077],
                    [0.5900006, 0.7652722, 0.4795154, 0.6409055, 0.6088242]
                ])

                audio._Audio__concat_pred(
                    pred_hc = arr_hc,
                    pred_melspectrogram = arr_melspectrogram,
                    out = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:
                :tab-width: 12

                [
                    array([
                        0.64113516, 0.6652424, 0.65318878, 0.65318878, 0.65318878,
                        0.65318878, 0.65318878, 0.65318878, 0.65318878, 0.65318878,
                        0.65318878, 0.65318878, 0.65318878, 0.65318878, 0.65318878,
                        0.65318878, 0.56030345, 0.5900006, 0.57515202, 0.57515202,
                        0.57515202, 0.57515202, 0.57515202, 0.57515202, 0.57515202,
                        0.57515202, 0.57515202, 0.57515202, 0.57515202, 0.57515202,
                        0.57515202, 0.57515202
                    ]),
                    array([
                        0.6217892, 0.63606125, 0.62892523, 0.62892523, 0.62892523,
                        0.62892523, 0.62892523, 0.62892523, 0.62892523, 0.62892523,
                        0.62892523, 0.62892523, 0.62892523, 0.62892523, 0.62892523,
                        0.62892523, 0.7488746, 0.7652722, 0.7570734, 0.7570734,
                        0.7570734, 0.7570734, 0.7570734, 0.7570734, 0.7570734,
                        0.7570734, 0.7570734, 0.7570734, 0.7570734, 0.7570734,
                        0.7570734, 0.7570734
                    ]),
                    array([
                        0.54451424, 0.572305, 0.55840962, 0.55840962, 0.55840962,
                        0.55840962, 0.55840962, 0.55840962, 0.55840962, 0.55840962,
                        0.55840962, 0.55840962, 0.55840962, 0.55840962, 0.55840962,
                        0.55840962, 0.44648764, 0.4795154, 0.46300152, 0.46300152,
                        0.46300152, 0.46300152, 0.46300152, 0.46300152, 0.46300152,
                        0.46300152, 0.46300152, 0.46300152, 0.46300152, 0.46300152,
                        0.46300152, 0.46300152
                    ]),
                    array([
                        0.6144415, 0.63169795, 0.62306972, 0.62306972, 0.62306972,
                        0.62306972, 0.62306972, 0.62306972, 0.62306972, 0.62306972,
                        0.62306972, 0.62306972, 0.62306972, 0.62306972, 0.62306972,
                        0.62306972, 0.59893465, 0.6409055, 0.61992008, 0.61992008,
                        0.61992008, 0.61992008, 0.61992008, 0.61992008, 0.61992008,
                        0.61992008, 0.61992008, 0.61992008, 0.61992008, 0.61992008,
                        0.61992008, 0.61992008
                    ]),
                    array([
                        0.59334993, 0.612515, 0.60293247, 0.60293247, 0.60293247,
                        0.60293247, 0.60293247, 0.60293247, 0.60293247, 0.60293247,
                        0.60293247, 0.60293247, 0.60293247, 0.60293247, 0.60293247,
                        0.60293247, 0.5701077, 0.6088242, 0.58946595, 0.58946595,
                        0.58946595, 0.58946595, 0.58946595, 0.58946595, 0.58946595,
                        0.58946595, 0.58946595, 0.58946595, 0.58946595, 0.58946595,
                        0.58946595, 0.58946595
                    ])
                ]

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                import numpy as np
                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                arr_hc = np.array([
                    [0.64113516, 0.6217892, 0.54451424, 0.6144415],
                    [0.6652424, 0.63606125, 0.572305, 0.63169795, 0.612515]
                ])

                arr_melspectrogram = np.array([
                    [0.56030345, 0.7488746, 0.44648764, 0.59893465, 0.5701077],
                    [0.5900006, 0.7652722, 0.4795154, 0.6409055, 0.6088242]
                ])

                audio._Audio__concat_pred(
                    pred_hc = arr_hc,
                    pred_melspectrogram = arr_melspectrogram,
                    out = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-20 22:33:31] Что-то пошло не так ... конкатенация оценок по экспертным и нейросетевым
                признакам не произведена (аудио модальность) ...

                []
        """

        try:
            # Проверка аргументов
            if (
                type(pred_hc) is not np.ndarray
                or len(pred_hc) < 1
                or type(pred_melspectrogram) is not np.ndarray
                or len(pred_melspectrogram) < 1
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__concat_pred.__name__, out=out)
            return []
        else:
            # Нормализация оценок по экспертным и нейросетевым признакам
            pred_hc_norm = self.__norm_pred(pred_hc, out=False)
            pred_melspectrogram_norm = self.__norm_pred(pred_melspectrogram, out=False)

            if len(pred_hc_norm) == 0 or len(pred_melspectrogram_norm) == 0:
                self._error(self._concat_audio_pred_error, out=out)
                return []

            concat = []

            try:
                # Проход по всем персональным качествам личности человека
                for i in range(len(self._b5["en"])):
                    concat.append(
                        np.hstack((np.asarray(pred_hc_norm)[:, i], np.asarray(pred_melspectrogram_norm)[:, i]))
                    )
            except IndexError:
                self._other_error(self._concat_audio_pred_error, last=False, out=out)
                return []
            except Exception:
                self._other_error(self._unknown_err, out=out)
                return []

            return concat

    def __load_audio_model_b5(
        self, show_summary: bool = False, out: bool = True
    ) -> Optional[tf.keras.Model]:
        """Формирование нейросетевой архитектуры модели для получения результата оценки персонального качества

        .. note::
            private (приватный метод)

        Args:
            show_summary (bool): Отображение сформированной нейросетевой архитектуры модели
            out (bool): Отображение

        Returns:
            Optional[tf.keras.Model]:
                **None** если неверные типы или значения аргументов, в обратном случае нейросетевая модель
                **tf.keras.Model** для получения результата оценки персонального качества

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio._Audio__load_audio_model_b5(
                    show_summary = True, out = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                Model: "model"
                _________________________________________________________________
                 Layer (type)                Output Shape              Param #
                =================================================================
                 input_1 (InputLayer)        [(None, 32)]              0

                 dense_1 (Dense)             (None, 1)                 33

                 activ_1 (Activation)        (None, 1)                 0

                =================================================================
                Total params: 33
                Trainable params: 33
                Non-trainable params: 0
                _________________________________________________________________
                <tf.keras.Model at 0x13d442940>

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio._Audio__load_audio_model_b5(
                    show_summary = True, out = []
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-17 10:53:03] Неверные типы или значения аргументов в "Audio.__load_audio_model_b5" ...
        """

        try:
            # Проверка аргументов
            if type(show_summary) is not bool or type(out) is not bool:
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__load_audio_model_b5.__name__, out=out)
            return None
        else:
            input_1 = tf.keras.Input(shape=(32,), name="input_1")
            x = tf.keras.layers.Dense(units=1, name="dense_1")(input_1)
            x = tf.keras.layers.Activation("sigmoid", name="activ_1")(x)

            model = tf.keras.Model(inputs=input_1, outputs=x)

            if show_summary and out:
                model.summary()

            return model

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    def _get_acoustic_features(
        self,
        path: str,
        sr: int = 44100,
        window: Union[int, float] = 2.0,
        step: Union[int, float] = 1.0,
        last: bool = False,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """Извлечение признаков из акустического сигнала (без очистки истории вывода сообщений в ячейке Jupyter)

        .. note::
            protected (защищенный метод)

        Args:
            path (str): Путь к аудио или видеофайлу
            sr (int): Частота дискретизации
            window (Union[int, float]): Размер окна сегмента сигнала (в секундах)
            step (Union[int, float]): Шаг сдвига окна сегмента сигнала (в секундах)
            last (bool): Замена последнего сообщения
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]: Кортеж с двумя списками:

                1. Список с экспертными признаками
                2. Список с лог мел-спектрограммами

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                sr = 44100
                path = '/Users/dl/GitHub/oceanai/oceanai/dataset/test80_01/glgfB3vFewc.004.mp4'

                hc_features, melspectrogram_features = audio._get_acoustic_features(
                    path = path, sr = sr,
                    window = 2, step = 1,
                    last = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-19 14:58:19] Извлечение признаков (экспертных и лог мел-спектрограмм) из акустического сигнала ...

                [2022-10-19 14:58:20] Статистика извлеченных признаков из акустического сигнала:
                    Общее количество сегментов с:
                        1. экспертными признаками: 12
                        2. лог мел-спектрограммами: 12
                    Размерность матрицы экспертных признаков одного сегмента: 196 ✕ 25
                    Размерность тензора с лог мел-спектрограммами одного сегмента: 224 ✕ 224 ✕ 3

                --- Время выполнения: 1.273 сек. ---

            :bdg-danger:`Ошибки` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                sr = 44100
                path = '/Users/dl/GitHub/oceanai/oceanai/dataset/test80_01/glgfB3vFewc.004.mp4'

                hc_features, melspectrogram_features = audio._get_acoustic_features(
                    path = 1, sr = sr,
                    window = 2, step = 1,
                    last = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-19 15:33:04] Неверные типы или значения аргументов в "Audio._get_acoustic_features" ...

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                sr = 44100
                path = '/Users/dl/GitHub/oceanai/oceanai/dataset/test80_01/glgfB3vFewc.004.mp4'

                hc_features, melspectrogram_features = audio._get_acoustic_features(
                    path = path, sr = sr,
                    window = 0.04, step = 1,
                    last = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-19 15:34:38] Извлечение признаков (экспертных и лог мел-спектрограмм) из акустического сигнала ...

                [2022-10-19 15:34:38] Что-то пошло не так ... указан слишком маленький размер (0.04) окна сегмента сигнала ...

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/audio.py
                    Линия: 863
                    Метод: _get_acoustic_features
                    Тип ошибки: IsSmallWindowSizeError

                --- Время выполнения: 0.049 сек. ---
        """

        try:
            # Проверка аргументов
            if (
                (type(path) is not str or not path) and (type(path) is not gradio.utils.NamedString)
                or type(sr) is not int
                or sr < 1
                or ((type(window) is not int or window < 1) and (type(window) is not float or window <= 0))
                or ((type(step) is not int or step < 1) and (type(step) is not float or step <= 0))
                or type(last) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self._get_acoustic_features.__name__, last=last, out=out)
            return [], []
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, last=last, out=out)
                return [], []

            if runtime:
                self._r_start()

            if last is False:
                # Информационное сообщение
                self._info(self._get_acoustic_feature_info, out=False)
                if out:
                    self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            try:
                # Считывание аудио или видеофайла
                path_to_wav = os.path.join(str(Path(path).parent), Path(path).stem + "." + "wav")

                if not Path(path_to_wav).is_file():
                    if Path(path).suffix not in ["mp3", "wav"]:
                        ff_audio = "ffmpeg -loglevel quiet -i {} -vn -acodec pcm_s16le -ar 44100 -ac 2 {}".format(
                            path, path_to_wav
                        )
                        call_audio = subprocess.call(ff_audio, shell=True)

                        try:
                            if call_audio == 1:
                                raise OSError
                        except OSError:
                            self._other_error(self._unknown_err, last=last, out=out)
                            return np.empty([]), np.empty([])
                        except Exception:
                            self._other_error(self._unknown_err, last=last, out=out)
                            return np.empty([]), np.empty([])
                        else:
                            audio, sr = librosa.load(path=path_to_wav, sr=sr)
                else:
                    audio, sr = librosa.load(path=path_to_wav, sr=sr)
            except FileNotFoundError:
                self._other_error(self._file_not_found.format(self._info_wrapper(path)), last=last, out=out)
                return [], []
            except IsADirectoryError:
                self._other_error(self._directory_inst_file.format(self._info_wrapper(path)), last=last, out=out)
                return [], []
            except audioread.NoBackendError:
                self._other_error(self._no_acoustic_signal.format(self._info_wrapper(path)), last=last, out=out)
                return [], []
            except Exception:
                self._other_error(self._unknown_err, last=last, out=out)
                return [], []
            else:
                hc_features = []  # Список с экспертными признаками
                melspectrogram_features = []  # Список с лог мел-спектрограммами

                try:
                    lhcf = int((window * 1000 - 40) / 10)

                    if lhcf < 2:
                        raise IsSmallWindowSizeError
                except IsSmallWindowSizeError:
                    self._other_error(
                        self._window_small_size_error.format(self._info_wrapper(str(window))), last=last, out=out
                    )
                    return [], []
                except Exception:
                    self._other_error(self._unknown_err, last=last, out=out)
                    return [], []
                else:
                    window_local = int(sr * window)

                    len_spec = window_local / self.__pl[1]
                    if math.modf(len_spec)[0] == 0:
                        len_spec += 1
                    len_spec = math.ceil(len_spec)

                    for cnt, val in enumerate(range(0, audio.shape[0] + 1, int(sr * step))):
                        val_end = val + window_local

                        curr_audio = audio[val:val_end]  # Часть аудио

                        # Формирование экспертных признаков
                        hc_feature = self.smile_.process_signal(curr_audio, sr).to_numpy()

                        try:
                            # Нормализация экспертных признаков
                            hc_feature = preprocessing.normalize(hc_feature, norm="l2", axis=0)
                        except Exception:
                            pass
                        else:
                            # Дополнение экспертных признаков нулями
                            hc_feature = np.pad(hc_feature, ((0, lhcf - hc_feature.shape[0]), (0, 0)))
                            hc_features.append(hc_feature)  # Добавление экспертных признаков в список

                        # Получение лог мел-спектрограмм
                        if len(curr_audio) > self.__pl[0]:
                            melspectrogram = librosa.feature.melspectrogram(
                                y=curr_audio,
                                sr=sr,
                                n_fft=self.__pl[0],
                                hop_length=self.__pl[1],
                                win_length=self.__pl[2],
                                center=self.__pl[3],
                                pad_mode=self.__pl[4],
                                power=self.__pl[5],
                                n_mels=self.__pl[6],
                                norm=self.__pl[7],
                                htk=self.__pl[8],
                                fmax=self.__pl[9],
                            )

                            # Преобразование спектрограммы из мощности (квадрат амплитуды) в децибелы (дБ)
                            melspectrogram_to_db = librosa.power_to_db(melspectrogram, top_db=80)

                            if melspectrogram_to_db.shape[1] < len_spec:
                                melspectrogram_to_db = np.pad(
                                    melspectrogram_to_db,
                                    ((0, 0), (0, len_spec - melspectrogram_to_db.shape[1])),
                                    "mean",
                                )
                            melspectrogram_to_db /= 255  # Линейная нормализация
                            melspectrogram_to_db = np.expand_dims(melspectrogram_to_db, axis=-1)
                            melspectrogram_to_db = tf.image.resize(melspectrogram_to_db, (224, 224))  # Масштабирование
                            melspectrogram_to_db = tf.repeat(melspectrogram_to_db, 3, axis=-1)  # GRAY -> RGB
                            # Добавление лог мел-спектрограммы в список
                            melspectrogram_features.append(melspectrogram_to_db)

                    if last is False:
                        # Статистика извлеченных признаков из акустического сигнала
                        self._stat_acoustic_features(
                            last=last,
                            out=out,
                            len_hc_features=len(hc_features),
                            len_melspectrogram_features=len(melspectrogram_features),
                            shape_hc_features=hc_features[0].shape,
                            shape_melspectrogram_features=melspectrogram_features[0].shape,
                        )

                    return hc_features, melspectrogram_features
            finally:
                if runtime:
                    self._r_end(out=out)

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def load_audio_model_hc(
        self, show_summary: bool = False, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Формирование нейросетевой архитектуры модели для получения оценок по экспертным признакам

        Args:
            show_summary (bool): Отображение сформированной нейросетевой архитектуры модели
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если нейросетевая архитектура модели сформирована, в обратном случае **False**

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()
                audio.load_audio_model_hc(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-17 13:16:23] Формирование нейросетевой архитектуры модели для получения оценок по экспертным признакам (аудио модальность) ...

                --- Время выполнения: 0.364 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()
                audio.load_audio_model_hc(
                    show_summary = 1, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-17 13:20:04] Неверные типы или значения аргументов в "Audio.load_audio_model_hc" ...

                False
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        try:
            # Проверка аргументов
            if (
                type(show_summary) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.load_audio_model_hc.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._formation_audio_model_hc, last=False, out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            input_lstm = tf.keras.Input(shape=(196, 25))

            x = tf.keras.layers.LSTM(64, return_sequences=True)(input_lstm)
            x = tf.keras.layers.Dropout(rate=0.2)(x)
            x = tf.keras.layers.LSTM(128, return_sequences=False, name="lstm_128_a_hc")(x)
            x = tf.keras.layers.Dropout(rate=0.2)(x)
            x = tf.keras.layers.Dense(5, activation="linear")(x)

            self._audio_model_hc = tf.keras.Model(inputs=input_lstm, outputs=x)

            if show_summary and out:
                self._audio_model_hc.summary()

            if runtime:
                self._r_end(out=out)

            return True

    def load_audio_model_nn(
        self, show_summary: bool = False, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Формирование нейросетевой архитектуры для получения оценок по нейросетевым признакам

        Args:
            show_summary (bool): Отображение сформированной нейросетевой архитектуры модели
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если нейросетевая архитектура модели сформирована, в обратном случае **False**

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()
                audio.load_audio_model_nn(
                    show_summary = True, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-17 13:25:34] Формирование нейросетевой архитектуры для получения оценок по нейросетевым признакам (аудио модальность) ...

                Model: "model"
                _________________________________________________________________
                 Layer (type)                Output Shape              Param #
                =================================================================
                 input_1 (InputLayer)        [(None, 224, 224, 3)]     0

                 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792

                 block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928

                 block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0

                 block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856

                 block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584

                 block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0

                 block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168

                 block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080

                 block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080

                 block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0

                 block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160

                 block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808

                 block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808

                 block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0

                 block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808

                 block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808

                 block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808

                 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0

                 flatten (Flatten)           (None, 25088)             0

                 dense (Dense)               (None, 512)               12845568

                 dropout (Dropout)           (None, 512)               0

                 dense_1 (Dense)             (None, 256)               131328

                 dense_2 (Dense)             (None, 5)                 1285

                =================================================================
                Total params: 27,692,869
                Trainable params: 27,692,869
                Non-trainable params: 0
                _________________________________________________________________
                --- Время выполнения: 0.407 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()
                audio.load_audio_model_nn(
                    show_summary = 1, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-17 13:25:40] Неверные типы или значения аргументов в "Audio.load_audio_model_nn" ...

                False
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        try:
            # Проверка аргументов
            if (
                type(show_summary) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.load_audio_model_nn.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._formation_audio_model_nn, last=False, out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            vgg_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))

            x = vgg_model.output
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(512, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
            x = tf.keras.layers.Dense(5, activation="linear")(x)

            self._audio_model_nn = tf.keras.models.Model(inputs=vgg_model.input, outputs=x)

            if show_summary and out:
                self._audio_model_nn.summary()

            if runtime:
                self._r_end(out=out)

            return True

    def load_audio_models_b5(
        self, show_summary: bool = False, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Формирование нейросетевых архитектур моделей для получения результатов оценки персональных качеств

        Args:
            show_summary (bool): Отображение последней сформированной нейросетевой архитектуры моделей
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если нейросетевые архитектуры модели сформированы, в обратном случае **False**

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()
                audio.load_audio_models_b5(
                    show_summary = True, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-18 11:39:22] Формирование нейросетевых архитектур моделей для получения результатов оценки
                персональных качеств (аудио модальность) ...

                Model: "model_4"
                _________________________________________________________________
                 Layer (type)                Output Shape              Param #
                =================================================================
                 input_1 (InputLayer)        [(None, 32)]              0

                 dense_1 (Dense)             (None, 1)                 33

                 activ_1 (Activation)        (None, 1)                 0

                =================================================================
                Total params: 33
                Trainable params: 33
                Non-trainable params: 0
                _________________________________________________________________
                --- Время выполнения: 0.163 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()
                audio.load_audio_models_b5(
                    show_summary = 1, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-18 13:47:36] Неверные типы или значения аргументов в "Audio.load_audio_models_b5" ...

                False
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        try:
            # Проверка аргументов
            if (
                type(show_summary) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.load_audio_models_b5.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._formation_audio_models_b5, last=False, out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            for key, _ in self._audio_models_b5.items():
                self._audio_models_b5[key] = self.__load_audio_model_b5()

            if show_summary and out:
                self._audio_models_b5[key].summary()

            if runtime:
                self._r_end(out=out)

            return True

    def load_audio_model_weights_hc(
        self, url: str, force_reload: bool = True, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Загрузка весов нейросетевой модели для получения оценок по экспертным признакам

        Args:
            url (str): Полный путь к файлу с весами нейросетевой модели
            force_reload (bool): Принудительная загрузка файла с весами нейросетевой модели из сети
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если веса нейросетевой модели загружены, в обратном случае **False**

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.load_audio_model_hc(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-17 14:24:28] Формирование нейросетевой архитектуры модели для получения оценок по экспертным признакам (аудио модальность) ...

                --- Время выполнения: 0.398 сек. ---

                True

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                audio.path_to_save_ = './models'
                audio.chunk_size_ = 2000000

                url = audio.weights_for_big5_['audio']['hc']['sberdisk']

                audio.load_audio_model_weights_hc(
                    url = url,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-17 14:24:30] Загрузка весов нейросетевой модели для получения оценок по экспертным признакам (аудио модальность) ...

                [2022-10-17 14:24:30] Загрузка файла "weights_2022-05-05_11-27-55.h5" (100.0%) ...

                --- Время выполнения: 0.414 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.path_to_save_ = './models'
                audio.chunk_size_ = 2000000

                url = audio.weights_for_big5_['audio']['hc']['sberdisk']

                audio.load_audio_model_weights_hc(
                    url = url,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-17 15:21:13] Загрузка весов нейросетевой модели для получения оценок по экспертным признакам (аудио модальность) ...

                [2022-10-17 15:21:14] Загрузка файла "weights_2022-05-05_11-27-55.h5" (100.0%) ...

                [2022-10-17 15:21:14] Что-то пошло не так ... нейросетевая архитектура модели для получения оценок по экспертным признакам не сформирована (аудио модальность) ...

                --- Время выполнения: 0.364 сек. ---

                False
        """

        if runtime:
            self._r_start()

        if self.__load_model_weights(url, force_reload, self._load_audio_model_weights_hc, out, False, run) is True:
            try:
                self._audio_model_hc.load_weights(self._url_last_filename)
                self._audio_model_hc = tf.keras.models.Model(
                    inputs=self._audio_model_hc.input,
                    outputs=[self._audio_model_hc.output, self._audio_model_hc.get_layer("lstm_128_a_hc").output],
                )
            except Exception:
                self._error(self._model_audio_hc_not_formation, out=out)
                return False
            else:
                return True
            finally:
                if runtime:
                    self._r_end(out=out)

        return False

    def load_audio_model_weights_nn(
        self, url: str, force_reload: bool = True, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Загрузка весов нейросетевой модели для получения оценок по нейросетевым признакам

        Args:
            url (str): Полный путь к файлу с весами нейросетевой модели
            force_reload (bool): Принудительная загрузка файла с весами нейросетевой модели из сети
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если веса нейросетевой модели загружены, в обратном случае **False**

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.load_audio_model_nn(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-17 15:47:20] Формирование нейросетевой архитектуры для получения оценок по нейросетевым
                признакам (аудио модальность) ...

                --- Время выполнения: 0.419 сек. ---

                True

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                audio.path_to_save_ = './models'
                audio.chunk_size_ = 2000000

                url = audio.weights_for_big5_['audio']['nn']['sberdisk']

                audio.load_audio_model_weights_nn(
                    url = url,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-17 15:47:22] Загрузка весов нейросетевой модели для получения оценок по нейросетевым
                признакам (аудио модальность) ...

                [2022-10-17 15:47:26] Загрузка файла "weights_2022-05-03_07-46-14.h5" (100.0%) ...

                --- Время выполнения: 3.884 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.path_to_save_ = './models'
                audio.chunk_size_ = 2000000

                url = audio.weights_for_big5_['audio']['nn']['sberdisk']

                audio.load_audio_model_weights_nn(
                    url = url,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-17 15:49:57] Загрузка весов нейросетевой модели для получения оценок по нейросетевым признакам (аудио модальность) ...

                [2022-10-17 15:50:04] Загрузка файла "weights_2022-05-03_07-46-14.h5" (100.0%) ...

                [2022-10-17 15:50:04] Что-то пошло не так ... нейросетевая архитектура модели для получения оценок по нейросетевым признакам не сформирована (аудио модальность) ...

                --- Время выполнения: 6.786 сек. ---

                False
        """

        if runtime:
            self._r_start()

        if self.__load_model_weights(url, force_reload, self._load_audio_model_weights_nn, out, False, run) is True:
            try:
                self._audio_model_nn.load_weights(self._url_last_filename)
                self._audio_model_nn = tf.keras.models.Model(
                    inputs=self._audio_model_nn.input,
                    outputs=[self._audio_model_nn.output, self._audio_model_nn.get_layer("dense_256").output],
                )
            except Exception:
                self._error(self._model_audio_nn_not_formation, out=out)
                return False
            else:
                return True
            finally:
                if runtime:
                    self._r_end(out=out)

        return False

    def load_audio_models_weights_b5(
        self,
        url_openness: str,
        url_conscientiousness: str,
        url_extraversion: str,
        url_agreeableness: str,
        url_non_neuroticism: str,
        force_reload: bool = True,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> bool:
        """Загрузка весов нейросетевых моделей для получения результатов оценки персональных качеств

        Args:
            url_openness (str): Полный путь к файлу с весами нейросетевой модели (открытость опыту)
            url_conscientiousness (str): Полный путь к файлу с весами нейросетевой модели (добросовестность)
            url_extraversion (str): Полный путь к файлу с весами нейросетевой модели (экстраверсия)
            url_agreeableness (str): Полный путь к файлу с весами нейросетевой модели (доброжелательность)
            url_non_neuroticism (str): Полный путь к файлу с весами нейросетевой модели (эмоциональная стабильность)
            force_reload (bool): Принудительная загрузка файлов с весами нейросетевых моделей из сети
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если веса нейросетевых моделей загружены, в обратном случае **False**

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.load_audio_models_b5(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-18 22:40:05] Формирование нейросетевых архитектур моделей для получения результатов оценки
                персональных качеств (аудио модальность) ...

                --- Время выполнения: 0.163 сек. ---

                True

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                audio.path_to_save_ = './models'
                audio.chunk_size_ = 2000000

                url_openness = audio.weights_for_big5_['audio']['b5']['openness']['sberdisk']
                url_conscientiousness = audio.weights_for_big5_['audio']['b5']['conscientiousness']['sberdisk']
                url_extraversion = audio.weights_for_big5_['audio']['b5']['extraversion']['sberdisk']
                url_agreeableness = audio.weights_for_big5_['audio']['b5']['agreeableness']['sberdisk']
                url_non_neuroticism = audio.weights_for_big5_['audio']['b5']['non_neuroticism']['sberdisk']

                audio.load_audio_models_weights_b5(
                    url_openness = url_openness,
                    url_conscientiousness = url_conscientiousness,
                    url_extraversion = url_extraversion,
                    url_agreeableness = url_agreeableness,
                    url_non_neuroticism = url_non_neuroticism,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-18 23:08:37] Загрузка весов нейросетевых моделей для получения результатов оценки
                персональных качеств (аудио модальность) ...

                [2022-10-18 23:08:37] Загрузка файла "weights_2022-06-15_16-16-20.h5" (100.0%) ... Открытость опыту

                [2022-10-18 23:08:38] Загрузка файла "weights_2022-06-15_16-21-57.h5" (100.0%) ... Добросовестность

                [2022-10-18 23:08:38] Загрузка файла "weights_2022-06-15_16-26-41.h5" (100.0%) ... Экстраверсия

                [2022-10-18 23:08:38] Загрузка файла "weights_2022-06-15_16-32-51.h5" (100.0%) ... Доброжелательность

                [2022-10-18 23:08:39] Загрузка файла "weights_2022-06-15_16-37-46.h5" (100.0%) ... Эмоциональная стабильность

                --- Время выполнения: 1.611 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.audio import Audio

                audio = Audio()

                audio.path_to_save_ = './models'
                audio.chunk_size_ = 2000000

                url_openness = audio.weights_for_big5_['audio']['b5']['openness']['sberdisk']
                url_conscientiousness = audio.weights_for_big5_['audio']['b5']['conscientiousness']['sberdisk']
                url_extraversion = audio.weights_for_big5_['audio']['b5']['extraversion']['sberdisk']
                url_agreeableness = audio.weights_for_big5_['audio']['b5']['agreeableness']['sberdisk']
                url_non_neuroticism = audio.weights_for_big5_['audio']['b5']['non_neuroticism']['sberdisk']

                audio.load_audio_models_weights_b5(
                    url_openness = url_openness,
                    url_conscientiousness = url_conscientiousness,
                    url_extraversion = url_extraversion,
                    url_agreeableness = url_agreeableness,
                    url_non_neuroticism = url_non_neuroticism,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-18 23:09:40] Загрузка весов нейросетевых моделей для получения результатов оценки
                персональных качеств (аудио модальность) ...

                [2022-10-18 23:09:41] Загрузка файла "weights_2022-06-15_16-16-20.h5" (100.0%) ...

                [2022-10-18 23:09:41] Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ...
                Открытость опыту

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/audio.py
                    Линия: 1764
                    Метод: load_audio_models_weights_b5
                    Тип ошибки: AttributeError

                [2022-10-18 23:09:41] Загрузка файла "weights_2022-06-15_16-21-57.h5" (100.0%) ...

                [2022-10-18 23:09:41] Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ...
                Добросовестность

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/audio.py
                    Линия: 1764
                    Метод: load_audio_models_weights_b5
                    Тип ошибки: AttributeError

                [2022-10-18 23:09:41] Загрузка файла "weights_2022-06-15_16-26-41.h5" (100.0%) ...

                [2022-10-18 23:09:41] Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ...
                Экстраверсия

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/audio.py
                    Линия: 1764
                    Метод: load_audio_models_weights_b5
                    Тип ошибки: AttributeError

                [2022-10-18 23:09:42] Загрузка файла "weights_2022-06-15_16-32-51.h5" (100.0%) ...

                [2022-10-18 23:09:42] Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ...
                Доброжелательность

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/audio.py
                    Линия: 1764
                    Метод: load_audio_models_weights_b5
                    Тип ошибки: AttributeError

                [2022-10-18 23:09:42] Загрузка файла "weights_2022-06-15_16-37-46.h5" (100.0%) ...

                [2022-10-18 23:09:42] Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ...
                Эмоциональная стабильность

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/audio.py
                    Линия: 1764
                    Метод: load_audio_models_weights_b5
                    Тип ошибки: AttributeError

                --- Время выполнения: 1.573 сек. ---

                False
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        try:
            # Проверка аргументов
            if (
                type(url_openness) is not str
                or not url_openness
                or type(url_conscientiousness) is not str
                or not url_conscientiousness
                or type(url_extraversion) is not str
                or not url_extraversion
                or type(url_agreeableness) is not str
                or not url_agreeableness
                or type(url_non_neuroticism) is not str
                or not url_non_neuroticism
                or type(force_reload) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.load_audio_models_weights_b5.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            result_download_models = 0  # Все веса нейросетевых моделей по умолчанию загружены

            # Информационное сообщение
            self._info(self._load_audio_models_weights_b5, last=False, out=out)

            # Проход по всем URL с весами нейросетевых моделей
            for cnt, url in enumerate(
                [
                    (url_openness, self._b5["ru"][0]),
                    (url_conscientiousness, self._b5["ru"][1]),
                    (url_extraversion, self._b5["ru"][2]),
                    (url_agreeableness, self._b5["ru"][3]),
                    (url_non_neuroticism, self._b5["ru"][4]),
                ]
            ):
                sections = urlparse(url[0])  # Парсинг URL адреса

                try:
                    # URL файл невалидный
                    if sections.scheme == "":
                        raise requests.exceptions.InvalidURL
                except requests.exceptions.InvalidURL:
                    url_norm = os.path.normpath(url[0])

                    try:
                        if os.path.isfile(url_norm) is False:
                            raise FileNotFoundError  # Не файл
                    except FileNotFoundError:
                        self._other_error(
                            self._load_model_weights_error + " " + self._bold_wrapper(url[1].capitalize()), out=out
                        )
                        continue
                    except Exception:
                        self._other_error(self._unknown_err, out=out)
                        continue
                    else:
                        self._url_last_filename = url_norm

                        # Отображение истории вывода сообщений в ячейке Jupyter
                        if out:
                            self.show_notebook_history_output()
                else:
                    try:
                        if force_reload is False:
                            clear_output(True)
                        # Загрузка файла из URL
                        res_download_file_from_url = self._download_file_from_url(
                            url=url[0], force_reload=force_reload, runtime=False, out=out, run=True
                        )
                    except Exception:
                        self._other_error(self._unknown_err, out=out)
                        continue
                    else:
                        # Файл загружен
                        if res_download_file_from_url != 200:
                            continue

                        try:
                            self._audio_models_b5[self._b5["en"][cnt]].load_weights(self._url_last_filename)
                        except Exception:
                            self._other_error(
                                self._load_model_weights_error + " " + self._bold_wrapper(url[1].capitalize()), out=out
                            )
                            continue
                        else:
                            self._add_last_el_notebook_history_output(self._bold_wrapper(url[1].capitalize()))

                            result_download_models += 1

            clear_output(True)
            # Отображение истории вывода сообщений в ячейке Jupyter
            if out:
                self.show_notebook_history_output()

            if runtime:
                self._r_end(out=out)

            if result_download_models != len(self._b5["ru"]):
                return False
            return True

    def get_acoustic_features(
        self,
        path: str,
        sr: int = 44100,
        window: Union[int, float] = 2.0,
        step: Union[int, float] = 1.0,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """Извлечение признаков из акустического сигнала

        Args:
            path (str): Путь к аудио или видеофайлу
            sr (int): Частота дискретизации
            window (Union[int, float]): Размер окна сегмента сигнала (в секундах)
            step (Union[int, float]): Шаг сдвига окна сегмента сигнала (в секундах)
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]: Кортеж с двумя списками:

                1. Список с экспертными признаками
                2. Список с лог мел-спектрограммами

        :bdg-link-light:`Пример <../../user_guide/notebooks/Audio-get_acoustic_features.ipynb>`
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        return self._get_acoustic_features(
            path=path, sr=sr, window=window, step=step, last=False, out=out, runtime=runtime, run=run
        )

    def get_audio_union_predictions(
        self,
        depth: int = 1,
        recursive: bool = False,
        sr: int = 44100,
        window: Union[int, float] = 2.0,
        step: Union[int, float] = 1.0,
        accuracy=True,
        url_accuracy: str = "",
        logs: bool = True,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> bool:
        """Получения прогнозов по аудио

        Args:
            depth (int): Глубина иерархии для получения данных
            recursive (bool): Рекурсивный поиск данных
            sr (int): Частота дискретизации
            window (Union[int, float]): Размер окна сегмента сигнала (в секундах)
            step (Union[int, float]): Шаг сдвига окна сегмента сигнала (в секундах)
            accuracy (bool): Вычисление точности
            url_accuracy (str): Полный путь к файлу с верными предсказаниями для подсчета точности
            logs (bool): При необходимости формировать LOG файл
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если прогнозы успешно получены, в обратном случае **False**

        :bdg-link-light:`Пример <../../user_guide/notebooks/Audio-get_audio_union_predictions.ipynb>`
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        # Сброс
        self._df_files = pd.DataFrame()  # Пустой DataFrame с данными
        self._df_accuracy = pd.DataFrame()  # Пустой DataFrame с результатами вычисления точности

        try:
            # Проверка аргументов
            if (
                type(depth) is not int
                or depth < 1
                or type(out) is not bool
                or type(recursive) is not bool
                or type(sr) is not int
                or sr < 1
                or ((type(window) is not int or window < 1) and (type(window) is not float or window <= 0))
                or ((type(step) is not int or step < 1) and (type(step) is not float or step <= 0))
                or type(accuracy) is not bool
                or type(url_accuracy) is not str
                or type(logs) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.get_audio_union_predictions.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            try:
                # Получение директорий, где хранятся данные
                path_to_data = self._get_paths(self.path_to_dataset_, depth, out=out)
                if type(path_to_data) is bool:
                    return False

                if type(self.keys_dataset_) is not list:
                    raise TypeError

                # Словарь для DataFrame набора данных с данными
                self._dict_of_files = dict(zip(self.keys_dataset_, [[] for _ in range(0, len(self.keys_dataset_))]))
                # Словарь для DataFrame набора данных с результатами вычисления точности
                self._dict_of_accuracy = dict(
                    zip(self.keys_dataset_[1:], [[] for _ in range(0, len(self.keys_dataset_[1:]))])
                )
            except (TypeError, FileNotFoundError):
                self._other_error(self._folder_not_found.format(self._info_wrapper(self.path_to_dataset_)), out=out)
                return False
            except Exception:
                self._other_error(self._unknown_err, out=out)
                return False
            else:
                # Вычисление точности
                if accuracy is True:
                    get_audio_union_predictions_info = self._get_union_predictions_info + self._get_accuracy_info
                else:
                    get_audio_union_predictions_info = self._get_union_predictions_info

                get_audio_union_predictions_info += self._audio_modality

                # Вычисление точности
                if accuracy is True:
                    # Информационное сообщение
                    self._info(get_audio_union_predictions_info, out=out)

                    if not url_accuracy:
                        url_accuracy = self._true_traits["sberdisk"]

                    try:
                        # Загрузка верных предсказаний
                        data_true_traits = pd.read_csv(url_accuracy)
                    except (FileNotFoundError, URLError, UnicodeDecodeError):
                        self._other_error(self._load_data_true_traits_error, out=out)
                        return False
                    except Exception:
                        self._other_error(self._unknown_err, out=out)
                        return False
                    else:
                        true_traits = []
                        self._del_last_el_notebook_history_output()

                paths = []  # Пути до искомых файлов

                # Проход по всем директориям
                for curr_path in path_to_data:
                    empty = True  # По умолчанию директория пустая

                    # Рекурсивный поиск данных
                    if recursive is True:
                        g = Path(curr_path).rglob("*")
                    else:
                        g = Path(curr_path).glob("*")

                    # Формирование словаря для DataFrame
                    for p in g:
                        try:
                            if type(self.ext_) is not list or len(self.ext_) < 1:
                                raise TypeError

                            self.ext_ = [x.lower() for x in self.ext_]
                        except TypeError:
                            self._other_error(self._wrong_ext, out=out)
                            return False
                        except Exception:
                            self._other_error(self._unknown_err, out=out)
                            return False
                        else:
                            # Расширение файла соответствует расширению искомых файлов
                            if p.suffix.lower() in self.ext_:
                                if empty is True:
                                    empty = False  # Каталог не пустой

                                paths.append(p.resolve())

                try:
                    self.__len_paths = len(paths)  # Количество искомых файлов

                    if self.__len_paths == 0:
                        raise TypeError
                except TypeError:
                    self._other_error(self._files_not_found, out=out)
                    return False
                except Exception:
                    self._other_error(self._unknown_err, out=out)
                    return False
                else:
                    # Локальный путь
                    self.__local_path = lambda path: os.path.join(
                        *Path(path).parts[-abs((len(Path(path).parts) - len(Path(self.path_to_dataset_).parts))) :]
                    )

                    last = False  # Замена последнего сообщения

                    # Проход по всем искомым файлов
                    for i, curr_path in enumerate(paths):
                        if i != 0:
                            last = True

                        # Индикатор выполнения
                        self._progressbar_union_predictions(
                            get_audio_union_predictions_info,
                            i,
                            self.__local_path(curr_path),
                            self.__len_paths,
                            True,
                            last,
                            out,
                        )

                        # Извлечение признаков из акустического сигнала
                        hc_features, melspectrogram_features = self._get_acoustic_features(
                            path=str(curr_path.resolve()),
                            sr=sr,
                            window=window,
                            step=step,
                            last=True,
                            out=False,
                            runtime=False,
                            run=run,
                        )

                        # Признаки из акустического сигнала извлечены
                        if len(hc_features) > 0 and len(melspectrogram_features) > 0:
                            # Коды ошибок нейросетевых моделей
                            code_error_pred_hc = -1
                            code_error_pred_melspectrogram = -1

                            try:
                                # Оправка экспертных признаков в нейросетевую модель
                                pred_hc, _ = self.audio_model_hc_(np.array(hc_features, dtype=np.float16))
                            except TypeError:
                                code_error_pred_hc = 1
                            except Exception:
                                code_error_pred_hc = 2

                            try:
                                # Отправка нейросетевых признаков в нейросетевую модель
                                pred_melspectrogram, _ = self.audio_model_nn_(
                                    np.array(melspectrogram_features, dtype=np.float16)
                                )
                            except TypeError:
                                code_error_pred_melspectrogram = 1
                            except Exception:
                                code_error_pred_melspectrogram = 2

                            if code_error_pred_hc != -1 and code_error_pred_melspectrogram != -1:
                                self._error(self._models_audio_not_formation, out=out)
                                return False

                            if code_error_pred_hc != -1:
                                self._error(self._model_audio_hc_not_formation, out=out)
                                return False

                            if code_error_pred_melspectrogram != -1:
                                self._error(self._model_audio_nn_not_formation, out=out)
                                return False

                            # Конкатенация оценок по экспертным и нейросетевым признакам
                            union_pred = self.__concat_pred(pred_hc.numpy(), pred_melspectrogram.numpy(), out=out)

                            if len(union_pred) == 0:
                                return False

                            final_pred = []

                            for cnt, (name_b5, model) in enumerate(self.audio_models_b5_.items()):
                                result = model(np.expand_dims(union_pred[cnt], axis=0)).numpy()[0][0]

                                final_pred.append(result)

                            # Добавление данных в словарь для DataFrame
                            if self._append_to_list_of_files(str(curr_path.resolve()), final_pred, out) is False:
                                return False

                            # Вычисление точности
                            if accuracy is True:
                                try:
                                    true_trait = (
                                        data_true_traits[data_true_traits.NAME_VIDEO == curr_path.name][
                                            list(self._b5["en"])
                                        ]
                                        .values[0]
                                        .tolist()
                                    )
                                except IndexError:
                                    self._other_error(self._expert_values_not_found, out=out)
                                    return False
                                except Exception:
                                    self._other_error(self._unknown_err, out=out)
                                    return False
                                else:
                                    true_traits.append(true_trait)
                        else:
                            # Добавление данных в словарь для DataFrame
                            if (
                                self._append_to_list_of_files(
                                    str(curr_path.resolve()), [None] * len(self._b5["en"]), out
                                )
                                is False
                            ):
                                return False

                    # Индикатор выполнения
                    self._progressbar_union_predictions(
                        get_audio_union_predictions_info,
                        self.__len_paths,
                        self.__local_path(paths[-1]),
                        self.__len_paths,
                        True,
                        last,
                        out,
                    )

                    # Отображение в DataFrame с данными
                    self._df_files = pd.DataFrame.from_dict(data=self._dict_of_files, orient="index").transpose()
                    self._df_files.index.name = self._keys_id
                    self._df_files.index += 1

                    self._df_files.index = self._df_files.index.map(str)

                    self._df_files.Path = [os.path.basename(i) for i in self._df_files.Path]

                    # Отображение
                    if out is True:
                        self._add_notebook_history_output(self._df_files.iloc[0 : self.num_to_df_display_, :])

                    # Подсчет точности
                    if accuracy is True:
                        mae_curr = []

                        for cnt, name_b5 in enumerate(self._df_files.keys().tolist()[1:]):
                            mae_curr.append(
                                mean_absolute_error(np.asarray(true_traits)[:, cnt], self._df_files[name_b5].to_list())
                            )

                        mae_curr = [round(float(i), 4) for i in mae_curr]
                        mae_mean = round(float(np.mean(mae_curr)), 4)
                        accuracy_curr = [round(float(i), 4) for i in 1 - np.asarray(mae_curr)]
                        accuracy_mean = round(float(np.mean(accuracy_curr)), 4)

                        for curr_acc in [mae_curr, accuracy_curr]:
                            # Добавление данных в словарь для DataFrame с результатами вычисления точности
                            if self._append_to_list_of_accuracy(curr_acc, out) is False:
                                return False

                        self._dict_of_accuracy.update({self.__df_accuracy_mean: [mae_mean, accuracy_mean]})
                        # Отображение в DataFrame с данными
                        self._df_accuracy = pd.DataFrame.from_dict(
                            data=self._dict_of_accuracy, orient="index"
                        ).transpose()
                        self._df_accuracy.index = self.__df_accuracy_index
                        self._df_accuracy.index.name = self.__df_accuracy_index_name

                        # Информационное сообщение
                        self._info(self._get_union_predictions_result, out=False)

                        # Отображение
                        if out is True:
                            self._add_notebook_history_output(self._df_accuracy.iloc[0 : self.num_to_df_display_, :])

                        self._info(
                            self._get_union_predictions_results_mean.format(
                                self._info_wrapper(str(mae_mean)), self._info_wrapper(str(accuracy_mean))
                            ),
                            out=False,
                        )

                    clear_output(True)
                    # Отображение истории вывода сообщений в ячейке Jupyter
                    if out is True:
                        self.show_notebook_history_output()

                    if logs is True:
                        # Текущее время для лог-файла
                        # см. datetime.fromtimestamp()
                        curr_ts = str(datetime.now().timestamp()).replace(".", "_")

                        name_logs_file = self.get_audio_union_predictions.__name__

                        # Сохранение LOG
                        res_save_logs_df_files = self._save_logs(
                            self._df_files, name_logs_file + "_df_files_" + curr_ts
                        )

                        # Подсчет точности
                        if accuracy is True:
                            # Сохранение LOG
                            res_save_logs_df_accuracy = self._save_logs(
                                self._df_accuracy, name_logs_file + "_df_accuracy_" + curr_ts
                            )

                        if res_save_logs_df_files is True:
                            # Сохранение LOG файла/файлов
                            if accuracy is True and res_save_logs_df_accuracy is True:
                                logs_s = self._logs_saves_true
                            else:
                                logs_s = self._logs_save_true

                            self._info_true(logs_s, out=out)

                    return True
            finally:
                if runtime:
                    self._r_end(out=out)
