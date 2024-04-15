#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Мультимодальное объединение информации
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

from urllib.parse import urlparse
from urllib.error import URLError
from pathlib import Path  # Работа с путями в файловой системе
from sklearn.metrics import mean_absolute_error
from datetime import datetime  # Работа со временем

# Типы данных
from typing import List, Dict, Union, Optional, Callable

from IPython.display import clear_output

# Персональные
from oceanai.modules.lab.audio import Audio  # Аудио
from oceanai.modules.lab.video import Video  # Видео
from oceanai.modules.lab.text import Text  # Текст

# Порог регистрации сообщений TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # Машинное обучение от Google
import keras

from oceanai.modules.lab.utils.gfl import GFL  # Модуль внимания
from oceanai.modules.lab.utils.addition import Concat


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class PredictionMessages(Audio, Video, Text):
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

        self._av_modality: str = self._(" (мультимодальное объединение) ...")

        self._formation_av_models_b5: str = self._formation_models_b5 + self._av_modality

        self._formation_avt_model_b5: str = (
            self._("Формирование нейросетевой архитектуры модели для получения " " оценок персональных качеств")
            + self._av_modality
        )
        self._load_avt_model_weights_b5: str = (
            self._("Загрузка весов нейросетевой модели для получения " "оценок персональных качеств")
            + self._av_modality
        )

        self._model_avt_not_formation: str = (
            self._oh
            + self._(
                "нейросетевая архитектура модели для получения " "оценок по мультимодальным данным не " "сформирована"
            )
            + self._av_modality
        )

        self._load_av_models_weights_b5: str = self._load_models_weights_b5 + self._av_modality

        self._concat_av_pred_error: str = self._concat_pred_error + self._av_modality
        self._norm_av_pred_error: str = self._norm_pred_error + self._av_modality


# ######################################################################################################################
# Мультимодальное объединение
# ######################################################################################################################
@dataclass
class Prediction(PredictionMessages):
    """Класс для мультимодального объединения информации

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

        # Нейросетевые модели **tf.keras.Model** для получения результатов оценки персональных качеств
        self._av_models_b5: Dict[str, Optional[tf.keras.Model]] = dict(
            zip(self._b5["en"], [None] * len(self._b5["en"]))
        )

        self._avt_model_b5: Optional[tf.keras.Model] = None

        # ----------------------- Только для внутреннего использования внутри класса

        self.__len_paths: int = 0  # Количество искомых файлов
        self.__local_path: Union[Callable[[str], str], None] = None  # Локальный путь

        # Ключи для точности
        self.__df_accuracy_index: List[str] = ["MAE", "Accuracy"]
        self.__df_accuracy_index_name: str = "Metrics"
        self.__df_accuracy_mean: str = "Mean"

        clear_output(False)  # Удаление сообщения: INFO: Created TensorFlow Lite XNNPACK delegate for CPU)

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def av_models_b5_(self) -> Dict[str, Optional[tf.keras.Model]]:
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

                from oceanai.modules.lab.prediction import Prediction

                pred = Prediction()

                pred.load_av_models_b5(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

                pred.av_models_b5_

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-12-08 15:21:22] Формирование нейросетевых архитектур моделей для получения результатов оценки персональных качеств (мультимодальное объединение) ...

                --- Время выполнения: 0.305 сек. ---

                {
                    'openness': <tf.keras.Model at 0x14eee5790>,
                    'conscientiousness': <tf.keras.Model at 0x14f2d9d00>,
                    'extraversion': <tf.keras.Model at 0x14f2fb190>,
                    'agreeableness': <tf.keras.Model at 0x14f2c7fd0>,
                    'non_neuroticism': <tf.keras.Model at 0x14f2ef940>
                }

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.prediction import Prediction

                pred = Prediction()

                pred.av_models_b5_

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

        return self._av_models_b5

    @property
    def avt_model_b5_(self) -> Optional[tf.keras.Model]:
        """Получение нейросетевой модели **tf.keras.Model** для получения оценок персональных качеств

        Returns:
            Dict: Нейроаетевая модель **tf.keras.Model**
        """

        return self._avt_model_b5

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)
    # ------------------------------------------------------------------------------------------------------------------

    def __norm_pred(self, pred_data: np.ndarray, len_nn: int = 16, out: bool = True) -> np.ndarray:
        """Нормализация оценок по экспертным и нейросетевым признакам (мультимодальная)

        .. note::
            private (приватный метод)

        Args:
            pred_data (np.ndarray): Оценки
            len_nn (int): Максимальный размер вектора оценок
            out (bool): Отображение

        Returns:
            np.ndarray: Нормализованные оценки по экспертным и нейросетевым признакам (мультимодальная)

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                import numpy as np
                from oceanai.modules.lab.prediction import Prediction

                pred = Prediction()

                arr = np.array([
                    [0.64113516, 0.6217892, 0.54451424, 0.6144415, 0.59334993],
                    [0.6652424, 0.63606125, 0.572305, 0.63169795, 0.612515]
                ])

                pred._Prediction__norm_pred(
                    pred_data = arr,
                    len_nn = 4,
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
                from oceanai.modules.lab.prediction import Prediction

                pred = Prediction()

                arr = np.array([])

                pred._Prediction__norm_pred(
                    pred_data = arr,
                    len_nn = 4,
                    out = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-20 22:03:17] Неверные типы или значения аргументов в "Prediction.__norm_pred" ...

                array([], dtype=float64)
        """

        try:
            # Проверка аргументов
            if (
                type(pred_data) is not np.ndarray
                or len(pred_data) < 1
                or type(len_nn) is not int
                or len_nn < 1
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__norm_pred.__name__, out=out)
            return np.array([])
        else:
            try:
                if pred_data.shape[0] < len_nn:
                    return np.pad(pred_data, ((0, len_nn - pred_data.shape[0]), (0, 0)), "mean")
                return pred_data[:len_nn]
            except ValueError:
                self._other_error(self._norm_av_pred_error, last=False, out=out)
                return np.array([])
            except Exception:
                self._other_error(self._unknown_err, out=out)
                return np.array([])

    def __concat_pred_av(
        self,
        pred_hc_audio: np.ndarray,
        pred_nn_audio: np.ndarray,
        pred_hc_video: np.ndarray,
        pred_nn_video: np.ndarray,
        out: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """Конкатенация оценок по экспертным и нейросетевым признакам (мультимодальная)

        .. note::
            private (приватный метод)

        Args:
            pred_hc_audio (np.ndarray): Оценки по экспертным признакам (аудио модальность)
            pred_nn_audio (np.ndarray): Оценки по нейросетевым признакам (аудио модальность)
            pred_hc_video (np.ndarray): Оценки по экспертным признакам (видео модальность)
            pred_nn_video (np.ndarray): Оценки по нейросетевым признакам (видео модальность)
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
                from oceanai.modules.lab.prediction import Prediction

                pred = Prediction()

                arr_hc_audio = np.array([
                    [0.64113516, 0.6217892, 0.54451424, 0.6144415, 0.59334993],
                    [0.6652424, 0.63606125, 0.572305, 0.63169795, 0.612515]
                ])

                arr_nn_audio = np.array([
                    [0.56030345, 0.7488746, 0.44648764, 0.59893465, 0.5701077],
                    [0.5900006, 0.7652722, 0.4795154, 0.6409055, 0.6088242]
                ])
                arr_hc_video = np.array([
                    [0.67113516, 0.6517892, 0.59451424, 0.6344415, 0.53334993],
                    [0.6852424, 0.62606125, 0.562305, 0.67169795, 0.672515]
                ])

                arr_nn_video = np.array([
                    [0.58030345, 0.7788746, 0.47648764, 0.53893465, 0.5901077],
                    [0.5100006, 0.7452722, 0.4495154, 0.6909055, 0.6488242]
                ])

                pred._Prediction__concat_pred_av(
                    pred_hc_audio = arr_hc_audio,
                    pred_nn_audio = arr_nn_audio,
                    pred_hc_video = arr_hc_video,
                    pred_nn_video = arr_nn_video,
                    out = True
                )
        """

        try:
            # Проверка аргументов
            if (
                type(pred_hc_audio) is not np.ndarray
                or len(pred_hc_audio) < 1
                or type(pred_nn_audio) is not np.ndarray
                or len(pred_nn_audio) < 1
                or type(pred_hc_video) is not np.ndarray
                or len(pred_hc_video) < 1
                or type(pred_nn_video) is not np.ndarray
                or len(pred_nn_video) < 1
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__concat_pred_av.__name__, out=out)
            return []
        else:
            # Нормализация оценок по экспертным и нейросетевым признакам (аудио модальность)
            pred_hc_audio_norm = self.__norm_pred(pred_hc_audio, out=False)
            pred_nn_audio_norm = self.__norm_pred(pred_nn_audio, out=False)

            # Нормализация оценок по экспертным и нейросетевым признакам (видео модальность)
            pred_hc_video_norm = self.__norm_pred(pred_hc_video, out=False)
            pred_nn_video_norm = self.__norm_pred(pred_nn_video, out=False)

            if (
                len(pred_hc_audio_norm) == 0
                or len(pred_nn_audio_norm) == 0
                or len(pred_hc_video_norm) == 0
                or len(pred_nn_video_norm) == 0
            ):
                self._error(self._concat_av_pred_error, out=out)
                return []

            concat = []

            try:
                # Проход по всем персональным качествам личности человека
                for i in range(len(self._b5["en"])):
                    concat.append(
                        np.hstack(
                            (
                                np.asarray(pred_hc_audio_norm)[:, i],
                                np.asarray(pred_nn_audio_norm)[:, i],
                                np.asarray(pred_hc_video_norm)[:, i],
                                np.asarray(pred_nn_video_norm)[:, i],
                            )
                        )
                    )
            except IndexError:
                self._other_error(self._concat_av_pred_error, last=False, out=out)
                return []
            except Exception:
                self._other_error(self._unknown_err, out=out)
                return []

            return concat

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

    def __load_avt_model_b5(self, show_summary: bool = False, out: bool = True) -> Optional[tf.keras.Model]:
        """Формирование нейросетевой архитектуры модели для получения оценок персональных качеств

        .. note::
            private (приватный метод)

        Args:
            show_summary (bool): Отображение сформированной нейросетевой архитектуры модели
            out (bool): Отображение

        Returns:
            Optional[tf.keras.Model]:
                **None** если неверные типы или значения аргументов, в обратном случае нейросетевая модель
                **tf.keras.Model** для получения оценок персональных качеств
        """

        try:
            # Проверка аргументов
            if type(show_summary) is not bool or type(out) is not bool:
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__load_av_model_b5.__name__, out=out)
            return None
        else:
            i_hc_t_1 = tf.keras.Input(shape=(128,), name="hc_t")
            i_nn_t_1 = tf.keras.Input(shape=(128,), name="nn_t")
            i_hc_a_1 = tf.keras.Input(shape=(256,), name="hc_a")
            i_nn_a_1 = tf.keras.Input(shape=(512,), name="nn_a")
            i_hc_v_1 = tf.keras.Input(shape=(256,), name="hc_v")
            i_nn_v_1 = tf.keras.Input(shape=(2048,), name="nn_v")

            i_hc_t_1_n = tf.keras.layers.LayerNormalization(axis=1, name="ln_hc_t")(i_hc_t_1)
            i_nn_t_1_n = tf.keras.layers.LayerNormalization(axis=1, name="ln_nn_t")(i_nn_t_1)
            i_hc_a_1_n = tf.keras.layers.LayerNormalization(axis=1, name="ln_hc_a")(i_hc_a_1)
            i_nn_a_1_n = tf.keras.layers.LayerNormalization(axis=1, name="ln_nn_a")(i_nn_a_1)
            i_hc_v_1_n = tf.keras.layers.LayerNormalization(axis=1, name="ln_hc_v")(i_hc_v_1)
            i_nn_v_1_n = tf.keras.layers.LayerNormalization(axis=1, name="ln_nn_v")(i_nn_v_1)

            gf_ta = GFL(output_dim=64, kernel_initializer=tf.keras.initializers.TruncatedNormal(seed=42), name="gata")
            gf_tv = GFL(output_dim=64, kernel_initializer=tf.keras.initializers.TruncatedNormal(seed=42), name="gatv")
            gf_av = GFL(output_dim=64, kernel_initializer=tf.keras.initializers.TruncatedNormal(seed=42), name="gaav")

            gf_ta_1 = gf_ta([i_hc_t_1_n, i_hc_a_1_n, i_nn_t_1_n, i_nn_a_1_n])
            gf_tv_1 = gf_tv([i_hc_t_1_n, i_hc_v_1_n, i_nn_t_1_n, i_nn_v_1_n])
            gf_av_1 = gf_av([i_hc_a_1_n, i_hc_v_1_n, i_nn_a_1_n, i_nn_v_1_n])

            concat_1 = Concat()((gf_ta_1, gf_tv_1, gf_av_1))

            dense = tf.keras.layers.Dense(50, activation="relu", name="dense")(concat_1)

            dense = tf.keras.layers.Dense(5, activation="sigmoid", name="dence_cl")(dense)

            model = tf.keras.Model(inputs=[i_hc_t_1, i_nn_t_1, i_hc_a_1, i_nn_a_1, i_hc_v_1, i_nn_v_1], outputs=dense)

            if show_summary and out:
                model.summary()

            return model

    def __load_av_model_b5(self, show_summary: bool = False, out: bool = True) -> Optional[tf.keras.Model]:
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

                from oceanai.modules.lab.prediction import Prediction

                pred = Prediction()

                pred._Prediction__load_av_model_b5(
                    show_summary = True, out = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                Model: "model"
                _________________________________________________________________
                 Layer (type)                Output Shape              Param #
                =================================================================
                 input_1 (InputLayer)        [(None, 64)]              0

                 dense_1 (Dense)             (None, 1)                 65

                 activ_1 (Activation)        (None, 1)                 0

                =================================================================
                Total params: 65
                Trainable params: 65
                Non-trainable params: 0
                _________________________________________________________________
                <tf.keras.Model at 0x147892ee0>

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.prediction import Prediction

                pred = Prediction()

                pred._Prediction__load_av_model_b5(
                    show_summary = True, out = []
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-17 10:53:03] Неверные типы или значения аргументов в "Prediction.__load_av_model_b5" ...
        """

        try:
            # Проверка аргументов
            if type(show_summary) is not bool or type(out) is not bool:
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__load_av_model_b5.__name__, out=out)
            return None
        else:
            input_1 = tf.keras.Input(shape=(64,), name="input_1")
            x = tf.keras.layers.Dense(units=1, name="dense_1")(input_1)
            x = tf.keras.layers.Activation("sigmoid", name="activ_1")(x)

            model = tf.keras.Model(inputs=input_1, outputs=x)

            if show_summary and out:
                model.summary()

            return model

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def load_av_models_b5(
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

                from oceanai.modules.lab.prediction import Prediction

                pred = Prediction()

                pred.load_av_models_b5(
                    show_summary = True, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-12-08 15:19:30] Формирование нейросетевых архитектур моделей для получения результатов оценки персональных качеств (мультимодальное объединение) ...

                Model: "model_4"
                _________________________________________________________________
                 Layer (type)                Output Shape              Param #
                =================================================================
                 input_1 (InputLayer)        [(None, 64)]              0

                 dense_1 (Dense)             (None, 1)                 65

                 activ_1 (Activation)        (None, 1)                 0

                =================================================================
                Total params: 65
                Trainable params: 65
                Non-trainable params: 0
                _________________________________________________________________
                --- Время выполнения: 0.141 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.prediction import Prediction

                pred = Prediction()

                pred.load_av_models_b5(
                    show_summary = 1, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-12-08 15:20:36] Неверные типы или значения аргументов в "Prediction.load_av_models_b5" ...

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
            self._inv_args(__class__.__name__, self.load_av_models_b5.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._formation_av_models_b5, last=False, out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            for key, _ in self._av_models_b5.items():
                self._av_models_b5[key] = self.__load_av_model_b5()

            if show_summary and out:
                self._av_models_b5[key].summary()

            if runtime:
                self._r_end(out=out)

            return True

    def load_avt_model_b5(
        self, show_summary: bool = False, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Формирование нейросетевой архитектуры модели для получения оценок персональных качеств

        Args:
            show_summary (bool): Отображение последней сформированной нейросетевой архитектуры моделей
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если нейросетевая архитектура модели сформирована, в обратном случае **False**
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
            self._inv_args(__class__.__name__, self.load_avt_model_b5.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._formation_avt_model_b5, last=False, out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            self._avt_model_b5 = self.__load_avt_model_b5()

            if show_summary and out:
                self._avt_model_b5.summary()

            if runtime:
                self._r_end(out=out)

            return True

    def load_av_models_weights_b5(
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
            url_non_neuroticism (str): Полный путь к файлу с весами нейросетевой модели ('эмоциональная стабильность')
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

                from oceanai.modules.lab.prediction import Prediction

                pred = Prediction()

                pred.load_av_models_b5(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-12-08 16:56:37] Формирование нейросетевых архитектур моделей для получения результатов оценки персональных качеств (мультимодальное объединение) ...

                --- Время выполнения: 0.075 сек. ---

                True

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                pred.path_to_save_ = './models'
                pred.chunk_size_ = 2000000

                url_openness = pred.weights_for_big5_['av']['b5']['openness']['sberdisk']
                url_conscientiousness = pred.weights_for_big5_['av']['b5']['conscientiousness']['sberdisk']
                url_extraversion = pred.weights_for_big5_['av']['b5']['extraversion']['sberdisk']
                url_agreeableness = pred.weights_for_big5_['av']['b5']['agreeableness']['sberdisk']
                url_non_neuroticism = pred.weights_for_big5_['av']['b5']['non_neuroticism']['sberdisk']

                pred.load_av_models_weights_b5(
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

                [2022-12-08 17:03:18] Загрузка весов нейросетевых моделей для получения результатов оценки персональных качеств (мультимодальное объединение) ...

                [2022-12-08 17:03:21] Загрузка файла "weights_2022-08-28_11-14-35.h5" (100.0%) ... Открытость опыту

                [2022-12-08 17:03:21] Загрузка файла "weights_2022-08-28_11-08-10.h5" (100.0%) ... Добросовестность

                [2022-12-08 17:03:21] Загрузка файла "weights_2022-08-28_11-17-57.h5" (100.0%) ... Экстраверсия

                [2022-12-08 17:03:21] Загрузка файла "weights_2022-08-28_11-25-11.h5" (100.0%) ... Доброжелательность

                [2022-12-08 17:03:21] Загрузка файла "weights_2022-06-14_21-44-09.h5" (100.0%) ... Эмоциональная стабильность

                --- Время выполнения: 3.399 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.prediction import Prediction

                pred = Prediction()

                pred.path_to_save_ = './models'
                pred.chunk_size_ = 2000000

                url_openness = pred.weights_for_big5_['av']['b5']['openness']['sberdisk']
                url_conscientiousness = pred.weights_for_big5_['av']['b5']['conscientiousness']['sberdisk']
                url_extraversion = pred.weights_for_big5_['av']['b5']['extraversion']['sberdisk']
                url_agreeableness = pred.weights_for_big5_['av']['b5']['agreeableness']['sberdisk']
                url_non_neuroticism = pred.weights_for_big5_['av']['b5']['non_neuroticism']['sberdisk']

                pred.load_av_models_weights_b5(
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

                [2022-12-08 17:05:32] Загрузка весов нейросетевых моделей для получения результатов оценки персональных качеств (мультимодальное объединение) ...

                [2022-12-08 17:05:32] Загрузка файла "weights_2022-08-28_11-14-35.h5" (100.0%) ...

                [2022-12-08 17:05:33] Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ... Открытость опыту

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/prediction.py
                    Линия: 639
                    Метод: load_av_models_weights_b5
                    Тип ошибки: AttributeError

                [2022-12-08 17:05:33] Загрузка файла "weights_2022-08-28_11-08-10.h5" (100.0%) ...

                [2022-12-08 17:05:33] Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ... Добросовестность

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/prediction.py
                    Линия: 639
                    Метод: load_av_models_weights_b5
                    Тип ошибки: AttributeError

                [2022-12-08 17:05:33] Загрузка файла "weights_2022-08-28_11-17-57.h5" (100.0%) ...

                [2022-12-08 17:05:33] Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ... Экстраверсия

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/prediction.py
                    Линия: 639
                    Метод: load_av_models_weights_b5
                    Тип ошибки: AttributeError

                [2022-12-08 17:05:33] Загрузка файла "weights_2022-08-28_11-25-11.h5" (100.0%) ...

                [2022-12-08 17:05:33] Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ... Доброжелательность

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/prediction.py
                    Линия: 639
                    Метод: load_av_models_weights_b5
                    Тип ошибки: AttributeError

                [2022-12-08 17:05:33] Загрузка файла "weights_2022-06-14_21-44-09.h5" (100.0%) ...

                [2022-12-08 17:05:33] Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ... Эмоциональная стабильность

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/prediction.py
                    Линия: 639
                    Метод: load_av_models_weights_b5
                    Тип ошибки: AttributeError

                --- Время выполнения: 1.024 сек. ---

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
            self._inv_args(__class__.__name__, self.load_av_models_weights_b5.__name__, out=out)
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
            self._info(self._load_av_models_weights_b5, last=False, out=out)

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
                            self._av_models_b5[self._b5["en"][cnt]].load_weights(self._url_last_filename)
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

    def get_av_union_predictions(
        self,
        depth: int = 1,
        recursive: bool = False,
        sr: int = 44100,
        window_audio: Union[int, float] = 2.0,
        step_audio: Union[int, float] = 1.0,
        reduction_fps: int = 5,
        window_video: int = 10,
        step_video: int = 5,
        lang: str = "ru",
        accuracy: bool = True,
        url_accuracy: str = "",
        logs: bool = True,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> bool:
        """Получения прогнозов по аудио и видео (мультимодальное объединение)

        Args:
            depth (int): Глубина иерархии для получения данных
            recursive (bool): Рекурсивный поиск данных
            sr (int): Частота дискретизации
            window_audio (Union[int, float]): Размер окна сегмента аудио сигнала (в секундах)
            step_audio (Union[int, float]): Шаг сдвига окна сегмента аудио сигнала (в секундах)
            reduction_fps (int): Понижение кадровой частоты
            window_video (int): Размер окна сегмента видео сигнала (в кадрах)
            step_video (int): Шаг сдвига окна сегмента видео сигнала (в кадрах)
            lang (str): Язык
            accuracy (bool): Вычисление точности
            url_accuracy (str): Полный путь к файлу с верными предсказаниями для подсчета точности
            logs (bool): При необходимости формировать LOG файл
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если прогнозы успешно получены, в обратном случае **False**

        :bdg-link-light:`Пример <../../user_guide/notebooks/Prediction-get_av_union_predictions.ipynb>`
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
                or type(recursive) is not bool
                or type(sr) is not int
                or sr < 1
                or (
                    (type(window_audio) is not int or window_audio < 1)
                    and (type(window_audio) is not float or window_audio <= 0)
                )
                or (
                    (type(step_audio) is not int or step_audio < 1)
                    and (type(step_audio) is not float or step_audio <= 0)
                )
                or type(reduction_fps) is not int
                or reduction_fps < 1
                or type(window_video) is not int
                or window_video < 1
                or type(step_video) is not int
                or step_video < 1
                or not isinstance(lang, str)
                or lang not in self.lang_traslate
                or type(accuracy) is not bool
                or type(url_accuracy) is not str
                or type(logs) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.get_av_union_predictions.__name__, out=out)
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
                    get_av_union_predictions_info = self._get_union_predictions_info + self._get_accuracy_info
                else:
                    get_av_union_predictions_info = self._get_union_predictions_info

                get_av_union_predictions_info += self._av_modality

                # Вычисление точности
                if accuracy is True:
                    # Информационное сообщение
                    self._info(get_av_union_predictions_info, out=out)

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
                            get_av_union_predictions_info,
                            i,
                            self.__local_path(curr_path),
                            self.__len_paths,
                            True,
                            last,
                            out,
                        )

                        # Извлечение признаков из акустического сигнала
                        hc_audio_features, melspectrogram_audio_features = self._get_acoustic_features(
                            path=str(curr_path.resolve()),
                            sr=sr,
                            window=window_audio,
                            step=step_audio,
                            last=True,
                            out=False,
                            runtime=False,
                            run=run,
                        )

                        # Извлечение признаков из визуального сигнала
                        hc_video_features, nn_video_features = self._get_visual_features(
                            path=str(curr_path.resolve()),
                            reduction_fps=reduction_fps,
                            window=window_video,
                            step=step_video,
                            lang=lang,
                            last=True,
                            out=False,
                            runtime=False,
                            run=run,
                        )

                        # Признаки из акустического сигнала извлечены
                        if (
                            type(hc_audio_features) is list
                            and type(melspectrogram_audio_features) is list
                            and type(hc_video_features) is np.ndarray
                            and type(nn_video_features) is np.ndarray
                            and len(hc_audio_features) > 0
                            and len(melspectrogram_audio_features) > 0
                            and len(hc_video_features) > 0
                            and len(nn_video_features) > 0
                        ):
                            # Коды ошибок нейросетевых моделей (аудио модальность)
                            code_error_pred_hc_audio = -1
                            code_error_pred_melspectrogram_audio = -1

                            try:
                                # Оправка экспертных признаков в нейросетевую модель
                                pred_hc_audio, _ = self.audio_model_hc_(np.array(hc_audio_features, dtype=np.float16))
                            except TypeError:
                                code_error_pred_hc_audio = 1
                            except Exception:
                                code_error_pred_melspectrogram_audio = 2

                            try:
                                # Отправка нейросетевых признаков в нейросетевую модель
                                pred_melspectrogram_audio, _ = self.audio_model_nn_(
                                    np.array(melspectrogram_audio_features, dtype=np.float16)
                                )
                            except TypeError:
                                code_error_pred_melspectrogram_audio = 1
                            except Exception:
                                code_error_pred_melspectrogram_audio = 2

                            if code_error_pred_hc_audio != -1 and code_error_pred_melspectrogram_audio != -1:
                                self._error(self._models_audio_not_formation, out=out)
                                return False

                            if code_error_pred_hc_audio != -1:
                                self._error(self._model_audio_hc_not_formation, out=out)
                                return False

                            if code_error_pred_melspectrogram_audio != -1:
                                self._error(self._model_audio_nn_not_formation, out=out)
                                return False

                            # Коды ошибок нейросетевых моделей (видео модальность)
                            code_error_pred_hc_video = -1
                            code_error_pred_nn_video = -1

                            try:
                                # Оправка экспертных признаков в нейросетевую модель
                                pred_hc_video, _ = self.video_model_hc_(np.array(hc_video_features, dtype=np.float16))
                            except TypeError:
                                code_error_pred_hc_video = 1
                            except Exception:
                                code_error_pred_hc_video = 2

                            try:
                                # Отправка нейросетевых признаков в нейросетевую модель
                                pred_nn_video, _ = self.video_model_nn_(np.array(nn_video_features, dtype=np.float16))
                            except TypeError:
                                code_error_pred_nn_video = 1
                            except Exception:
                                code_error_pred_nn_video = 2

                            if code_error_pred_hc_video != -1 and code_error_pred_nn_video != -1:
                                self._error(self._models_video_not_formation, out=out)
                                return False

                            if code_error_pred_hc_video != -1:
                                self._error(self._model_video_hc_not_formation, out=out)
                                return False

                            if code_error_pred_nn_video != -1:
                                self._error(self._model_video_nn_not_formation, out=out)
                                return False

                            # Конкатенация оценок по экспертным и нейросетевым признакам
                            union_pred = self.__concat_pred_av(
                                pred_hc_audio.numpy(),
                                pred_melspectrogram_audio.numpy(),
                                pred_hc_video.numpy(),
                                pred_nn_video.numpy(),
                                out=out,
                            )

                            if len(union_pred) == 0:
                                return False

                            final_pred = []

                            for cnt, (name_b5, model) in enumerate(self.av_models_b5_.items()):
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

                            self._del_last_el_notebook_history_output()

                    # Индикатор выполнения
                    self._progressbar_union_predictions(
                        get_av_union_predictions_info,
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
                            try:
                                mae_curr.append(
                                    mean_absolute_error(
                                        np.asarray(true_traits)[:, cnt], self._df_files[name_b5].to_list()
                                    )
                                )
                            except IndexError:
                                continue
                            except Exception:
                                continue

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

                        name_logs_file = self.get_av_union_predictions.__name__

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

    def load_avt_model_weights_b5(
        self,
        url: str,
        force_reload: bool = True,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> bool:
        """Загрузка весов нейросетевой модели для получения оценок персональных качеств

        Args:
            url (str): Полный путь к файлу с весами нейросетевой модели
            force_reload (bool): Принудительная загрузка файлов с весами нейросетевых моделей из сети
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если веса нейросетевой модели загружены, в обратном случае **False**
        """

        if runtime:
            self._r_start()

        if self.__load_model_weights(url, force_reload, self._load_avt_model_weights_b5, out, False, run) is True:
            try:
                self._avt_model_b5.load_weights(self._url_last_filename)
            except Exception:
                self._error(self._model_avt_not_formation, out=out)
                return False
            else:
                return True
            finally:
                if runtime:
                    self._r_end(out=out)

        return False

    def get_avt_predictions(
        self,
        depth: int = 1,
        recursive: bool = False,
        sr: int = 44100,
        window_audio: Union[int, float] = 2.0,
        step_audio: Union[int, float] = 1.0,
        reduction_fps: int = 5,
        window_video: int = 10,
        step_video: int = 5,
        asr: bool = False,
        lang: str = "ru",
        accuracy=True,
        url_accuracy: str = "",
        logs: bool = True,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> bool:
        """Получения прогнозов по аудио, видео и тексту (мультимодальное объединение)

        Args:
            depth (int): Глубина иерархии для получения данных
            recursive (bool): Рекурсивный поиск данных
            sr (int): Частота дискретизации
            window_audio (Union[int, float]): Размер окна сегмента аудио сигнала (в секундах)
            step_audio (Union[int, float]): Шаг сдвига окна сегмента аудио сигнала (в секундах)
            reduction_fps (int): Понижение кадровой частоты
            window_video (int): Размер окна сегмента видео сигнала (в кадрах)
            step_video (int): Шаг сдвига окна сегмента видео сигнала (в кадрах)
            asr (bool): Автоматическое распознавание речи
            lang (str): Язык
            accuracy (bool): Вычисление точности
            url_accuracy (str): Полный путь к файлу с верными предсказаниями для подсчета точности
            logs (bool): При необходимости формировать LOG файл
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если прогнозы успешно получены, в обратном случае **False**
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
                or type(recursive) is not bool
                or type(sr) is not int
                or sr < 1
                or (
                    (type(window_audio) is not int or window_audio < 1)
                    and (type(window_audio) is not float or window_audio <= 0)
                )
                or (
                    (type(step_audio) is not int or step_audio < 1)
                    and (type(step_audio) is not float or step_audio <= 0)
                )
                or type(reduction_fps) is not int
                or reduction_fps < 1
                or type(window_video) is not int
                or window_video < 1
                or type(step_video) is not int
                or step_video < 1
                or type(asr) is not bool
                or not isinstance(lang, str)
                or lang not in self.lang_traslate
                or type(accuracy) is not bool
                or type(url_accuracy) is not str
                or type(logs) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.get_avt_predictions.__name__, out=out)
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
                    get_avt_predictions_info = self._get_union_predictions_info + self._get_accuracy_info
                else:
                    get_avt_predictions_info = self._get_union_predictions_info

                get_avt_predictions_info += self._av_modality

                # Вычисление точности
                if accuracy is True:
                    # Информационное сообщение
                    self._info(get_avt_predictions_info, out=out)

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
                            get_avt_predictions_info,
                            i,
                            self.__local_path(curr_path),
                            self.__len_paths,
                            True,
                            last,
                            out,
                        )

                        # Извлечение признаков из акустического сигнала
                        hc_audio_features, melspectrogram_audio_features = self._get_acoustic_features(
                            path=str(curr_path.resolve()),
                            sr=sr,
                            window=window_audio,
                            step=step_audio,
                            last=True,
                            out=False,
                            runtime=False,
                            run=run,
                        )

                        # Извлечение признаков из визуального сигнала
                        hc_video_features, nn_video_features = self._get_visual_features(
                            path=str(curr_path.resolve()),
                            reduction_fps=reduction_fps,
                            window=window_video,
                            step=step_video,
                            lang=lang,
                            last=True,
                            out=False,
                            runtime=False,
                            run=run,
                        )

                        # Извлечение признаков из текста
                        hc_text_features, nn_text_features = self.get_text_features(
                            path=str(curr_path.resolve()),
                            asr=asr,
                            lang=lang,
                            show_text=False,
                            out=False,
                            runtime=False,
                            run=run,
                        )

                        hc_text_features = np.expand_dims(hc_text_features, axis=0)
                        nn_text_features = np.expand_dims(nn_text_features, axis=0)

                        if (
                            type(hc_audio_features) is list
                            and type(melspectrogram_audio_features) is list
                            and type(hc_video_features) is np.ndarray
                            and type(nn_video_features) is np.ndarray
                            and type(hc_text_features) is np.ndarray
                            and type(nn_text_features) is np.ndarray
                            and len(hc_audio_features) > 0
                            and len(melspectrogram_audio_features) > 0
                            and len(hc_video_features) > 0
                            and len(nn_video_features) > 0
                            and len(hc_text_features) > 0
                            and len(nn_text_features) > 0
                        ):
                            feature_lambda = lambda feature: np.concatenate(
                                (np.mean(feature, axis=0), np.std(feature, axis=0))
                            )

                            # Коды ошибок нейросетевых моделей (аудио модальность)
                            code_error_pred_hc_audio = -1
                            code_error_pred_melspectrogram_audio = -1

                            try:
                                # Оправка экспертных признаков в нейросетевую модель
                                _, features_hc_audio = self.audio_model_hc_(
                                    np.array(hc_audio_features, dtype=np.float16)
                                )
                            except TypeError:
                                code_error_pred_hc_audio = 1
                            except Exception:
                                code_error_pred_melspectrogram_audio = 2

                            try:
                                # Отправка нейросетевых признаков в нейросетевую модель
                                _, features_nn_audio = self.audio_model_nn_(
                                    np.array(melspectrogram_audio_features, dtype=np.float16)
                                )
                            except TypeError:
                                code_error_pred_melspectrogram_audio = 1
                            except Exception:
                                code_error_pred_melspectrogram_audio = 2

                            if code_error_pred_hc_audio != -1 and code_error_pred_melspectrogram_audio != -1:
                                self._error(self._models_audio_not_formation, out=out)
                                return False

                            if code_error_pred_hc_audio != -1:
                                self._error(self._model_audio_hc_not_formation, out=out)
                                return False

                            if code_error_pred_melspectrogram_audio != -1:
                                self._error(self._model_audio_nn_not_formation, out=out)
                                return False

                            features_hc_audio = np.expand_dims(feature_lambda(features_hc_audio.numpy()), axis=0)
                            features_nn_audio = np.expand_dims(feature_lambda(features_nn_audio.numpy()), axis=0)

                            # Коды ошибок нейросетевых моделей (видео модальность)
                            code_error_pred_hc_video = -1
                            code_error_pred_nn_video = -1

                            try:
                                # Оправка экспертных признаков в нейросетевую модель
                                _, features_hc_video = self.video_model_hc_(
                                    np.array(hc_video_features, dtype=np.float16)
                                )
                            except TypeError:
                                code_error_pred_hc_video = 1
                            except Exception:
                                code_error_pred_hc_video = 2

                            try:
                                # Отправка нейросетевых признаков в нейросетевую модель
                                _, features_nn_video = self.video_model_nn_(
                                    np.array(nn_video_features, dtype=np.float16)
                                )
                            except TypeError:
                                code_error_pred_nn_video = 1
                            except Exception:
                                code_error_pred_nn_video = 2

                            if code_error_pred_hc_video != -1 and code_error_pred_nn_video != -1:
                                self._error(self._models_video_not_formation, out=out)
                                return False

                            if code_error_pred_hc_video != -1:
                                self._error(self._model_video_hc_not_formation, out=out)
                                return False

                            if code_error_pred_nn_video != -1:
                                self._error(self._model_video_nn_not_formation, out=out)
                                return False

                            features_hc_video = np.expand_dims(feature_lambda(features_hc_video.numpy()), axis=0)
                            features_nn_video = np.expand_dims(feature_lambda(features_nn_video.numpy()), axis=0)

                            # Коды ошибок нейросетевых моделей (текст)
                            code_error_pred_hc_text = -1
                            code_error_pred_nn_text = -1

                            try:
                                # Оправка экспертных признаков в нейросетевую модель
                                _, features_hc_text = self.text_model_hc_(np.array(hc_text_features, dtype=np.float16))
                            except TypeError:
                                code_error_pred_hc_text = 1
                            except Exception:
                                code_error_pred_hc_text = 2

                            try:
                                # Отправка нейросетевых признаков в нейросетевую модель
                                _, features_nn_text = self.text_model_nn_(np.array(nn_text_features, dtype=np.float16))
                            except TypeError:
                                code_error_pred_nn_text = 1
                            except Exception:
                                code_error_pred_nn_text = 2

                            if code_error_pred_hc_text != -1 and code_error_pred_nn_text != -1:
                                self._error(self._model_text_not_formation, out=out)
                                return False

                            if code_error_pred_hc_text != -1:
                                self._error(self._model_text_hc_not_formation, out=out)
                                return False

                            if code_error_pred_nn_text != -1:
                                self._error(self._model_text_nn_not_formation, out=out)
                                return False

                            try:
                                final_pred = (
                                    self.avt_model_b5_(
                                        [
                                            features_hc_text.numpy(),
                                            features_nn_text.numpy(),
                                            features_hc_audio,
                                            features_nn_audio,
                                            features_hc_video,
                                            features_nn_video,
                                        ]
                                    )
                                    .numpy()[0]
                                    .tolist()
                                )
                            except Exception:
                                self._other_error(self._unknown_err, out=out)
                                return False

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

                            self._del_last_el_notebook_history_output()

                    # Индикатор выполнения
                    self._progressbar_union_predictions(
                        get_avt_predictions_info,
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
                            try:
                                mae_curr.append(
                                    mean_absolute_error(
                                        np.asarray(true_traits)[:, cnt], self._df_files[name_b5].to_list()
                                    )
                                )
                            except IndexError:
                                continue
                            except Exception:
                                continue

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

                        name_logs_file = self.get_avt_predictions.__name__

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

    def get_avt_predictions_gradio(
        self,
        paths: list[str] = [],
        depth: int = 1,
        recursive: bool = False,
        sr: int = 44100,
        window_audio: Union[int, float] = 2.0,
        step_audio: Union[int, float] = 1.0,
        reduction_fps: int = 5,
        window_video: int = 10,
        step_video: int = 5,
        asr: bool = False,
        lang: str = "ru",
        accuracy=True,
        url_accuracy: str = "",
        logs: bool = True,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> bool:
        """Получения прогнозов по аудио, видео и тексту (мультимодальное объединение)

        Args:
            depth (int): Глубина иерархии для получения данных
            recursive (bool): Рекурсивный поиск данных
            sr (int): Частота дискретизации
            window_audio (Union[int, float]): Размер окна сегмента аудио сигнала (в секундах)
            step_audio (Union[int, float]): Шаг сдвига окна сегмента аудио сигнала (в секундах)
            reduction_fps (int): Понижение кадровой частоты
            window_video (int): Размер окна сегмента видео сигнала (в кадрах)
            step_video (int): Шаг сдвига окна сегмента видео сигнала (в кадрах)
            asr (bool): Автоматическое распознавание речи
            lang (str): Язык
            accuracy (bool): Вычисление точности
            url_accuracy (str): Полный путь к файлу с верными предсказаниями для подсчета точности
            logs (bool): При необходимости формировать LOG файл
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если прогнозы успешно получены, в обратном случае **False**
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
                or type(recursive) is not bool
                or type(sr) is not int
                or sr < 1
                or (
                    (type(window_audio) is not int or window_audio < 1)
                    and (type(window_audio) is not float or window_audio <= 0)
                )
                or (
                    (type(step_audio) is not int or step_audio < 1)
                    and (type(step_audio) is not float or step_audio <= 0)
                )
                or type(reduction_fps) is not int
                or reduction_fps < 1
                or type(window_video) is not int
                or window_video < 1
                or type(step_video) is not int
                or step_video < 1
                or type(asr) is not bool
                or not isinstance(lang, str)
                or lang not in self.lang_traslate
                or type(accuracy) is not bool
                or type(url_accuracy) is not str
                or type(logs) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.get_avt_predictions.__name__, out=out)
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
                    get_avt_predictions_info = self._get_union_predictions_info + self._get_accuracy_info
                else:
                    get_avt_predictions_info = self._get_union_predictions_info

                get_avt_predictions_info += self._av_modality

                # Вычисление точности
                if accuracy is True:
                    # Информационное сообщение
                    self._info(get_avt_predictions_info, out=out)

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

                last = False  # Замена последнего сообщения

                paths = [i for i in paths if i.endswith((".mp4", ".avi", "mov", "flv"))]
                self.__len_paths = len(paths)

                # Проход по всем искомым файлов
                for i, curr_path in enumerate(paths):
                    if i != 0:
                        last = True

                    # Извлечение признаков из акустического сигнала
                    hc_audio_features, melspectrogram_audio_features = self._get_acoustic_features(
                        path=curr_path,
                        sr=sr,
                        window=window_audio,
                        step=step_audio,
                        last=True,
                        out=True,
                        runtime=False,
                        run=run,
                    )

                    # Извлечение признаков из визуального сигнала
                    hc_video_features, nn_video_features = self._get_visual_features(
                        path=curr_path,
                        reduction_fps=reduction_fps,
                        window=window_video,
                        step=step_video,
                        lang=lang,
                        last=True,
                        out=False,
                        runtime=False,
                        run=run,
                    )

                    # Извлечение признаков из текста
                    hc_text_features, nn_text_features = self.get_text_features(
                        path=curr_path,
                        asr=asr,
                        lang=lang,
                        show_text=False,
                        out=False,
                        runtime=False,
                        run=run,
                    )

                    hc_text_features = np.expand_dims(hc_text_features, axis=0)
                    nn_text_features = np.expand_dims(nn_text_features, axis=0)

                    if (
                        type(hc_audio_features) is list
                        and type(melspectrogram_audio_features) is list
                        and type(hc_video_features) is np.ndarray
                        and type(nn_video_features) is np.ndarray
                        and type(hc_text_features) is np.ndarray
                        and type(nn_text_features) is np.ndarray
                        and len(hc_audio_features) > 0
                        and len(melspectrogram_audio_features) > 0
                        and len(hc_video_features) > 0
                        and len(nn_video_features) > 0
                        and len(hc_text_features) > 0
                        and len(nn_text_features) > 0
                    ):
                        feature_lambda = lambda feature: np.concatenate(
                            (np.mean(feature, axis=0), np.std(feature, axis=0))
                        )

                        # Коды ошибок нейросетевых моделей (аудио модальность)
                        code_error_pred_hc_audio = -1
                        code_error_pred_melspectrogram_audio = -1

                        try:
                            # Оправка экспертных признаков в нейросетевую модель
                            _, features_hc_audio = self.audio_model_hc_(np.array(hc_audio_features, dtype=np.float16))
                        except TypeError:
                            code_error_pred_hc_audio = 1
                        except Exception:
                            code_error_pred_melspectrogram_audio = 2

                        try:
                            # Отправка нейросетевых признаков в нейросетевую модель
                            _, features_nn_audio = self.audio_model_nn_(
                                np.array(melspectrogram_audio_features, dtype=np.float16)
                            )
                        except TypeError:
                            code_error_pred_melspectrogram_audio = 1
                        except Exception:
                            code_error_pred_melspectrogram_audio = 2

                        if code_error_pred_hc_audio != -1 and code_error_pred_melspectrogram_audio != -1:
                            self._error(self._models_audio_not_formation, out=out)
                            return False

                        if code_error_pred_hc_audio != -1:
                            self._error(self._model_audio_hc_not_formation, out=out)
                            return False

                        if code_error_pred_melspectrogram_audio != -1:
                            self._error(self._model_audio_nn_not_formation, out=out)
                            return False

                        features_hc_audio = np.expand_dims(feature_lambda(features_hc_audio.numpy()), axis=0)
                        features_nn_audio = np.expand_dims(feature_lambda(features_nn_audio.numpy()), axis=0)

                        # Коды ошибок нейросетевых моделей (видео модальность)
                        code_error_pred_hc_video = -1
                        code_error_pred_nn_video = -1

                        try:
                            # Оправка экспертных признаков в нейросетевую модель
                            _, features_hc_video = self.video_model_hc_(np.array(hc_video_features, dtype=np.float16))
                        except TypeError:
                            code_error_pred_hc_video = 1
                        except Exception:
                            code_error_pred_hc_video = 2

                        try:
                            # Отправка нейросетевых признаков в нейросетевую модель
                            _, features_nn_video = self.video_model_nn_(np.array(nn_video_features, dtype=np.float16))
                        except TypeError:
                            code_error_pred_nn_video = 1
                        except Exception:
                            code_error_pred_nn_video = 2

                        if code_error_pred_hc_video != -1 and code_error_pred_nn_video != -1:
                            self._error(self._models_video_not_formation, out=out)
                            return False

                        if code_error_pred_hc_video != -1:
                            self._error(self._model_video_hc_not_formation, out=out)
                            return False

                        if code_error_pred_nn_video != -1:
                            self._error(self._model_video_nn_not_formation, out=out)
                            return False

                        features_hc_video = np.expand_dims(feature_lambda(features_hc_video.numpy()), axis=0)
                        features_nn_video = np.expand_dims(feature_lambda(features_nn_video.numpy()), axis=0)

                        # Коды ошибок нейросетевых моделей (текст)
                        code_error_pred_hc_text = -1
                        code_error_pred_nn_text = -1

                        try:
                            # Оправка экспертных признаков в нейросетевую модель
                            _, features_hc_text = self.text_model_hc_(np.array(hc_text_features, dtype=np.float16))
                        except TypeError:
                            code_error_pred_hc_text = 1
                        except Exception:
                            code_error_pred_hc_text = 2

                        try:
                            # Отправка нейросетевых признаков в нейросетевую модель
                            _, features_nn_text = self.text_model_nn_(np.array(nn_text_features, dtype=np.float16))
                        except TypeError:
                            code_error_pred_nn_text = 1
                        except Exception:
                            code_error_pred_nn_text = 2

                        if code_error_pred_hc_text != -1 and code_error_pred_nn_text != -1:
                            self._error(self._model_text_not_formation, out=out)
                            return False

                        if code_error_pred_hc_text != -1:
                            self._error(self._model_text_hc_not_formation, out=out)
                            return False

                        if code_error_pred_nn_text != -1:
                            self._error(self._model_text_nn_not_formation, out=out)
                            return False

                        try:
                            final_pred = (
                                self.avt_model_b5_(
                                    [
                                        features_hc_text.numpy(),
                                        features_nn_text.numpy(),
                                        features_hc_audio,
                                        features_nn_audio,
                                        features_hc_video,
                                        features_nn_video,
                                    ]
                                )
                                .numpy()[0]
                                .tolist()
                            )
                        except Exception:
                            self._other_error(self._unknown_err, out=out)
                            return False

                        # Добавление данных в словарь для DataFrame
                        if self._append_to_list_of_files(curr_path, final_pred, out) is False:
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
                        if self._append_to_list_of_files(curr_path, [None] * len(self._b5["en"]), out) is False:
                            return False

                        self._del_last_el_notebook_history_output()

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
                        try:
                            mae_curr.append(
                                mean_absolute_error(np.asarray(true_traits)[:, cnt], self._df_files[name_b5].to_list())
                            )
                        except IndexError:
                            continue
                        except Exception:
                            continue

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
                    self._df_accuracy = pd.DataFrame.from_dict(data=self._dict_of_accuracy, orient="index").transpose()
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

                    name_logs_file = self.get_avt_predictions.__name__

                    # Сохранение LOG
                    res_save_logs_df_files = self._save_logs(self._df_files, name_logs_file + "_df_files_" + curr_ts)

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
