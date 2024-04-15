#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Видео
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
import math
import gradio

from urllib.parse import urlparse
from urllib.error import URLError
from pathlib import Path  # Работа с путями в файловой системе
from scipy.spatial import distance
from scipy import stats

# from pymediainfo import MediaInfo  # Получение meta данных из медиафайлов
from datetime import datetime  # Работа со временем
from sklearn.metrics import mean_absolute_error

# Типы данных
from typing import Dict, List, Tuple, Union, Optional, Callable
from types import ModuleType

from IPython.display import clear_output

# Персональные
from oceanai.modules.lab.download import Download  # Загрузка файлов

# Порог регистрации сообщений TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # Машинное обучение от Google
import keras
import cv2
import mediapipe as mp  # Набор нейросетевых моделей и решений для компьютерного зрения

# Исправленная версия Keras_VGGFace
from oceanai.modules.lab.keras_vggface import utils
from oceanai.modules.lab.keras_vggface.vggface import VGGFace

mp.solutions.face_mesh.FaceMesh()  # Удаление сообщения: INFO: Created TensorFlow Lite XNNPACK delegate for CPU)


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class VideoMessages(Download):
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

        self._video_modality: str = self._(" (видео модальность) ...")
        self._formation_video_model_hc: str = self._formation_model_hc + self._video_modality
        self._formation_video_model_nn: str = self._formation_model_nn + self._video_modality
        self._formation_video_deep_fe: str = (
            self._("Формирование нейросетевой архитектуры для получения нейросетевых " "признаков")
            + self._video_modality
        )
        self._formation_video_models_b5: str = self._formation_models_b5 + self._video_modality

        self._load_video_model_weights_hc: str = self._load_model_weights_hc + self._video_modality
        self._load_video_model_weights_nn: str = self._load_model_weights_nn + self._video_modality
        self._load_video_model_weights_deep_fe: str = (
            self._("Загрузка весов нейросетевой модели для получения " "нейросетевых признаков") + self._video_modality
        )
        self._load_video_models_weights_b5: str = self._load_models_weights_b5 + self._video_modality

        self._model_video_hc_not_formation: str = self._model_hc_not_formation + self._video_modality
        self._model_video_nn_not_formation: str = self._model_nn_not_formation + self._video_modality
        self._model_video_deep_fe_not_formation: str = (
            self._oh
            + self._("нейросетевая архитектура модели для " "получения нейросетевых признаков не " "сформирована")
            + self._video_modality
        )
        self._models_video_not_formation: str = self._models_not_formation + self._video_modality

        self._get_visual_feature_info: str = self._(
            "Извлечение признаков (экспертных и нейросетевых) из визуального " "сигнала ..."
        )

        self._wrong_extension_video_formats = self._oh + self._('расширение видеофайла должно быть одним из: "{}"')
        self._all_frames_is_zero: str = self._oh + self._("общее количество кадров в видеопотоке: {} ...")
        self._calc_reshape_img_coef_error: str = self._oh + self._(
            "вычисление коэффициента изменения размера " "изображения не произведено ..."
        )

        self._faces_not_found: str = self._oh + self._("не на одном кадре видеопотока лицо не найдено ...")

        self._concat_video_pred_error: str = self._concat_pred_error + self._video_modality
        self._norm_video_pred_error: str = self._norm_pred_error + self._video_modality


# ######################################################################################################################
# Аудио
# ######################################################################################################################
@dataclass
class Video(VideoMessages):
    """Класс для обработки видео

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
        self._video_model_hc: Optional[tf.keras.Model] = None
        # Нейросетевая модель **tf.keras.Model** для получения нейросетевых признаков
        self._video_model_deep_fe: Optional[tf.keras.Model] = None
        # Нейросетевая модель **tf.keras.Model** для получения оценок по нейросетевым признакам
        self._video_model_nn: Optional[tf.keras.Model] = None
        # Нейросетевые модели **tf.keras.Model** для получения результатов оценки персональных качеств
        self._video_models_b5: Dict[str, Optional[tf.keras.Model]] = dict(
            zip(self._b5["en"], [None] * len(self._b5["en"]))
        )

        # ----------------------- Только для внутреннего использования внутри класса

        # Поддерживаемые видео форматы
        self.__supported_video_formats: List[str] = ["mp4", "mov", "avi", "flv"]

        self.__mp_face_mesh: ModuleType = mp.solutions.face_mesh  # 468 3D-ориентиров лица
        self.__mp_drawing: ModuleType = mp.solutions.drawing_utils  # Утилиты MediaPipe

        self.__bndbox_face_size: List[int] = [224, 224]  # Размер изображения с лицом

        self.__lang_traslate: List[str] = ["ru", "en"]
        self.lang_traslate: List[str] = self.__lang_traslate

        # Используемые координаты ориентиров лица
        self.__coords_face_mesh_fi: List[int] = [
            0,
            1,
            386,
            133,
            6,
            8,
            267,
            13,
            14,
            17,
            145,
            276,
            152,
            282,
            411,
            285,
            159,
            291,
            37,
            299,
            46,
            52,
            55,
            187,
            61,
            69,
            331,
            334,
            336,
            102,
            105,
            362,
            107,
            374,
            33,
            263,
        ]

        self.__coords_face_mesh_mupta: List[int] = [
            0,
            1,
            386,
            133,
            6,
            8,
            267,
            13,
            14,
            17,
            274,
            145,
            276,
            152,
            282,
            411,
            285,
            159,
            291,
            37,
            299,
            46,
            52,
            55,
            187,
            61,
            69,
            331,
            334,
            336,
            102,
            105,
            362,
        ]

        self.__couples_face_mesh_fi: List[List[int]] = [
            [133, 46],
            [133, 52],
            [133, 55],
            [362, 285],
            [362, 282],
            [362, 276],
            [55, 285],
            [1, 6],
            [8, 6],
            [0, 1],
            [0, 17],
            [61, 291],
            [0, 13],
            [61, 291],
            [37, 13],
            [267, 13],
            [13, 14],
            [17, 152],
            [102, 331],
            [102, 133],
            [331, 362],
            [291, 362],
            [61, 133],
            [386, 374],
            [159, 145],
            [69, 105],
            [69, 107],
            [299, 336],
            [299, 334],
            [187, 133],
            [411, 362],
        ]

        self.__couples_face_mesh_mupta: List[List[int]] = [
            [133, 46],
            [133, 52],
            [133, 55],
            [362, 285],
            [362, 282],
            [362, 276],
            [55, 285],
            [1, 6],
            [8, 6],
            [0, 1],
            [0, 17],
            [61, 291],
            [0, 13],
            [61, 291],
            [37, 13],
            [267, 13],
            [13, 14],
            [17, 152],
            [102, 331],
            [102, 133],
            [331, 362],
            [291, 362],
            [61, 133],
            [386, 274],
            [159, 145],
            [69, 105],
            [69, 145],
            [299, 336],
            [299, 334],
            [187, 133],
            [411, 362],
        ]

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
    def video_model_hc_(self) -> Optional[tf.keras.Model]:
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

                from oceanai.modules.lab.video import Video

                video = Video()

                video.load_video_model_hc(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

                video.video_model_hc_

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-26 12:37:42] Формирование нейросетевой архитектуры модели для получения оценок по экспертным признакам (видео модальность) ...

                --- Время выполнения: 1.112 сек. ---

                <tf.keras.Model at 0x1434eb1f0>

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                video.video_model_hc_

            .. output-cell::
                :execution-count: 2
                :linenos:


        """

        return self._video_model_hc

    @property
    def video_model_nn_(self) -> Optional[tf.keras.Model]:
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

                from oceanai.modules.lab.video import Video

                video = Video()

                video.load_video_model_nn(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

                video.video_model_nn_

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-27 14:49:00] Формирование нейросетевой архитектуры для получения оценок по нейросетевым признакам (видео модальность) ...

                --- Время выполнения: 1.986 сек. ---

                <tf.keras.Model at 0x13d5295b0>

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                video.video_model_nn_

            .. output-cell::
                :execution-count: 2
                :linenos:


        """

        return self._video_model_nn

    @property
    def video_model_deep_fe_(self) -> Optional[tf.keras.Model]:
        """Получение нейросетевой модели **tf.keras.Model** для получения нейросетевых признаков

        Returns:
            Optional[tf.keras.Model]: Нейросетевая модель **tf.keras.Model** или None

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                video.load_video_model_deep_fe(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

                video.video_model_deep_fe_

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-11-01 12:12:35] Формирование нейросетевой архитектуры для получения нейросетевых признаков (видео модальность) ...

                --- Время выполнения: 1.468 сек. ---

                <tf.keras.Model at 0x14e138100>

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                video.video_model_deep_fe_

            .. output-cell::
                :execution-count: 2
                :linenos:


        """

        return self._video_model_deep_fe

    @property
    def video_models_b5_(self) -> Dict[str, Optional[tf.keras.Model]]:
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

                from oceanai.modules.lab.video import Video

                video = Video()

                video.load_video_models_b5(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

                video.video_models_b5_

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-19 15:45:35] Формирование нейросетевых архитектур моделей для получения результатов оценки персональных качеств ...

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

                from oceanai.modules.lab.video import Video

                video = Video()

                video.video_models_b5_

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

        return self._video_models_b5

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

                from oceanai.modules.lab.video import Video

                video = Video()

                video.path_to_save_ = './models'
                video.chunk_size_ = 2000000

                video._Video__load_model_weights(
                    url = 'https://download.sberdisk.ru/download/file/412059444?token=JXerCfAjJZg6crD&filename=weights_2022-08-27_18-53-35.h5',
                    force_reload = True,
                    info_text = 'Загрузка весов нейросетевой модели',
                    out = True, runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-27 12:46:55] Загрузка весов нейросетевой модели

                [2022-10-27 12:46:55] Загрузка файла "weights_2022-08-27_18-53-35.h5" (100.0%) ...

                --- Время выполнения: 0.626 сек. ---

                True

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                video.path_to_save_ = './models'
                video.chunk_size_ = 2000000

                video._Video__load_model_weights(
                    url = './models/weights_2022-08-27_18-53-35.h5',
                    force_reload = True,
                    info_text = 'Загрузка весов нейросетевой модели',
                    out = True, runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-27 12:47:52] Загрузка весов нейросетевой модели

                --- Время выполнения: 0.002 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                video.path_to_save_ = './models'
                video.chunk_size_ = 2000000

                video._Video__load_model_weights(
                    url = 'https://download.sberdisk.ru/download/file/412059444?token=JXerCfAjJZg6crD&filename=weights_2022-08-27_18-53-35.h5',
                    force_reload = True, info_text = '',
                    out = True, runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-27 12:48:24] Неверные типы или значения аргументов в "Video.__load_model_weights" ...

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

    def __calc_reshape_img_coef(
        self, shape: Union[Tuple[int], List[int]], new_shape: Union[int, Tuple[int], List[int]], out: bool = True
    ) -> float:
        """Вычисление коэффициента изменения размера изображения

        .. note::
            private (приватный метод)

        Args:
            shape (Union[Tuple[int], List[int]]): Текущий размер изображения (ширина, высота)
            new_shape (Union[int, Tuple[int], List[int]]): Желаемый размер изображения
            out (bool): Отображение

        Returns:
            float: Коэффициент изменения размера изображения

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                video._Video__calc_reshape_img_coef(
                    shape = (1280, 720),
                    new_shape = 224,
                    out = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                0.175

            :bdg-success:`Верно` :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                video._Video__calc_reshape_img_coef(
                    shape = (1280, 720),
                    new_shape = (1920, 1080),
                    out = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                1.5

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.lab.video import Video

                video = Video()

                video._Video__calc_reshape_img_coef(
                    shape = (1280, 720),
                    new_shape = '',
                    out = True
                )

            .. output-cell::
                :execution-count: 4
                :linenos:

                [2022-10-29 13:24:27] Неверные типы или значения аргументов в "Video.__calc_reshape_img_coef" ...

                -1.0
        """

        try:
            # Проверка аргументов
            if (
                (isinstance(shape, list) is False and isinstance(shape, tuple) is False)
                or len(shape) != 2
                or (
                    isinstance(new_shape, list) is False
                    and isinstance(new_shape, tuple) is False
                    and type(new_shape) is not int
                )
                or type(out) is not bool
            ):
                raise TypeError

            if type(shape[0]) is not int or type(shape[1]) is not int:
                raise TypeError

            if shape[0] < 1 or shape[1] < 1:
                raise ValueError

            if isinstance(new_shape, list) is True or isinstance(new_shape, tuple) is True:
                if len(new_shape) != 2:
                    raise TypeError

                if type(new_shape[0]) is not int or type(new_shape[1]) is not int:
                    raise TypeError

                if new_shape[0] < 1 or new_shape[1] < 1:
                    raise ValueError
            else:
                if new_shape < 1:
                    raise ValueError
        except (TypeError, ValueError):
            self._inv_args(__class__.__name__, self.__calc_reshape_img_coef.__name__, out=out)
            return -1.0
        else:
            if isinstance(new_shape, list) is False and isinstance(new_shape, tuple) is False:
                new_shape = (new_shape, new_shape)
            return min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    def __norm_pred(self, pred_data: np.ndarray, len_nn: int = 16, out: bool = True) -> np.ndarray:
        """Нормализация оценок по экспертным и нейросетевым признакам

        .. note::
            private (приватный метод)

        Args:
            pred_data (np.ndarray): Оценки
            len_nn (int): Максимальный размер вектора оценок
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
                from oceanai.modules.lab.video import Video

                video = Video()

                arr = np.array([
                    [0.64113516, 0.6217892, 0.54451424, 0.6144415, 0.59334993],
                    [0.6652424, 0.63606125, 0.572305, 0.63169795, 0.612515]
                ])

                video._Video__norm_pred(
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
                from oceanai.modules.lab.video import Video

                video = Video()

                arr = np.array([])

                video._Video__norm_pred(
                    pred_data = arr,
                    len_nn = 4,
                    out = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-20 22:03:17] Неверные типы или значения аргументов в "Video.__norm_pred" ...

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
                self._other_error(self._norm_video_pred_error, last=False, out=out)
                return np.array([])
            except Exception:
                self._other_error(self._unknown_err, out=out)
                return np.array([])

    def __concat_pred(self, pred_hc: np.ndarray, pred_nn: np.ndarray, out: bool = True) -> List[Optional[np.ndarray]]:
        """Конкатенация оценок по экспертным и нейросетевым признакам

        .. note::
            private (приватный метод)

        Args:
            pred_hc (np.ndarray): Оценки по экспертным признакам
            pred_nn (np.ndarray): Оценки по нейросетевым признакам
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
                from oceanai.modules.lab.video import Video

                video = Video()

                arr_hc = np.array([
                    [0.64113516, 0.6217892, 0.54451424, 0.6144415, 0.59334993],
                    [0.6652424, 0.63606125, 0.572305, 0.63169795, 0.612515]
                ])

                arr_nn = np.array([
                    [0.56030345, 0.7488746, 0.44648764, 0.59893465, 0.5701077],
                    [0.5900006, 0.7652722, 0.4795154, 0.6409055, 0.6088242]
                ])

                video._Video__concat_pred(
                    pred_hc = arr_hc,
                    pred_nn = arr_nn,
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
                from oceanai.modules.lab.video import Video

                video = Video()

                arr_hc = np.array([
                    [0.64113516, 0.6217892, 0.54451424, 0.6144415],
                    [0.6652424, 0.63606125, 0.572305, 0.63169795, 0.612515]
                ])

                arr_nn = np.array([
                    [0.56030345, 0.7488746, 0.44648764, 0.59893465, 0.5701077],
                    [0.5900006, 0.7652722, 0.4795154, 0.6409055, 0.6088242]
                ])

                video._Video__concat_pred(
                    pred_hc = arr_hc,
                    pred_nn = arr_nn,
                    out = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-20 22:33:31] Ой! Что-то пошло не так ... конкатенация оценок по экспертным и нейросетевым
                признакам не произведена (видео модальность) ...

                []
        """

        try:
            # Проверка аргументов
            if (
                type(pred_hc) is not np.ndarray
                or len(pred_hc) < 1
                or type(pred_nn) is not np.ndarray
                or len(pred_nn) < 1
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__concat_pred.__name__, out=out)
            return []
        else:
            # Нормализация оценок по экспертным и нейросетевым признакам
            pred_hc_norm = self.__norm_pred(pred_hc, out=False)
            pred_nn_norm = self.__norm_pred(pred_nn, out=False)

            if len(pred_hc_norm) == 0 or len(pred_nn_norm) == 0:
                self._error(self._concat_video_pred_error, out=out)
                return []

            concat = []

            try:
                # Проход по всем персональным качествам личности человека
                for i in range(len(self._b5["en"])):
                    concat.append(np.hstack((np.asarray(pred_hc_norm)[:, i], np.asarray(pred_nn_norm)[:, i])))
            except IndexError:
                self._other_error(self._concat_video_pred_error, last=False, out=out)
                return []
            except Exception:
                self._other_error(self._unknown_err, out=out)
                return []

            return concat

    def __load_video_model_b5(self, show_summary: bool = False, out: bool = True) -> Optional[tf.keras.Model]:
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

                from oceanai.modules.lab.video import Video

                video = Video()

                video._Video__load_video_model_b5(
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

                from oceanai.modules.lab.video import Video

                video = Video()

                video._Video__load_video_model_b5(
                    show_summary = True, out = []
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-17 10:53:03] Неверные типы или значения аргументов в "Video.__load_video_model_b5" ...
        """

        try:
            # Проверка аргументов
            if type(show_summary) is not bool or type(out) is not bool:
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__load_video_model_b5.__name__, out=out)
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

    def _get_visual_features(
        self,
        path: str,
        reduction_fps: int = 5,
        window: int = 10,
        step: int = 5,
        lang: str = "ru",
        last: bool = False,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Извлечение признаков из визуального сигнала (без очистки истории вывода сообщений в ячейке Jupyter)

        .. note::
            protected (защищенный метод)

        Args:
            path (str): Путь к видеофайлу
            reduction_fps (int): Понижение кадровой частоты
            window (int): Размер окна сегмента сигнала (в кадрах)
            step (int): Шаг сдвига окна сегмента сигнала (в кадрах)
            lang (str): Язык
            last (bool): Замена последнего сообщения
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж с двумя np.ndarray:

                1. np.ndarray с экспертными признаками
                2. np.ndarray с нейросетевыми признаками

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                res_load_model_deep_fe = video.load_video_model_deep_fe(
                    show_summary = False,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-11-03 16:37:12] Формирование нейросетевой архитектуры для получения нейросетевых признаков (видео модальность) ...

                --- Время выполнения: 1.564 сек. ---

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                video.path_to_save_ = './models'
                video.chunk_size_ = 2000000

                url = video.weights_for_big5_['video']['fe']['sberdisk']

                res_load_video_model_weights_deep_fe = video.load_video_model_weights_deep_fe(
                    url = url,
                    force_reload = True, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-11-03 16:39:10] Загрузка весов нейросетевой модели для получения нейросетевых признаков (видео модальность) ...

                [2022-11-03 16:39:14] Загрузка файла "weights_2022-11-01_12-27-07.h5" (100.0%) ...

                --- Время выполнения: 4.874 сек. ---

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                path = '/Users/dl/GitHub/oceanai/oceanai/dataset/test80_01/glgfB3vFewc.004.mp4'

                hc_features, nn_features = video.get_visual_features(
                    path = path, reduction_fps = 5,
                    window = 10, step = 5,
                    out = True, runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-11-03 16:56:52] Извлечение признаков (экспертных и нейросетевых) из визуального сигнала ...

                [2022-11-03 16:56:58] Статистика извлеченных признаков из визуального сигнала:
                    Общее количество сегментов с:
                        1. экспертными признаками: 12
                        2. нейросетевыми признаками: 12
                    Размерность матрицы экспертных признаков одного сегмента: 10 ✕ 115
                    Размерность тензора с нейросетевыми признаками одного сегмента: 10 ✕ 512
                    Понижение кадровой частоты: с 30 до 5

                --- Время выполнения: 6.109 сек. ---

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                path = '/Users/dl/GitHub/oceanai/oceanai/dataset/test80_01/glgfB3vFewc.004.mp4'

                hc_features, nn_features = video.get_visual_features(
                    path = path, reduction_fps = 5,
                    window = 10, step = 5,
                    out = True, runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 4
                :linenos:

                [2022-11-03 16:59:45] Извлечение признаков (экспертных и нейросетевых) из визуального сигнала ...

                [2022-11-03 16:59:46] Ой! Что-то пошло не так ... нейросетевая архитектура модели для получения нейросетевых признаков не сформирована (видео модальность) ...

                --- Время выполнения: 1.358 сек. ---
        """

        try:
            # Проверка аргументов
            if (
                (type(path) is not str or not path)
                and (type(path) is not gradio.utils.NamedString)
                or type(reduction_fps) is not int
                or reduction_fps < 1
                or type(window) is not int
                or window < 1
                or type(step) is not int
                or step < 1
                or not isinstance(lang, str)
                or lang not in self.__lang_traslate
                or type(last) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self._get_visual_features.__name__, last=last, out=out)
            return np.empty([]), np.empty([])
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, last=last, out=out)
                return np.empty([]), np.empty([])

            if runtime:
                self._r_start()

            if last is False:
                # Информационное сообщение
                self._info(self._get_visual_feature_info, out=False)
                if out:
                    self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            try:
                if os.path.isfile(path) is False:
                    raise FileNotFoundError  # Не файл
            except FileNotFoundError:
                self._other_error(self._file_not_found.format(self._info_wrapper(path)), last=last, out=out)
                return np.empty([]), np.empty([])
            except Exception:
                self._other_error(self._unknown_err, last=last, out=out)
                return np.empty([]), np.empty([])
            else:
                try:
                    # Расширение файла не соответствует расширению искомых файлов
                    if Path(path).suffix[1:].lower() not in self.__supported_video_formats:
                        raise TypeError
                except TypeError:
                    self._other_error(
                        self._wrong_extension_video_formats.format(
                            self._info_wrapper(", ".join(x for x in self.__supported_video_formats))
                        ),
                        out=out,
                    )
                    return np.empty([]), np.empty([])
                except Exception:
                    self._other_error(self._unknown_err, out=out)
                    return np.empty([]), np.empty([])
                else:
                    # metadata = MediaInfo.parse(path).to_data()  # Meta данные

                    # media_info = {}  # Словарь для meta данных

                    # # Проход по всем meta словарям
                    # for track in metadata["tracks"]:
                    #     # Извлечение meta данных
                    #     if track["track_type"] in [*self._type_meta_info]:
                    #         media_info[track["track_type"]] = {}  # Словарь для meta данных определенного формата

                    #         # Проход по всем необходимым meta данным
                    #         for i, curr_necessary in enumerate(self._type_meta_info[track["track_type"]]):
                    #             try:
                    #                 val = track[curr_necessary]  # Текущее значение
                    #             except Exception:
                    #                 continue
                    #             else:
                    #                 try:
                    #                     if curr_necessary == "encoded_date":
                    #                         val = datetime.strptime(val.replace("UTC ", ""), "%Y-%m-%d %H:%M:%S")
                    #                     if (
                    #                         curr_necessary == "frame_rate"
                    #                         or curr_necessary == "minimum_frame_rate"
                    #                         or curr_necessary == "maximum_frame_rate"
                    #                     ):
                    #                         val = float(val)
                    #                 except Exception:
                    #                     continue

                    #                 # Список в строку
                    #                 if type(val) is list:
                    #                     if len(val) < 2:
                    #                         val = val[0]
                    #                     else:
                    #                         val = ", ".join([str(elem) for elem in val])

                    #                 media_info[track["track_type"]][curr_necessary] = val

                    # try:
                    #     # Всего кадров в видеопотоке
                    #     all_frames = int(media_info["Video"]["duration"] / 1000 * media_info["Video"]["frame_rate"])
                    # except Exception:
                    #     all_frames = 0

                    # try:
                    #     if all_frames == 0:
                    #         raise ValueError
                    # except ValueError:
                    #     self._other_error(self._all_frames_is_zero.format(self._info_wrapper(str(all_frames))), out=out)
                    #     return np.empty([]), np.empty([])
                    # except Exception:
                    #     self._other_error(self._unknown_err, out=out)
                    #     return np.empty([]), np.empty([])
                    # else:
                    cap = cv2.VideoCapture(path)  # Захват видеофайла для чтения
                    width_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ширина кадров в видеопотоке
                    height_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Высота кадров в видеопотоке
                    all_frames_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # # Всего кадров в видеопотоке

                    fps_cv2 = np.round(cap.get(cv2.CAP_PROP_FPS))  # Частота кадров (FPS)

                    # Вычисление коэффициента изменения размера изображения
                    reshape_img_coef = self.__calc_reshape_img_coef(
                        shape=[width_video, height_video], new_shape=self.__bndbox_face_size, out=False
                    )

                    try:
                        if reshape_img_coef == -1:
                            raise ValueError
                    except ValueError:
                        self._other_error(self._calc_reshape_img_coef_error, out=out)
                        return np.empty([]), np.empty([])
                    else:
                        # Прореживание кадров
                        if reduction_fps > fps_cv2:
                            reduction_fps = fps_cv2
                        # Всего кадров после прореживания
                        all_frms_reduct = all_frames_cv2 / (fps_cv2 / reduction_fps)

                        # Индексы кадров, которые останутся после прореживания
                        idx_reduction_frames = list(
                            map(
                                self._round_math,
                                np.arange(0, all_frames_cv2, all_frames_cv2 / all_frms_reduct, dtype=float),
                            )
                        )

                        def alignment_procedure(left_eye: List[int], right_eye: List[int]) -> float:
                            """Выравнивание угла наклона головы относительно центров глаз

                            .. note::
                                внутренняя функция

                            Args:
                                left_eye (List[int]): Координаты центра левого глаза
                                right_eye (List[int]): Координаты центра правого глаза

                            Returns:
                                float: Градус расхождения центров глаз
                            """

                            left_eye_x, left_eye_y = left_eye
                            right_eye_x, right_eye_y = right_eye

                            if left_eye_y > right_eye_y:
                                point_3rd = (right_eye_x, left_eye_y)
                                direction = -1
                            else:
                                point_3rd = (left_eye_x, right_eye_y)
                                direction = 1

                            a = distance.euclidean(np.array(left_eye), np.array(point_3rd))
                            b = distance.euclidean(np.array(right_eye), np.array(point_3rd))
                            c = distance.euclidean(np.array(right_eye), np.array(left_eye))

                            if b != 0 and c != 0:
                                cos_a = (b * b + c * c - a * a) / (2 * b * c)
                                angle = np.arccos(cos_a)
                                angle = (angle * 180) / math.pi

                                if direction == -1:
                                    angle = 90 - angle
                            else:
                                angle = 0

                            return angle

                        cnt_frame = 0  # Счетчик кадров
                        vt, pt = self.__mp_drawing._VISIBILITY_THRESHOLD, self.__mp_drawing._PRESENCE_THRESHOLD

                        hcs = []  # Набор экспертных признаков
                        bndbox_faces = []  # Области с лицами

                        # Получение 468 3D-ориентиров лица
                        with self.__mp_face_mesh.FaceMesh(
                            max_num_faces=1,  # Максимальное количество лиц для обнаружения
                            # Необходимо ли дополнительно уточнять координаты ориентиров вокруг глаз и губ
                            # и выводить дополнительные ориентиры вокруг радужной оболочки
                            refine_landmarks=True,
                            # Минимальное значение достоверности из модели обнаружения лиц,
                            # при котором обнаружение считается успешным
                            min_detection_confidence=0.5,
                            # Минимальное значение достоверности из модели отслеживания ориентиров для того,
                            # чтобы ориентиры лиц считались успешно отслеженными
                            min_tracking_confidence=0.5,
                        ) as face_mesh:
                            # Проход по всем кадрам видеопотока
                            while cap.isOpened():
                                _, frame = cap.read()  # Захват, декодирование и возврат кадра

                                if frame is None:
                                    break  # Кадр не найден

                                if cnt_frame in idx_reduction_frames:
                                    # Запись недоступна (увеличение производительности)
                                    frame.flags.writeable = False
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    results = face_mesh.process(frame)
                                    # Запись доступна
                                    frame.flags.writeable = True

                                    # Найдены 468 3D-ориентиров лица
                                    if results.multi_face_landmarks:
                                        # Проход по всем лицам
                                        for idx_face, face_landmarks in enumerate(results.multi_face_landmarks):
                                            idx_to_coors = {}  # Координаты всех ориентиров лица

                                            # Проход по всем ориентирам лица
                                            for idx_landmark, lmk in enumerate(face_landmarks.landmark):
                                                if (lmk.HasField("visibility") and lmk.visibility < vt) or (
                                                    lmk.HasField("presence") and lmk.presence < pt
                                                ):
                                                    continue

                                                # Нормализация координат
                                                norm_x = min(math.floor(lmk.x * width_video), width_video - 1)
                                                norm_y = min(math.floor(lmk.y * height_video), height_video - 1)

                                                norm_x = int(norm_x * reshape_img_coef)
                                                norm_y = int(norm_y * reshape_img_coef)

                                                # Добавление нормализованных координат ориентиров лица в словарь
                                                idx_to_coors[idx_landmark] = (norm_x, norm_y)

                                            # Вычисление ограничивающей рамки из ориентиров лица
                                            x_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 0])
                                            y_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 1])
                                            x_max = np.max(np.asarray(list(idx_to_coors.values()))[:, 0])
                                            y_max = np.max(np.asarray(list(idx_to_coors.values()))[:, 1])

                                            # Коррекция ограничивающей рамки
                                            start_x, start_y = (max(0, x_min), max(0, y_min))
                                            end_x, end_y = (
                                                min(width_video - 1, x_max),
                                                min(height_video - 1, y_max),
                                            )

                                            # Область с лицом
                                            bndbox_face = frame[
                                                int(start_y / reshape_img_coef) : int(end_y / reshape_img_coef),
                                                int(start_x / reshape_img_coef) : int(end_x / reshape_img_coef),
                                            ]
                                            # Приведение изображения с лицом в нужному размеру
                                            bndbox_face = cv2.resize(
                                                bndbox_face, self.__bndbox_face_size, interpolation=cv2.INTER_AREA
                                            )

                                            bndbox_face = tf.keras.preprocessing.image.img_to_array(bndbox_face)
                                            bndbox_face = utils.preprocess_input(bndbox_face)

                                            bndbox_face = bndbox_face.reshape(
                                                -1, self.__bndbox_face_size[0], self.__bndbox_face_size[1], 3
                                            )

                                            # 1) Координаты центра глаз
                                            # 2) Текущие экспертные признаки
                                            point_eyes, curr_seq_hc = [], []

                                            # Вычисление центра глаз
                                            for i in [[474, 476], [469, 471]]:
                                                eye_x_min = min(idx_to_coors[i[0]][0], idx_to_coors[i[1]][0])
                                                eye_y_min = min(idx_to_coors[i[0]][1], idx_to_coors[i[1]][1])
                                                # Разница между yголками глаза
                                                eye_x_diff = int(abs(idx_to_coors[i[0]][0] - idx_to_coors[i[1]][0]) / 2)
                                                eye_y_diff = int(abs(idx_to_coors[i[0]][1] - idx_to_coors[i[1]][1]) / 2)

                                                point_eyes.append(
                                                    [eye_x_min + eye_x_diff - x_min, eye_y_min + eye_y_diff - y_min]
                                                )
                                                curr_seq_hc.extend(
                                                    [eye_x_min + eye_x_diff - x_min, eye_y_min + eye_y_diff - y_min]
                                                )

                                            coords_left_eye = point_eyes[0]  # Координата центра левого глаза
                                            coords_right_eye = point_eyes[1]  # Координата центра правого глаза

                                            # Вычисление расстояния между центрами глаз
                                            curr_seq_hc.append(distance.euclidean(coords_left_eye, coords_right_eye))
                                            # Вычисление угла наклона головы
                                            curr_seq_hc.append(alignment_procedure(coords_left_eye, coords_right_eye))
                                            # Вычисление расстояния между центром левого глаза и его левым углом
                                            curr_seq_hc.append(
                                                distance.euclidean(
                                                    coords_left_eye,
                                                    np.asarray(idx_to_coors[263]) - np.asarray([x_min, y_min]),
                                                )
                                            )
                                            # Вычисление расстояния между центром левого глаза и его правым углом
                                            curr_seq_hc.append(
                                                distance.euclidean(
                                                    coords_left_eye,
                                                    np.asarray(idx_to_coors[362]) - np.asarray([x_min, y_min]),
                                                )
                                            )
                                            # Вычисление расстояния между центром правого глаза и его левым углом
                                            curr_seq_hc.append(
                                                distance.euclidean(
                                                    coords_right_eye,
                                                    np.asarray(idx_to_coors[133]) - np.asarray([x_min, y_min]),
                                                )
                                            )
                                            # Вычисление расстояния между центром правого глаза и его правым углом
                                            curr_seq_hc.append(
                                                distance.euclidean(
                                                    coords_right_eye,
                                                    np.asarray(idx_to_coors[33]) - np.asarray([x_min, y_min]),
                                                )
                                            )
                                            # Вычисление угла наклона уголков рта
                                            curr_seq_hc.append(
                                                alignment_procedure(
                                                    np.asarray(idx_to_coors[105]) - np.asarray([x_min, y_min]),
                                                    np.asarray(idx_to_coors[334]) - np.asarray([x_min, y_min]),
                                                )
                                            )
                                            # Вычисление угла наклона бровей
                                            curr_seq_hc.append(
                                                alignment_procedure(
                                                    np.asarray(idx_to_coors[61]) - np.asarray([x_min, y_min]),
                                                    np.asarray(idx_to_coors[291]) - np.asarray([x_min, y_min]),
                                                )
                                            )

                                            if lang == self.__lang_traslate[0]:
                                                coords_face_mesh = self.__coords_face_mesh_mupta
                                                couples_face_mesh = self.__couples_face_mesh_mupta
                                            else:
                                                coords_face_mesh = self.__coords_face_mesh_fi
                                                couples_face_mesh = self.__couples_face_mesh_fi

                                            for coord in coords_face_mesh:
                                                curr_seq_hc.extend(
                                                    (
                                                        np.asarray(idx_to_coors[coord]) - np.asarray([x_min, y_min])
                                                    ).tolist()
                                                )

                                            for cpl in couples_face_mesh:
                                                curr_seq_hc.append(
                                                    distance.euclidean(
                                                        np.asarray(idx_to_coors[cpl[0]]) - np.asarray([x_min, y_min]),
                                                        np.asarray(idx_to_coors[cpl[1]]) - np.asarray([x_min, y_min]),
                                                    )
                                                )

                                        bndbox_faces.append(bndbox_face)
                                        hcs.append(curr_seq_hc)
                                cnt_frame += 1

                            cap.release()

                        # Лицо не найдено не на одном кадре
                        if len(bndbox_faces) == 0:
                            self._error(self._faces_not_found, out=out)
                            return np.empty([]), np.empty([])

                        hcs = np.asarray(hcs)

                        # Коды ошибок нейросетевой модели
                        code_error_pred_deep_fe = -1

                        try:
                            # Отправка областей с лицами в нейросетевую модель для получения нейросетевых признаков
                            extract_deep_fe = self._video_model_deep_fe(np.vstack(bndbox_faces))
                        except TypeError:
                            code_error_pred_deep_fe = 1
                        except Exception:
                            code_error_pred_deep_fe = 2

                        if code_error_pred_deep_fe != -1:
                            self._error(self._model_video_deep_fe_not_formation, out=out)
                            return np.empty([]), np.empty([])

                        # 1. Список с экспертными признаками
                        # 2. Список с нейросетевыми признаками
                        hc_features, nn_features = [], []

                        # Проход по всему набору экспертных и нейросетевых признаков
                        for idx_hc_nn in range(0, len(hcs) + 1, step):
                            last_idx__hc_nn = idx_hc_nn + window  # ID последнего элемента в подвыборке

                            # Текущие подвыборки
                            curr_seq_nn = extract_deep_fe[idx_hc_nn:last_idx__hc_nn].numpy().tolist()
                            curr_seq_hc = hcs[idx_hc_nn:last_idx__hc_nn].tolist()
                            if len(curr_seq_nn) < window and len(curr_seq_nn) != 0:
                                curr_seq_hc.extend([curr_seq_hc[-1]] * (window - len(curr_seq_hc)))
                                curr_seq_nn.extend([curr_seq_nn[-1]] * (window - len(curr_seq_nn)))
                            if len(curr_seq_nn) != 0:
                                hc_features.append(curr_seq_hc)
                                nn_features.append(curr_seq_nn)

                        hc_features = stats.zscore(hc_features, axis=-1)

                        if last is False:
                            # Статистика извлеченных признаков из визуального сигнала
                            self._stat_visual_features(
                                last=last,
                                out=out,
                                len_hc_features=len(hc_features),
                                len_nn_features=len(nn_features),
                                shape_hc_features=np.array(hc_features[0]).shape,
                                shape_nn_features=np.array(nn_features[0]).shape,
                                fps_before=self._round_math(fps_cv2, out),
                                fps_after=self._round_math(reduction_fps, out),
                            )

                        return hc_features, np.asarray(nn_features)
            finally:
                if runtime:
                    self._r_end(out=out)

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def load_video_model_hc(
        self, lang: str, show_summary: bool = False, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Формирование нейросетевой архитектуры модели для получения оценок по экспертным признакам

        Args:
            lang (str): Язык
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

                from oceanai.modules.lab.video import Video

                video = Video()
                video.load_video_model_hc(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-25 16:37:43] Формирование нейросетевой архитектуры модели для получения оценок по экспертным признакам (видео модальность) ...

                --- Время выполнения: 0.659 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()
                video.load_video_model_hc(
                    show_summary = 1, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-26 12:27:41] Неверные типы или значения аргументов в "Video.load_video_model_hc" ...

                False
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        try:
            # Проверка аргументов
            if (
                not isinstance(lang, str)
                or lang not in self.__lang_traslate
                or type(show_summary) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.load_video_model_hc.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._formation_video_model_hc, last=False, out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            if lang == self.__lang_traslate[0]:
                input_lstm = tf.keras.Input(shape=(10, 109))
            else:
                input_lstm = tf.keras.Input(shape=(10, 115))

            x = tf.keras.layers.LSTM(64, return_sequences=True)(input_lstm)
            x = tf.keras.layers.Dropout(rate=0.2)(x)
            x = tf.keras.layers.LSTM(128, return_sequences=False, name="lstm_128_v_hc")(x)
            x = tf.keras.layers.Dropout(rate=0.2)(x)
            x = tf.keras.layers.Dense(5, activation="linear")(x)

            self._video_model_hc = tf.keras.Model(inputs=input_lstm, outputs=x)

            if show_summary and out:
                self._video_model_hc.summary()

            if runtime:
                self._r_end(out=out)

            return True

    def load_video_model_deep_fe(
        self, show_summary: bool = False, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Формирование нейросетевой архитектуры для получения нейросетевых признаков

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

                from oceanai.modules.lab.video import Video

                video = Video()
                video.load_video_model_deep_fe(
                    show_summary = True, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-11-01 12:18:14] Формирование нейросетевой архитектуры для получения нейросетевых признаков (видео модальность) ...

                Model: "model_1"
                __________________________________________________________________________________________________
                 Layer (type)                   Output Shape         Param #     Connected to
                ==================================================================================================
                 input_2 (InputLayer)           [(None, 224, 224, 3  0           []
                                                )]

                 conv1/7x7_s2 (Conv2D)          (None, 112, 112, 64  9408        ['input_2[0][0]']
                                                )

                 conv1/7x7_s2/bn (BatchNormaliz  (None, 112, 112, 64  256        ['conv1/7x7_s2[0][0]']
                 ation)                         )

                 activation_49 (Activation)     (None, 112, 112, 64  0           ['conv1/7x7_s2/bn[0][0]']
                                                )

                 max_pooling2d_1 (MaxPooling2D)  (None, 55, 55, 64)  0           ['activation_49[0][0]']

                 conv2_1_1x1_reduce (Conv2D)    (None, 55, 55, 64)   4096        ['max_pooling2d_1[0][0]']

                 conv2_1_1x1_reduce/bn (BatchNo  (None, 55, 55, 64)  256         ['conv2_1_1x1_reduce[0][0]']
                 rmalization)

                 activation_50 (Activation)     (None, 55, 55, 64)   0           ['conv2_1_1x1_reduce/bn[0][0]']

                 conv2_1_3x3 (Conv2D)           (None, 55, 55, 64)   36864       ['activation_50[0][0]']

                 conv2_1_3x3/bn (BatchNormaliza  (None, 55, 55, 64)  256         ['conv2_1_3x3[0][0]']
                 tion)

                 activation_51 (Activation)     (None, 55, 55, 64)   0           ['conv2_1_3x3/bn[0][0]']

                 conv2_1_1x1_increase (Conv2D)  (None, 55, 55, 256)  16384       ['activation_51[0][0]']

                 conv2_1_1x1_proj (Conv2D)      (None, 55, 55, 256)  16384       ['max_pooling2d_1[0][0]']

                 conv2_1_1x1_increase/bn (Batch  (None, 55, 55, 256)  1024       ['conv2_1_1x1_increase[0][0]']
                 Normalization)

                 conv2_1_1x1_proj/bn (BatchNorm  (None, 55, 55, 256)  1024       ['conv2_1_1x1_proj[0][0]']
                 alization)

                 add_16 (Add)                   (None, 55, 55, 256)  0           ['conv2_1_1x1_increase/bn[0][0]',
                                                                                  'conv2_1_1x1_proj/bn[0][0]']

                 activation_52 (Activation)     (None, 55, 55, 256)  0           ['add_16[0][0]']

                 conv2_2_1x1_reduce (Conv2D)    (None, 55, 55, 64)   16384       ['activation_52[0][0]']

                 conv2_2_1x1_reduce/bn (BatchNo  (None, 55, 55, 64)  256         ['conv2_2_1x1_reduce[0][0]']
                 rmalization)

                 activation_53 (Activation)     (None, 55, 55, 64)   0           ['conv2_2_1x1_reduce/bn[0][0]']

                 conv2_2_3x3 (Conv2D)           (None, 55, 55, 64)   36864       ['activation_53[0][0]']

                 conv2_2_3x3/bn (BatchNormaliza  (None, 55, 55, 64)  256         ['conv2_2_3x3[0][0]']
                 tion)

                 activation_54 (Activation)     (None, 55, 55, 64)   0           ['conv2_2_3x3/bn[0][0]']

                 conv2_2_1x1_increase (Conv2D)  (None, 55, 55, 256)  16384       ['activation_54[0][0]']

                 conv2_2_1x1_increase/bn (Batch  (None, 55, 55, 256)  1024       ['conv2_2_1x1_increase[0][0]']
                 Normalization)

                 add_17 (Add)                   (None, 55, 55, 256)  0           ['conv2_2_1x1_increase/bn[0][0]',
                                                                                  'activation_52[0][0]']

                 activation_55 (Activation)     (None, 55, 55, 256)  0           ['add_17[0][0]']

                 conv2_3_1x1_reduce (Conv2D)    (None, 55, 55, 64)   16384       ['activation_55[0][0]']

                 conv2_3_1x1_reduce/bn (BatchNo  (None, 55, 55, 64)  256         ['conv2_3_1x1_reduce[0][0]']
                 rmalization)

                 activation_56 (Activation)     (None, 55, 55, 64)   0           ['conv2_3_1x1_reduce/bn[0][0]']

                 conv2_3_3x3 (Conv2D)           (None, 55, 55, 64)   36864       ['activation_56[0][0]']

                 conv2_3_3x3/bn (BatchNormaliza  (None, 55, 55, 64)  256         ['conv2_3_3x3[0][0]']
                 tion)

                 activation_57 (Activation)     (None, 55, 55, 64)   0           ['conv2_3_3x3/bn[0][0]']

                 conv2_3_1x1_increase (Conv2D)  (None, 55, 55, 256)  16384       ['activation_57[0][0]']

                 conv2_3_1x1_increase/bn (Batch  (None, 55, 55, 256)  1024       ['conv2_3_1x1_increase[0][0]']
                 Normalization)

                 add_18 (Add)                   (None, 55, 55, 256)  0           ['conv2_3_1x1_increase/bn[0][0]',
                                                                                  'activation_55[0][0]']

                 activation_58 (Activation)     (None, 55, 55, 256)  0           ['add_18[0][0]']

                 conv3_1_1x1_reduce (Conv2D)    (None, 28, 28, 128)  32768       ['activation_58[0][0]']

                 conv3_1_1x1_reduce/bn (BatchNo  (None, 28, 28, 128)  512        ['conv3_1_1x1_reduce[0][0]']
                 rmalization)

                 activation_59 (Activation)     (None, 28, 28, 128)  0           ['conv3_1_1x1_reduce/bn[0][0]']

                 conv3_1_3x3 (Conv2D)           (None, 28, 28, 128)  147456      ['activation_59[0][0]']

                 conv3_1_3x3/bn (BatchNormaliza  (None, 28, 28, 128)  512        ['conv3_1_3x3[0][0]']
                 tion)

                 activation_60 (Activation)     (None, 28, 28, 128)  0           ['conv3_1_3x3/bn[0][0]']

                 conv3_1_1x1_increase (Conv2D)  (None, 28, 28, 512)  65536       ['activation_60[0][0]']

                 conv3_1_1x1_proj (Conv2D)      (None, 28, 28, 512)  131072      ['activation_58[0][0]']

                 conv3_1_1x1_increase/bn (Batch  (None, 28, 28, 512)  2048       ['conv3_1_1x1_increase[0][0]']
                 Normalization)

                 conv3_1_1x1_proj/bn (BatchNorm  (None, 28, 28, 512)  2048       ['conv3_1_1x1_proj[0][0]']
                 alization)

                 add_19 (Add)                   (None, 28, 28, 512)  0           ['conv3_1_1x1_increase/bn[0][0]',
                                                                                  'conv3_1_1x1_proj/bn[0][0]']

                 activation_61 (Activation)     (None, 28, 28, 512)  0           ['add_19[0][0]']

                 conv3_2_1x1_reduce (Conv2D)    (None, 28, 28, 128)  65536       ['activation_61[0][0]']

                 conv3_2_1x1_reduce/bn (BatchNo  (None, 28, 28, 128)  512        ['conv3_2_1x1_reduce[0][0]']
                 rmalization)

                 activation_62 (Activation)     (None, 28, 28, 128)  0           ['conv3_2_1x1_reduce/bn[0][0]']

                 conv3_2_3x3 (Conv2D)           (None, 28, 28, 128)  147456      ['activation_62[0][0]']

                 conv3_2_3x3/bn (BatchNormaliza  (None, 28, 28, 128)  512        ['conv3_2_3x3[0][0]']
                 tion)

                 activation_63 (Activation)     (None, 28, 28, 128)  0           ['conv3_2_3x3/bn[0][0]']

                 conv3_2_1x1_increase (Conv2D)  (None, 28, 28, 512)  65536       ['activation_63[0][0]']

                 conv3_2_1x1_increase/bn (Batch  (None, 28, 28, 512)  2048       ['conv3_2_1x1_increase[0][0]']
                 Normalization)

                 add_20 (Add)                   (None, 28, 28, 512)  0           ['conv3_2_1x1_increase/bn[0][0]',
                                                                                  'activation_61[0][0]']

                 activation_64 (Activation)     (None, 28, 28, 512)  0           ['add_20[0][0]']

                 conv3_3_1x1_reduce (Conv2D)    (None, 28, 28, 128)  65536       ['activation_64[0][0]']

                 conv3_3_1x1_reduce/bn (BatchNo  (None, 28, 28, 128)  512        ['conv3_3_1x1_reduce[0][0]']
                 rmalization)

                 activation_65 (Activation)     (None, 28, 28, 128)  0           ['conv3_3_1x1_reduce/bn[0][0]']

                 conv3_3_3x3 (Conv2D)           (None, 28, 28, 128)  147456      ['activation_65[0][0]']

                 conv3_3_3x3/bn (BatchNormaliza  (None, 28, 28, 128)  512        ['conv3_3_3x3[0][0]']
                 tion)

                 activation_66 (Activation)     (None, 28, 28, 128)  0           ['conv3_3_3x3/bn[0][0]']

                 conv3_3_1x1_increase (Conv2D)  (None, 28, 28, 512)  65536       ['activation_66[0][0]']

                 conv3_3_1x1_increase/bn (Batch  (None, 28, 28, 512)  2048       ['conv3_3_1x1_increase[0][0]']
                 Normalization)

                 add_21 (Add)                   (None, 28, 28, 512)  0           ['conv3_3_1x1_increase/bn[0][0]',
                                                                                  'activation_64[0][0]']

                 activation_67 (Activation)     (None, 28, 28, 512)  0           ['add_21[0][0]']

                 conv3_4_1x1_reduce (Conv2D)    (None, 28, 28, 128)  65536       ['activation_67[0][0]']

                 conv3_4_1x1_reduce/bn (BatchNo  (None, 28, 28, 128)  512        ['conv3_4_1x1_reduce[0][0]']
                 rmalization)

                 activation_68 (Activation)     (None, 28, 28, 128)  0           ['conv3_4_1x1_reduce/bn[0][0]']

                 conv3_4_3x3 (Conv2D)           (None, 28, 28, 128)  147456      ['activation_68[0][0]']

                 conv3_4_3x3/bn (BatchNormaliza  (None, 28, 28, 128)  512        ['conv3_4_3x3[0][0]']
                 tion)

                 activation_69 (Activation)     (None, 28, 28, 128)  0           ['conv3_4_3x3/bn[0][0]']

                 conv3_4_1x1_increase (Conv2D)  (None, 28, 28, 512)  65536       ['activation_69[0][0]']

                 conv3_4_1x1_increase/bn (Batch  (None, 28, 28, 512)  2048       ['conv3_4_1x1_increase[0][0]']
                 Normalization)

                 add_22 (Add)                   (None, 28, 28, 512)  0           ['conv3_4_1x1_increase/bn[0][0]',
                                                                                  'activation_67[0][0]']

                 activation_70 (Activation)     (None, 28, 28, 512)  0           ['add_22[0][0]']

                 conv4_1_1x1_reduce (Conv2D)    (None, 14, 14, 256)  131072      ['activation_70[0][0]']

                 conv4_1_1x1_reduce/bn (BatchNo  (None, 14, 14, 256)  1024       ['conv4_1_1x1_reduce[0][0]']
                 rmalization)

                 activation_71 (Activation)     (None, 14, 14, 256)  0           ['conv4_1_1x1_reduce/bn[0][0]']

                 conv4_1_3x3 (Conv2D)           (None, 14, 14, 256)  589824      ['activation_71[0][0]']

                 conv4_1_3x3/bn (BatchNormaliza  (None, 14, 14, 256)  1024       ['conv4_1_3x3[0][0]']
                 tion)

                 activation_72 (Activation)     (None, 14, 14, 256)  0           ['conv4_1_3x3/bn[0][0]']

                 conv4_1_1x1_increase (Conv2D)  (None, 14, 14, 1024  262144      ['activation_72[0][0]']
                                                )

                 conv4_1_1x1_proj (Conv2D)      (None, 14, 14, 1024  524288      ['activation_70[0][0]']
                                                )

                 conv4_1_1x1_increase/bn (Batch  (None, 14, 14, 1024  4096       ['conv4_1_1x1_increase[0][0]']
                 Normalization)                 )

                 conv4_1_1x1_proj/bn (BatchNorm  (None, 14, 14, 1024  4096       ['conv4_1_1x1_proj[0][0]']
                 alization)                     )

                 add_23 (Add)                   (None, 14, 14, 1024  0           ['conv4_1_1x1_increase/bn[0][0]',
                                                )                                 'conv4_1_1x1_proj/bn[0][0]']

                 activation_73 (Activation)     (None, 14, 14, 1024  0           ['add_23[0][0]']
                                                )

                 conv4_2_1x1_reduce (Conv2D)    (None, 14, 14, 256)  262144      ['activation_73[0][0]']

                 conv4_2_1x1_reduce/bn (BatchNo  (None, 14, 14, 256)  1024       ['conv4_2_1x1_reduce[0][0]']
                 rmalization)

                 activation_74 (Activation)     (None, 14, 14, 256)  0           ['conv4_2_1x1_reduce/bn[0][0]']

                 conv4_2_3x3 (Conv2D)           (None, 14, 14, 256)  589824      ['activation_74[0][0]']

                 conv4_2_3x3/bn (BatchNormaliza  (None, 14, 14, 256)  1024       ['conv4_2_3x3[0][0]']
                 tion)

                 activation_75 (Activation)     (None, 14, 14, 256)  0           ['conv4_2_3x3/bn[0][0]']

                 conv4_2_1x1_increase (Conv2D)  (None, 14, 14, 1024  262144      ['activation_75[0][0]']
                                                )

                 conv4_2_1x1_increase/bn (Batch  (None, 14, 14, 1024  4096       ['conv4_2_1x1_increase[0][0]']
                 Normalization)                 )

                 add_24 (Add)                   (None, 14, 14, 1024  0           ['conv4_2_1x1_increase/bn[0][0]',
                                                )                                 'activation_73[0][0]']

                 activation_76 (Activation)     (None, 14, 14, 1024  0           ['add_24[0][0]']
                                                )

                 conv4_3_1x1_reduce (Conv2D)    (None, 14, 14, 256)  262144      ['activation_76[0][0]']

                 conv4_3_1x1_reduce/bn (BatchNo  (None, 14, 14, 256)  1024       ['conv4_3_1x1_reduce[0][0]']
                 rmalization)

                 activation_77 (Activation)     (None, 14, 14, 256)  0           ['conv4_3_1x1_reduce/bn[0][0]']

                 conv4_3_3x3 (Conv2D)           (None, 14, 14, 256)  589824      ['activation_77[0][0]']

                 conv4_3_3x3/bn (BatchNormaliza  (None, 14, 14, 256)  1024       ['conv4_3_3x3[0][0]']
                 tion)

                 activation_78 (Activation)     (None, 14, 14, 256)  0           ['conv4_3_3x3/bn[0][0]']

                 conv4_3_1x1_increase (Conv2D)  (None, 14, 14, 1024  262144      ['activation_78[0][0]']
                                                )

                 conv4_3_1x1_increase/bn (Batch  (None, 14, 14, 1024  4096       ['conv4_3_1x1_increase[0][0]']
                 Normalization)                 )

                 add_25 (Add)                   (None, 14, 14, 1024  0           ['conv4_3_1x1_increase/bn[0][0]',
                                                )                                 'activation_76[0][0]']

                 activation_79 (Activation)     (None, 14, 14, 1024  0           ['add_25[0][0]']
                                                )

                 conv4_4_1x1_reduce (Conv2D)    (None, 14, 14, 256)  262144      ['activation_79[0][0]']

                 conv4_4_1x1_reduce/bn (BatchNo  (None, 14, 14, 256)  1024       ['conv4_4_1x1_reduce[0][0]']
                 rmalization)

                 activation_80 (Activation)     (None, 14, 14, 256)  0           ['conv4_4_1x1_reduce/bn[0][0]']

                 conv4_4_3x3 (Conv2D)           (None, 14, 14, 256)  589824      ['activation_80[0][0]']

                 conv4_4_3x3/bn (BatchNormaliza  (None, 14, 14, 256)  1024       ['conv4_4_3x3[0][0]']
                 tion)

                 activation_81 (Activation)     (None, 14, 14, 256)  0           ['conv4_4_3x3/bn[0][0]']

                 conv4_4_1x1_increase (Conv2D)  (None, 14, 14, 1024  262144      ['activation_81[0][0]']
                                                )

                 conv4_4_1x1_increase/bn (Batch  (None, 14, 14, 1024  4096       ['conv4_4_1x1_increase[0][0]']
                 Normalization)                 )

                 add_26 (Add)                   (None, 14, 14, 1024  0           ['conv4_4_1x1_increase/bn[0][0]',
                                                )                                 'activation_79[0][0]']

                 activation_82 (Activation)     (None, 14, 14, 1024  0           ['add_26[0][0]']
                                                )

                 conv4_5_1x1_reduce (Conv2D)    (None, 14, 14, 256)  262144      ['activation_82[0][0]']

                 conv4_5_1x1_reduce/bn (BatchNo  (None, 14, 14, 256)  1024       ['conv4_5_1x1_reduce[0][0]']
                 rmalization)

                 activation_83 (Activation)     (None, 14, 14, 256)  0           ['conv4_5_1x1_reduce/bn[0][0]']

                 conv4_5_3x3 (Conv2D)           (None, 14, 14, 256)  589824      ['activation_83[0][0]']

                 conv4_5_3x3/bn (BatchNormaliza  (None, 14, 14, 256)  1024       ['conv4_5_3x3[0][0]']
                 tion)

                 activation_84 (Activation)     (None, 14, 14, 256)  0           ['conv4_5_3x3/bn[0][0]']

                 conv4_5_1x1_increase (Conv2D)  (None, 14, 14, 1024  262144      ['activation_84[0][0]']
                                                )

                 conv4_5_1x1_increase/bn (Batch  (None, 14, 14, 1024  4096       ['conv4_5_1x1_increase[0][0]']
                 Normalization)                 )

                 add_27 (Add)                   (None, 14, 14, 1024  0           ['conv4_5_1x1_increase/bn[0][0]',
                                                )                                 'activation_82[0][0]']

                 activation_85 (Activation)     (None, 14, 14, 1024  0           ['add_27[0][0]']
                                                )

                 conv4_6_1x1_reduce (Conv2D)    (None, 14, 14, 256)  262144      ['activation_85[0][0]']

                 conv4_6_1x1_reduce/bn (BatchNo  (None, 14, 14, 256)  1024       ['conv4_6_1x1_reduce[0][0]']
                 rmalization)

                 activation_86 (Activation)     (None, 14, 14, 256)  0           ['conv4_6_1x1_reduce/bn[0][0]']

                 conv4_6_3x3 (Conv2D)           (None, 14, 14, 256)  589824      ['activation_86[0][0]']

                 conv4_6_3x3/bn (BatchNormaliza  (None, 14, 14, 256)  1024       ['conv4_6_3x3[0][0]']
                 tion)

                 activation_87 (Activation)     (None, 14, 14, 256)  0           ['conv4_6_3x3/bn[0][0]']

                 conv4_6_1x1_increase (Conv2D)  (None, 14, 14, 1024  262144      ['activation_87[0][0]']
                                                )

                 conv4_6_1x1_increase/bn (Batch  (None, 14, 14, 1024  4096       ['conv4_6_1x1_increase[0][0]']
                 Normalization)                 )

                 add_28 (Add)                   (None, 14, 14, 1024  0           ['conv4_6_1x1_increase/bn[0][0]',
                                                )                                 'activation_85[0][0]']

                 activation_88 (Activation)     (None, 14, 14, 1024  0           ['add_28[0][0]']
                                                )

                 conv5_1_1x1_reduce (Conv2D)    (None, 7, 7, 512)    524288      ['activation_88[0][0]']

                 conv5_1_1x1_reduce/bn (BatchNo  (None, 7, 7, 512)   2048        ['conv5_1_1x1_reduce[0][0]']
                 rmalization)

                 activation_89 (Activation)     (None, 7, 7, 512)    0           ['conv5_1_1x1_reduce/bn[0][0]']

                 conv5_1_3x3 (Conv2D)           (None, 7, 7, 512)    2359296     ['activation_89[0][0]']

                 conv5_1_3x3/bn (BatchNormaliza  (None, 7, 7, 512)   2048        ['conv5_1_3x3[0][0]']
                 tion)

                 activation_90 (Activation)     (None, 7, 7, 512)    0           ['conv5_1_3x3/bn[0][0]']

                 conv5_1_1x1_increase (Conv2D)  (None, 7, 7, 2048)   1048576     ['activation_90[0][0]']

                 conv5_1_1x1_proj (Conv2D)      (None, 7, 7, 2048)   2097152     ['activation_88[0][0]']

                 conv5_1_1x1_increase/bn (Batch  (None, 7, 7, 2048)  8192        ['conv5_1_1x1_increase[0][0]']
                 Normalization)

                 conv5_1_1x1_proj/bn (BatchNorm  (None, 7, 7, 2048)  8192        ['conv5_1_1x1_proj[0][0]']
                 alization)

                 add_29 (Add)                   (None, 7, 7, 2048)   0           ['conv5_1_1x1_increase/bn[0][0]',
                                                                                  'conv5_1_1x1_proj/bn[0][0]']

                 activation_91 (Activation)     (None, 7, 7, 2048)   0           ['add_29[0][0]']

                 conv5_2_1x1_reduce (Conv2D)    (None, 7, 7, 512)    1048576     ['activation_91[0][0]']

                 conv5_2_1x1_reduce/bn (BatchNo  (None, 7, 7, 512)   2048        ['conv5_2_1x1_reduce[0][0]']
                 rmalization)

                 activation_92 (Activation)     (None, 7, 7, 512)    0           ['conv5_2_1x1_reduce/bn[0][0]']

                 conv5_2_3x3 (Conv2D)           (None, 7, 7, 512)    2359296     ['activation_92[0][0]']

                 conv5_2_3x3/bn (BatchNormaliza  (None, 7, 7, 512)   2048        ['conv5_2_3x3[0][0]']
                 tion)

                 activation_93 (Activation)     (None, 7, 7, 512)    0           ['conv5_2_3x3/bn[0][0]']

                 conv5_2_1x1_increase (Conv2D)  (None, 7, 7, 2048)   1048576     ['activation_93[0][0]']

                 conv5_2_1x1_increase/bn (Batch  (None, 7, 7, 2048)  8192        ['conv5_2_1x1_increase[0][0]']
                 Normalization)

                 add_30 (Add)                   (None, 7, 7, 2048)   0           ['conv5_2_1x1_increase/bn[0][0]',
                                                                                  'activation_91[0][0]']

                 activation_94 (Activation)     (None, 7, 7, 2048)   0           ['add_30[0][0]']

                 conv5_3_1x1_reduce (Conv2D)    (None, 7, 7, 512)    1048576     ['activation_94[0][0]']

                 conv5_3_1x1_reduce/bn (BatchNo  (None, 7, 7, 512)   2048        ['conv5_3_1x1_reduce[0][0]']
                 rmalization)

                 activation_95 (Activation)     (None, 7, 7, 512)    0           ['conv5_3_1x1_reduce/bn[0][0]']

                 conv5_3_3x3 (Conv2D)           (None, 7, 7, 512)    2359296     ['activation_95[0][0]']

                 conv5_3_3x3/bn (BatchNormaliza  (None, 7, 7, 512)   2048        ['conv5_3_3x3[0][0]']
                 tion)

                 activation_96 (Activation)     (None, 7, 7, 512)    0           ['conv5_3_3x3/bn[0][0]']

                 conv5_3_1x1_increase (Conv2D)  (None, 7, 7, 2048)   1048576     ['activation_96[0][0]']

                 conv5_3_1x1_increase/bn (Batch  (None, 7, 7, 2048)  8192        ['conv5_3_1x1_increase[0][0]']
                 Normalization)

                 add_31 (Add)                   (None, 7, 7, 2048)   0           ['conv5_3_1x1_increase/bn[0][0]',
                                                                                  'activation_94[0][0]']

                 activation_97 (Activation)     (None, 7, 7, 2048)   0           ['add_31[0][0]']

                 avg_pool (AveragePooling2D)    (None, 1, 1, 2048)   0           ['activation_97[0][0]']

                 global_average_pooling2d_1 (Gl  (None, 2048)        0           ['avg_pool[0][0]']
                 obalAveragePooling2D)

                 gaussian_noise_1 (GaussianNois  (None, 2048)        0           ['global_average_pooling2d_1[0][0
                 e)                                                              ]']

                 dense_x (Dense)                (None, 512)          1049088     ['gaussian_noise_1[0][0]']

                 dropout_1 (Dropout)            (None, 512)          0           ['dense_x[0][0]']

                 dense_1 (Dense)                (None, 7)            3591        ['dropout_1[0][0]']

                ==================================================================================================
                Total params: 24,613,831
                Trainable params: 24,560,711
                Non-trainable params: 53,120
                __________________________________________________________________________________________________
                --- Время выполнения: 2.222 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()
                video.load_video_model_deep_fe(
                    show_summary = 1, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-11-01 12:21:23] Неверные типы или значения аргументов в "Video.load_video_model_deep_fe" ...

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
            self._inv_args(__class__.__name__, self.load_video_model_deep_fe.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._formation_video_deep_fe, last=False, out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            basis_model = VGGFace(
                model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling="avg", weights=None
            )

            gauss_noise = tf.keras.layers.GaussianNoise(0.1)(basis_model.output)
            x = tf.keras.layers.Dense(
                units=512, kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation="relu", name="dense_x"
            )(gauss_noise)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(7, activation="softmax")(x)

            self._video_model_deep_fe = tf.keras.Model(basis_model.input, x)

            if show_summary and out:
                self._video_model_deep_fe.summary()

            if runtime:
                self._r_end(out=out)

            return True

    def load_video_model_nn(
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

                from oceanai.modules.lab.video import Video

                video = Video()
                video.load_video_model_nn(
                    show_summary = True, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-27 14:46:11] Формирование нейросетевой архитектуры для получения оценок по нейросетевым признакам (видео модальность) ...

                Model: "model"
                _________________________________________________________________
                 Layer (type)                Output Shape              Param #
                =================================================================
                 input_1 (InputLayer)        [(None, 10, 512)]         0

                 lstm (LSTM)                 (None, 1024)              6295552

                 dropout (Dropout)           (None, 1024)              0

                 dense (Dense)               (None, 5)                 5125

                 activation (Activation)     (None, 5)                 0

                =================================================================
                Total params: 6,300,677
                Trainable params: 6,300,677
                Non-trainable params: 0
                _________________________________________________________________
                --- Время выполнения: 2.018 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()
                video.load_video_model_nn(
                    show_summary = 1, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-27 14:47:22] Неверные типы или значения аргументов в "Video.load_video_model_nn" ...

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
            self._inv_args(__class__.__name__, self.load_video_model_nn.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._formation_video_model_nn, last=False, out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            input_lstm = tf.keras.Input(shape=(10, 512))

            x = tf.keras.layers.LSTM(
                1024, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-3), name="lstm_1024_v_nn"
            )(input_lstm)
            x = tf.keras.layers.Dropout(rate=0.2)(x)
            x = tf.keras.layers.Dense(units=5)(x)
            x = tf.keras.layers.Activation("linear")(x)

            self._video_model_nn = tf.keras.Model(inputs=input_lstm, outputs=x)

            if show_summary and out:
                self._video_model_nn.summary()

            if runtime:
                self._r_end(out=out)

            return True

    def load_video_models_b5(
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

                from oceanai.modules.lab.video import Video

                video = Video()
                video.load_video_models_b5(
                    show_summary = True, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-11-04 15:29:26] Формирование нейросетевых архитектур моделей для получения результатов оценки персональных качеств (видео модальность) ...

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
                --- Время выполнения: 0.116 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()
                video.load_video_models_b5(
                    show_summary = 1, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-11-04 15:30:15] Неверные типы или значения аргументов в "Video.load_video_models_b5" ...

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
            self._inv_args(__class__.__name__, self.load_video_models_b5.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._formation_video_models_b5, last=False, out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            for key, _ in self._video_models_b5.items():
                self._video_models_b5[key] = self.__load_video_model_b5()

            if show_summary and out:
                self._video_models_b5[key].summary()

            if runtime:
                self._r_end(out=out)

            return True

    def load_video_model_weights_hc(
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

                from oceanai.modules.lab.video import Video

                video = Video()

                video.load_video_model_hc(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-27 12:55:31] Формирование нейросетевой архитектуры модели для получения оценок по экспертным признакам (видео модальность) ...

                --- Время выполнения: 0.606 сек. ---

                True

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                video.path_to_save_ = './models'
                video.chunk_size_ = 2000000

                url = video.weights_for_big5_['video']['hc']['sberdisk']

                video.load_video_model_weights_hc(
                    url = url,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-27 13:08:04] Загрузка весов нейросетевой модели для получения оценок по экспертным признакам (видео модальность) ...

                [2022-10-27 13:08:05] Загрузка файла "weights_2022-08-27_18-53-35.h5" (100.0%) ...

                --- Время выполнения: 0.493 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                video.path_to_save_ = './models'
                video.chunk_size_ = 2000000

                url = video.weights_for_big5_['video']['hc']['sberdisk']

                video.load_video_model_weights_hc(
                    url = url,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-27 13:09:54] Загрузка весов нейросетевой модели для получения оценок по экспертным признакам (видео модальность) ...

                [2022-10-27 13:09:54] Загрузка файла "weights_2022-08-27_18-53-35.h5" (100.0%) ...

                [2022-10-27 13:09:54] Ой! Что-то пошло не так ... нейросетевая архитектура модели для получения оценок по экспертным признакам не сформирована (видео модальность) ...

                --- Время выполнения: 0.424 сек. ---

                False
        """

        if runtime:
            self._r_start()

        if self.__load_model_weights(url, force_reload, self._load_video_model_weights_hc, out, False, run) is True:
            try:
                self._video_model_hc.load_weights(self._url_last_filename)
                self._video_model_hc = tf.keras.models.Model(
                    inputs=self._video_model_hc.input,
                    outputs=[self._video_model_hc.output, self._video_model_hc.get_layer("lstm_128_v_hc").output],
                )
            except Exception:
                self._error(self._model_video_hc_not_formation, out=out)
                return False
            else:
                return True
            finally:
                if runtime:
                    self._r_end(out=out)

        return False

    def load_video_model_weights_deep_fe(
        self, url: str, force_reload: bool = True, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Загрузка весов нейросетевой модели для получения нейросетевых признаков

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

                from oceanai.modules.lab.video import Video

                video = Video()

                video.load_video_model_deep_fe(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-11-01 12:41:59] Формирование нейросетевой архитектуры для получения нейросетевых признаков (видео модальность) ...

                --- Время выполнения: 1.306 сек. ---

                True

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                video.path_to_save_ = './models'
                video.chunk_size_ = 2000000

                url = video.weights_for_big5_['video']['fe']['sberdisk']

                video.load_video_model_weights_deep_fe(
                    url = url,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-11-01 12:42:51] Загрузка весов нейросетевой модели для получения нейросетевых признаков (видео модальность) ...

                [2022-11-01 12:43:06] Загрузка файла "weights_2022-11-01_12-27-07.h5" (100.0%) ...

                --- Время выполнения: 14.781 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                video.path_to_save_ = './models'
                video.chunk_size_ = 2000000

                url = video.weights_for_big5_['video']['fe']['sberdisk']

                video.load_video_model_weights_deep_fe(
                    url = url,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-11-01 12:44:14] Загрузка весов нейросетевой модели для получения нейросетевых признаков (видео модальность) ...

                [2022-11-01 12:44:28] Загрузка файла "weights_2022-11-01_12-27-07.h5" (100.0%) ...

                [2022-11-01 12:44:28] Ой! Что-то пошло не так ... нейросетевая архитектура модели для получения нейросетевых признаков не сформирована (видео модальность) ...

                --- Время выполнения: 13.926 сек. ---

                False
        """

        if runtime:
            self._r_start()

        if (
            self.__load_model_weights(url, force_reload, self._load_video_model_weights_deep_fe, out, False, run)
            is True
        ):
            try:
                self._video_model_deep_fe.load_weights(self._url_last_filename)
                self._video_model_deep_fe = tf.keras.Model(
                    inputs=self._video_model_deep_fe.input,
                    outputs=[self._video_model_deep_fe.get_layer("dense_x").output],
                )
            except Exception:
                self._error(self._model_video_deep_fe_not_formation, out=out)
                return False
            else:
                return True
            finally:
                if runtime:
                    self._r_end(out=out)

        return False

    def load_video_model_weights_nn(
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

                from oceanai.modules.lab.video import Video

                video = Video()

                video.load_video_model_nn(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-27 15:17:13] Формирование нейросетевой архитектуры для получения оценок по нейросетевым признакам (видео модальность) ...

                --- Время выполнения: 1.991 сек. ---

                True

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                video.path_to_save_ = './models'
                video.chunk_size_ = 2000000

                url = video.weights_for_big5_['video']['nn']['sberdisk']

                video.load_video_model_weights_nn(
                    url = url,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-27 15:19:08] Загрузка весов нейросетевой модели для получения оценок по нейросетевым признакам (видео модальность) ...

                [2022-10-27 15:19:11] Загрузка файла "weights_2022-03-22_16-31-48.h5" (100.0%) ...

                --- Время выполнения: 3.423 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                video.path_to_save_ = './models'
                video.chunk_size_ = 2000000

                url = video.weights_for_big5_['video']['nn']['sberdisk']

                video.load_video_model_weights_nn(
                    url = url,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-27 15:19:40] Загрузка весов нейросетевой модели для получения оценок по нейросетевым признакам (видео модальность) ...

                [2022-10-27 15:19:43] Загрузка файла "weights_2022-03-22_16-31-48.h5" (100.0%) ...

                [2022-10-27 15:19:43] Ой! Что-то пошло не так ... нейросетевая архитектура модели для получения оценок по нейросетевым признакам не сформирована (видео модальность) ...

                --- Время выполнения: 3.469 сек. ---

                False
        """

        if runtime:
            self._r_start()

        if self.__load_model_weights(url, force_reload, self._load_video_model_weights_nn, out, False, run) is True:
            try:
                self._video_model_nn.load_weights(self._url_last_filename)
                self._video_model_nn = tf.keras.models.Model(
                    inputs=self._video_model_nn.input,
                    outputs=[self._video_model_nn.output, self._video_model_nn.get_layer("lstm_1024_v_nn").output],
                )
            except Exception:
                self._error(self._model_video_nn_not_formation, out=out)
                return False
            else:
                return True
            finally:
                if runtime:
                    self._r_end(out=out)

        return False

    def load_video_models_weights_b5(
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

                from oceanai.modules.lab.video import Video

                video = Video()

                video.load_video_models_b5(
                    show_summary = False, out = True,
                    runtime = True, run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-11-04 18:56:41] Формирование нейросетевых архитектур моделей для получения результатов оценки персональных качеств (видео модальность) ...

                --- Время выполнения: 0.117 сек. ---

                True

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                video.path_to_save_ = './models'
                video.chunk_size_ = 2000000

                url_openness = video.weights_for_big5_['video']['b5']['openness']['sberdisk']
                url_conscientiousness = video.weights_for_big5_['video']['b5']['conscientiousness']['sberdisk']
                url_extraversion = video.weights_for_big5_['video']['b5']['extraversion']['sberdisk']
                url_agreeableness = video.weights_for_big5_['video']['b5']['agreeableness']['sberdisk']
                url_non_neuroticism = video.weights_for_big5_['video']['b5']['non_neuroticism']['sberdisk']

                video.load_video_models_weights_b5(
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

                [2022-11-04 18:58:59] Загрузка весов нейросетевых моделей для получения результатов оценки персональных качеств (видео модальность) ...

                [2022-11-04 18:59:00] Загрузка файла "weights_2022-06-15_16-46-30.h5" (100.0%) ... Открытость опыту

                [2022-11-04 18:59:00] Загрузка файла "weights_2022-06-15_16-48-50.h5" (100.0%) ... Добросовестность

                [2022-11-04 18:59:00] Загрузка файла "weights_2022-06-15_16-54-06.h5" (100.0%) ... Экстраверсия

                [2022-11-04 18:59:01] Загрузка файла "weights_2022-06-15_17-02-03.h5" (100.0%) ... Доброжелательность

                [2022-11-04 18:59:01] Загрузка файла "weights_2022-06-15_17-06-15.h5" (100.0%) ... Эмоциональная стабильность

                --- Время выполнения: 1.827 сек. ---

                True

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.video import Video

                video = Video()

                video.path_to_save_ = './models'
                video.chunk_size_ = 2000000

                url_openness = video.weights_for_big5_['video']['b5']['openness']['sberdisk']
                url_conscientiousness = video.weights_for_big5_['video']['b5']['conscientiousness']['sberdisk']
                url_extraversion = video.weights_for_big5_['video']['b5']['extraversion']['sberdisk']
                url_agreeableness = video.weights_for_big5_['video']['b5']['agreeableness']['sberdisk']
                url_non_neuroticism = video.weights_for_big5_['video']['b5']['non_neuroticism']['sberdisk']

                video.load_video_models_weights_b5(
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

                [2022-11-04 19:02:32] Загрузка весов нейросетевых моделей для получения результатов оценки персональных качеств (видео модальность) ...

                [2022-11-04 19:02:32] Загрузка файла "weights_2022-06-15_16-46-30.h5" (100.0%) ...

                [2022-11-04 19:02:32] Ой! Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ... Открытость опыту

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/video.py
                    Линия: 2833
                    Метод: load_video_models_weights_b5
                    Тип ошибки: AttributeError

                [2022-11-04 19:02:32] Загрузка файла "weights_2022-06-15_16-48-50.h5" (100.0%) ...

                [2022-11-04 19:02:32] Ой! Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ... Добросовестность

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/video.py
                    Линия: 2833
                    Метод: load_video_models_weights_b5
                    Тип ошибки: AttributeError

                [2022-11-04 19:02:33] Загрузка файла "weights_2022-06-15_16-54-06.h5" (100.0%) ...

                [2022-11-04 19:02:33] Ой! Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ... Экстраверсия

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/video.py
                    Линия: 2833
                    Метод: load_video_models_weights_b5
                    Тип ошибки: AttributeError

                [2022-11-04 19:02:33] Загрузка файла "weights_2022-06-15_17-02-03.h5" (100.0%) ...

                [2022-11-04 19:02:33] Ой! Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ... Доброжелательность

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/video.py
                    Линия: 2833
                    Метод: load_video_models_weights_b5
                    Тип ошибки: AttributeError

                [2022-11-04 19:02:34] Загрузка файла "weights_2022-06-15_17-06-15.h5" (100.0%) ...

                [2022-11-04 19:02:34] Ой! Что-то пошло не так ... не удалось загрузить веса нейросетевой модели ... Эмоциональная стабильность

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/video.py
                    Линия: 2833
                    Метод: load_video_models_weights_b5
                    Тип ошибки: AttributeError

                --- Время выполнения: 1.831 сек. ---

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
            self._inv_args(__class__.__name__, self.load_video_models_weights_b5.__name__, out=out)
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
            self._info(self._load_video_models_weights_b5, last=False, out=out)

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
                            self._video_models_b5[self._b5["en"][cnt]].load_weights(self._url_last_filename)
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

    def get_visual_features(
        self,
        path: str,
        reduction_fps: int = 5,
        window: int = 10,
        step: int = 5,
        lang: str = "ru",
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Извлечение признаков из визуального сигнала

        Args:
            path (str): Путь к видеофайлу
            reduction_fps (int): Понижение кадровой частоты
            window (int): Размер окна сегмента сигнала (в кадрах)
            step (int): Шаг сдвига окна сегмента сигнала (в кадрах)
            lang (str): Язык
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж с двумя np.ndarray:

                1. np.ndarray с экспертными признаками
                2. np.ndarray с нейросетевыми признаками

        :bdg-link-light:`Пример <../../user_guide/notebooks/Video-get_visual_features.ipynb>`
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        return self._get_visual_features(
            path=path,
            reduction_fps=reduction_fps,
            window=window,
            step=step,
            lang=lang,
            last=False,
            out=out,
            runtime=runtime,
            run=run,
        )

    def get_video_union_predictions(
        self,
        depth: int = 1,
        recursive: bool = False,
        reduction_fps: int = 5,
        window: int = 10,
        step: int = 5,
        lang: str = "ru",
        accuracy=True,
        url_accuracy: str = "",
        logs: bool = True,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> bool:
        """Получения прогнозов по видео

        Args:
            depth (int): Глубина иерархии для получения данных
            recursive (bool): Рекурсивный поиск данных
            reduction_fps (int): Понижение кадровой частоты
            window (int): Размер окна сегмента сигнала (в кадрах)
            step (int): Шаг сдвига окна сегмента сигнала (в кадрах)
            lang (str): Язык
            accuracy (bool): Вычисление точности
            url_accuracy (str): Полный путь к файлу с верными предсказаниями для подсчета точности
            logs (bool): При необходимости формировать LOG файл
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если прогнозы успешно получены, в обратном случае **False**

        :bdg-link-light:`Пример <../../user_guide/notebooks/Video-get_video_union_predictions.ipynb>`
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
                or type(reduction_fps) is not int
                or reduction_fps < 1
                or type(window) is not int
                or window < 1
                or type(step) is not int
                or step < 1
                or not isinstance(lang, str)
                or lang not in self.__lang_traslate
                or type(accuracy) is not bool
                or type(url_accuracy) is not str
                or type(logs) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.get_video_union_predictions.__name__, out=out)
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
                    get_video_union_predictions_info = self._get_union_predictions_info + self._get_accuracy_info
                else:
                    get_video_union_predictions_info = self._get_union_predictions_info

                get_video_union_predictions_info += self._video_modality

                # Вычисление точности
                if accuracy is True:
                    # Информационное сообщение
                    self._info(get_video_union_predictions_info, out=out)

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
                            get_video_union_predictions_info,
                            i,
                            self.__local_path(curr_path),
                            self.__len_paths,
                            True,
                            last,
                            out,
                        )

                        # Извлечение признаков из визуального сигнала
                        hc_features, nn_features = self._get_visual_features(
                            path=str(curr_path.resolve()),
                            reduction_fps=reduction_fps,
                            window=window,
                            step=step,
                            lang=lang,
                            last=True,
                            out=False,
                            runtime=False,
                            run=run,
                        )

                        # Признаки из акустического сигнала извлечены
                        if (
                            type(hc_features) is np.ndarray
                            and type(nn_features) is np.ndarray
                            and len(hc_features) > 0
                            and len(nn_features) > 0
                        ):
                            # Коды ошибок нейросетевых моделей
                            code_error_pred_hc = -1
                            code_error_pred_nn = -1

                            try:
                                # Оправка экспертных признаков в нейросетевую модель
                                pred_hc, _ = self.video_model_hc_(np.array(hc_features, dtype=np.float16))
                            except TypeError:
                                code_error_pred_hc = 1
                            except Exception:
                                code_error_pred_hc = 2

                            try:
                                # Отправка нейросетевых признаков в нейросетевую модель
                                pred_nn, _ = self.video_model_nn_(np.array(nn_features, dtype=np.float16))
                            except TypeError:
                                code_error_pred_nn = 1
                            except Exception:
                                code_error_pred_nn = 2

                            if code_error_pred_hc != -1 and code_error_pred_nn != -1:
                                self._error(self._models_video_not_formation, out=out)
                                return False

                            if code_error_pred_hc != -1:
                                self._error(self._model_video_hc_not_formation, out=out)
                                return False

                            if code_error_pred_nn != -1:
                                self._error(self._model_video_nn_not_formation, out=out)
                                return False

                            # Конкатенация оценок по экспертным и нейросетевым признакам
                            union_pred = self.__concat_pred(pred_hc.numpy(), pred_nn.numpy(), out=out)

                            if len(union_pred) == 0:
                                return False

                            final_pred = []

                            for cnt, (name_b5, model) in enumerate(self.video_models_b5_.items()):
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
                        get_video_union_predictions_info,
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

                        name_logs_file = self.get_video_union_predictions.__name__

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
