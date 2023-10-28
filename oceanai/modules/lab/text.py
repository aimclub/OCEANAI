#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Текст
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################

import warnings

# Подавление Warning
for warn in [UserWarning, FutureWarning]:
    warnings.filterwarnings("ignore", category=warn)

import os  # Взаимодействие с файловой системой
import logging
import requests  # Отправка HTTP запросов

from urllib.parse import urlparse

from dataclasses import dataclass  # Класс данных

from typing import List, Optional  # Типы данных

from IPython.display import clear_output

# Персональные
from oceanai.modules.lab.download import Download  # Загрузка файлов

# Порог регистрации сообщений TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # Машинное обучение от Google
import keras

from oceanai.modules.lab.utils.attention import Attention  # Модуль внимания

# Слой статистических функционалов (средние значения и стандартные отклонения)
from oceanai.modules.lab.utils.addition import Addition


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class TextMessages(Download):
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

        self._text_modality: str = self._(" (текстовая модальность) ...")
        self._formation_text_model_hc: str = self._formation_model_hc + self._text_modality

        self._load_text_model_weights_hc: str = self._load_model_weights_hc + self._text_modality

        self._model_text_hc_not_formation: str = self._model_hc_not_formation + self._text_modality


# ######################################################################################################################
# Текст
# ######################################################################################################################
@dataclass
class Text(TextMessages):
    """Класс для обработки текста

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
        self._text_model_hc: Optional[keras.engine.functional.Functional] = None

        # ----------------------- Только для внутреннего использования внутри класса

        # Названия мультимодальных корпусов
        self.__multi_corpora: List[str] = ["fi", "mupta"]

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def text_model_hc_(self) -> Optional[keras.engine.functional.Functional]:
        """Получение нейросетевой модели **tf.keras.Model** для получения оценок по экспертным признакам

        Returns:
            Optional[keras.engine.functional.Functional]: Нейросетевая модель **tf.keras.Model** или None
        """

        return self._text_model_hc

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

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def load_text_model_hc(
        self, corpus: str = "", show_summary: bool = False, out: bool = True, runtime: bool = True, run: bool = True
    ):
        """Формирование нейросетевой архитектуры модели для получения оценок по экспертным признакам

        Args:
            corpus (str): Корпус для тестирования нейросетевой модели
            show_summary (bool): Отображение сформированной нейросетевой архитектуры модели
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
                type(corpus) is not str
                or not corpus
                or (corpus in self.__multi_corpora) is False
                or type(show_summary) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.load_text_model_hc.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._formation_text_model_hc, last=False, out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            # fi
            if corpus is self.__multi_corpora[0]:
                input_shape = (89, 64)
            # mupta
            elif corpus is self.__multi_corpora[1]:
                input_shape = (365, 64)
            else:
                input_shape = (89, 64)

            input_lstm = tf.keras.Input(shape=input_shape, name="model_hc/input")

            x_lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True), name="model_hc/bilstm_1"
            )(input_lstm)

            x_attention = Attention(use_scale=False, score_mode="dot", name="model_hc/attention")(x_lstm, x_lstm)

            x_dence = tf.keras.layers.Dense(32 * 2, name="model_hc/dence_2", activation="relu")(input_lstm)
            x_dence = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True), name="model_hc/bilstm_2"
            )(x_dence)

            x = tf.keras.layers.Add()([x_lstm, x_attention, x_dence])
            x = Addition(name="model_hc/add")(x)

            x = tf.keras.layers.Dense(5, activation="sigmoid")(x)
            self._text_model_hc = tf.keras.Model(input_lstm, outputs=x, name="model_hc")

            if show_summary and out:
                self._text_model_hc.summary()

            if runtime:
                self._r_end(out=out)

            return True

    def load_text_model_weights_hc(
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
        """

        if runtime:
            self._r_start()

        if self.__load_model_weights(url, force_reload, self._load_text_model_weights_hc, out, False, run) is True:
            try:
                self._text_model_hc.load_weights(self._url_last_filename)
            except Exception:
                self._error(self._model_text_hc_not_formation, out=out)
                return False
            else:
                return True
            finally:
                if runtime:
                    self._r_end(out=out)

        return False
