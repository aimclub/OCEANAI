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

from dataclasses import dataclass  # Класс данных

import os  # Взаимодействие с файловой системой
import logging
import requests  # Отправка HTTP запросов
import liwc  # Анализатор лингвистических запросов и подсчета слов
import numpy as np  # Научные вычисления

from transformers import MarianTokenizer, MarianMTModel, AutoModelForSeq2SeqLM

from urllib.parse import urlparse
from pathlib import Path  # Работа с путями в файловой системе

# Типы данных
from typing import List, Tuple, Optional
from types import FunctionType

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
        self._formation_text_model_nn: str = self._formation_model_nn + self._text_modality

        self._load_text_model_weights_hc: str = self._load_model_weights_hc + self._text_modality
        self._load_text_model_weights_nn: str = self._load_model_weights_nn + self._text_modality

        self._model_text_hc_not_formation: str = self._model_hc_not_formation + self._text_modality
        self._model_text_nn_not_formation: str = self._model_nn_not_formation + self._text_modality

        self._load_text_features: str = self._("Загрузка словаря с экспертными признаками ...")
        self._load_text_features_error: str = self._oh + self._(
            "не удалось загрузить словарь с экспертными признаками ..."
        )

        self._load_token_parser_error: str = self._oh + self._("не удалось считать лексикон LIWC ...")

        self._load_translation_model: str = self._(
            "Формирование токенизатора и нейросетевой модели машинного перевода ..."
        )
        self._load_bert_model: str = self._("Формирование токенизатора и нейросетевой модели BERT ...")
        self._load_bert_model_error: str = self._oh + self._(
            "не удалось загрузить токенизатор и нейросетевую модель BERT ..."
        )

        self._get_text_feature_info: str = self._("Извлечение признаков (экспертных и нейросетевых) из текста ...")

        self._text_is_empty: str = self._oh + self._('текстовый файл "{}" пуст ...')


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
        # Нейросетевая модель **tf.keras.Model** для получения оценок по нейросетевым признакам
        self._text_model_nn: Optional[keras.engine.functional.Functional] = None

        # Словарь для формирования экспертных признаков
        self._text_features: str = (
            "https://download.sberdisk.ru/download/file/473268573?token=X3NB5VYGyPn8mjw&filename=LIWC2007.txt",
        )
        # BERT модель
        self._bert_multi_model: str = "https://download.sberdisk.ru/download/file/473319508?token=p8hYNIjxacEARxl&filename=bert-base-multilingual-cased.zip"

        # Нейросетевая модель машинного перевода (RU -> EN)
        self._translation_model: str = "Helsinki-NLP/opus-mt-ru-en"

        self._tokenizer: Optional[MarianTokenizer] = None  # Токенизатор для машинного перевода
        self._traslate_model: Optional[MarianMTModel] = None  # Нейросетевая модель для машинного перевода

        # ----------------------- Только для внутреннего использования внутри класса

        # Названия мультимодальных корпусов
        self.__multi_corpora: List[str] = ["fi", "mupta"]

        self.__parse_text_features: Optional[FunctionType] = None  # Парсинг экспертных признаков
        self.__category_text_features: List[str] = []  # Словарь с экспертными признаками

        # Поддерживаемые текстовые форматы
        self.__supported_text_formats: List[str] = ["txt"]

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

    @property
    def text_model_nn_(self) -> Optional[keras.engine.functional.Functional]:
        """Получение нейросетевой модели **tf.keras.Model** для получения оценок по нейросетевым признакам

        Returns:
            Optional[keras.engine.functional.Functional]: Нейросетевая модель **tf.keras.Model** или None
        """

        return self._text_model_nn

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

    def __load_text_features(
        self,
        url: str,
        force_reload: bool = True,
        info_text: str = "",
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> bool:
        """Загрузка словаря с экспертными признаками

        .. note::
            private (приватный метод)

        Args:
            url (str): Полный путь к файлу с экспертными признаками
            force_reload (bool): Принудительная загрузка файла с экспертными признаками из сети
            info_text (str): Текст для информационного сообщения
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если словарь с экспертными признаками загружен, в обратном случае **False**
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
            self._inv_args(__class__.__name__, self.__load_text_features.__name__, out=out)
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
                    self._other_error(self._load_text_features_error, out=out)
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

    def __load_bert_model(
        self,
        url: str,
        force_reload: bool = True,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> bool:
        """Загрузка нейросетевой модели BERT

        .. note::
            private (приватный метод)

        Args:
            url (str): Полный путь к файлу с нейросетевой модели BERT
            force_reload (bool): Принудительная загрузка файла с нейросетевой модели BERT из сети
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если нейросетевая модель BERT загружена, в обратном случае **False**
        """

        try:
            # Проверка аргументов
            if (
                type(url) is not str
                or not url
                or type(force_reload) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__load_bert_model.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

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
                    self._other_error(self._load_bert_model_error, out=out)
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
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    def _get_text_features(
        self,
        path: str,
        asr: bool = False,
        last: bool = False,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Извлечение признаков из текста (без очистки истории вывода сообщений в ячейке Jupyter)

        .. note::
            protected (защищенный метод)

        Args:
            path (str): Путь к видеофайлу или текст
            asr (bool): Автоматическое распознавание речи
            last (bool): Замена последнего сообщения
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж с двумя np.ndarray:

                1. np.ndarray с экспертными признаками
                2. np.ndarray с нейросетевыми признаками
        """

        try:
            # Проверка аргументов
            if (
                type(path) is not str
                or not path
                or type(asr) is not bool
                or type(last) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self._get_text_features.__name__, last=last, out=out)
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
                self._info(self._get_text_feature_info, out=False)
                if out:
                    self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            try:
                text = ""

                if os.path.isfile(path) is False:
                    raise FileNotFoundError  # Не файл
            except FileNotFoundError:
                try:
                    path_to_text = os.path.join(
                        str(Path(path).parent), Path(path).stem + "." + self.__supported_text_formats[0]
                    )

                    if os.path.isfile(path_to_text) is False:
                        raise FileNotFoundError  # Не текстовый файл
                except FileNotFoundError:
                    text = path.strip()

                    print(text)

                    return np.empty([]), np.empty([])
                except Exception:
                    self._other_error(self._unknown_err, last=last, out=out)
                    return np.empty([]), np.empty([])
                else:
                    try:
                        with open(path_to_text, "r", encoding="utf-8") as file:
                            lines = file.readlines()
                            text = " ".join(line.strip() for line in lines)
                    except Exception:
                        self._other_error(self._unknown_err, last=last, out=out)
                        return np.empty([]), np.empty([])
                    else:
                        try:
                            if not text:
                                raise ValueError
                        except ValueError:
                            self._other_error(
                                self._text_is_empty.format(self._info_wrapper(path_to_text)), last=last, out=out
                            )
                            return np.empty([]), np.empty([])
                        else:
                            print(text)

                            return np.empty([]), np.empty([])
            except Exception:
                self._other_error(self._unknown_err, last=last, out=out)
                return np.empty([]), np.empty([])
            else:
                return np.empty([]), np.empty([])
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

    def load_text_model_nn(
        self, corpus: str = "", show_summary: bool = False, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Формирование нейросетевой архитектуры для получения оценок по нейросетевым признакам

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
            self._inv_args(__class__.__name__, self.load_text_model_nn.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._formation_text_model_nn, last=False, out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            # fi
            if corpus is self.__multi_corpora[0]:
                input_shape = (104, 768)
            # mupta
            elif corpus is self.__multi_corpora[1]:
                input_shape = (414, 768)
            else:
                input_shape = (104, 768)

            input_lstm = tf.keras.Input(shape=input_shape, name="model_nn/input")

            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True), name="model_nn/bilstm_1"
            )(input_lstm)

            x = Attention(use_scale=False, score_mode="dot", name="model_nn/attention")(x, x)

            x = tf.keras.layers.Dense(128, name="model_nn/dence_2")(x)
            x = Addition(name="model_nn/add")(x)
            x = tf.keras.layers.Dense(128, name="model_nn/dence_3")(x)

            x = tf.keras.layers.Dense(5, activation="sigmoid")(x)
            self._text_model_nn = tf.keras.Model(input_lstm, outputs=x, name="model_nn")

            if show_summary and out:
                self._text_model_nn.summary()

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

    def load_text_model_weights_nn(
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
        """

        if runtime:
            self._r_start()

        if self.__load_model_weights(url, force_reload, self._load_text_model_weights_nn, out, False, run) is True:
            try:
                self._text_model_nn.load_weights(self._url_last_filename)
            except Exception:
                self._error(self._model_text_nn_not_formation, out=out)
                return False
            else:
                return True
            finally:
                if runtime:
                    self._r_end(out=out)

        return False

    def load_text_features(
        self, force_reload: bool = True, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Загрузка словаря с экспертными признаками

        Args:
            force_reload (bool): Принудительная загрузка файла с весами нейросетевой модели из сети
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если словарь с экспертными признаками загружен, в обратном случае **False**
        """

        if runtime:
            self._r_start()

        if (
            self.__load_text_features(self._text_features[0], force_reload, self._load_text_features, out, False, run)
            is True
        ):
            try:
                self.__parse_text_features, self.__category_text_features = liwc.load_token_parser(
                    self._url_last_filename
                )
                self.__category_text_features = sorted(self.__category_text_features)
            except Exception:
                self._error(self._load_token_parser_error, out=out)
                return False
            else:
                return True
            finally:
                if runtime:
                    self._r_end(out=out)

    def setup_translation_model(self, out: bool = True, runtime: bool = True, run: bool = True) -> bool:
        """Формирование токенизатора и нейросетевой модели машинного перевода

        Args:
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если токенизатор и нейросетевая модель сформированы, в обратном случае **False**
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        try:
            # Проверка аргументов
            if type(out) is not bool or type(runtime) is not bool or type(run) is not bool:
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.setup_translation_model.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._load_translation_model, last=False, out=out)

            try:
                self._tokenizer = MarianTokenizer.from_pretrained(self._translation_model)
                self._traslate_model = AutoModelForSeq2SeqLM.from_pretrained(self._translation_model).to(self._device)
            except Exception:
                self._other_error(self._unknown_err, out=out)
                return False
            else:
                return True
        finally:
            if runtime:
                self._r_end(out=out)

    def setup_bert_encoder(
        self, force_reload: bool = True, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Формирование токенизатора и нейросетевой модели BERT

        Args:
            force_reload (bool): Принудительная загрузка файла с нейросетевой моделью BERT из сети
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            bool: **True** если токенизатор и нейросетевая модель BERT сформированы, в обратном случае **False**
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        try:
            # Проверка аргументов
            if (
                type(force_reload) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.setup_bert_encoder.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._load_bert_model, last=False, out=out)

            if self.__load_bert_model(self._bert_multi_model, force_reload, out, False, run) is True:
                try:
                    # Распаковка архива
                    res_unzip = self._unzip(
                        path_to_zipfile=os.path.join(self._url_last_filename),
                        new_name=None,
                        force_reload=force_reload,
                    )
                except Exception:
                    self.message_error(self._unknown_err, start=True, out=out)
                    return False
                else:
                    # Файл распакован
                    if res_unzip is True:
                        return True
            else:
                return False
        finally:
            if runtime:
                self._r_end(out=out)

    def get_text_features(
        self,
        path: str,
        asr: bool = False,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ):
        """Извлечение признаков из текста

        Args:
            path (str): Путь к видеофайлу или текст
            asr (bool): Автоматическое распознавание речи
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж с двумя np.ndarray:

                1. np.ndarray с экспертными признаками
                2. np.ndarray с нейросетевыми признаками
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        return self._get_text_features(
            path=path,
            asr=asr,
            last=False,
            out=out,
            runtime=runtime,
            run=run,
        )
