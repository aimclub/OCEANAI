#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Текст
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
import pandas as pd  # Обработка и анализ данных
import subprocess
import torchaudio  # Работа с аудио от Facebook
import re
import gradio

from urllib.error import URLError
from sklearn.metrics import mean_absolute_error
from datetime import datetime  # Работа со временем

from transformers import (
    MarianTokenizer,
    MarianMTModel,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    WhisperForConditionalGeneration,
    BertTokenizer,
    TFBertModel,
)

from keras import backend as K

from urllib.parse import urlparse
from pathlib import Path  # Работа с путями в файловой системе

# Типы данных
from typing import List, Tuple, Optional, Union, Optional, Callable  # Типы данных
from types import FunctionType

from IPython.display import clear_output

# Персональные
from oceanai.modules.lab.download import Download  # Загрузка файлов

# Порог регистрации сообщений TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Игнорировать конкретное предупреждение TensorFlow
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")

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
        self._formation_text_model_b5: str = (
            self._("Формирование нейросетевой архитектуры модели для получения " " оценок персональных качеств")
            + self._text_modality
        )

        self._load_text_model_weights_hc: str = self._load_model_weights_hc + self._text_modality
        self._load_text_model_weights_nn: str = self._load_model_weights_nn + self._text_modality
        self._load_text_model_weights_b5: str = (
            self._("Загрузка весов нейросетевой модели для получения " "оценок персональных качеств")
            + self._text_modality
        )

        self._model_text_hc_not_formation: str = self._model_hc_not_formation + self._text_modality
        self._model_text_nn_not_formation: str = self._model_nn_not_formation + self._text_modality
        self._model_text_not_formation: str = (
            self._oh
            + self._(
                "нейросетевая архитектура модели для получения "
                "оценок по экспертным и нейросетевым признакам не "
                "сформирована"
            )
            + self._text_modality
        )

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
        self._text_model_hc: Optional[tf.keras.Model] = None
        # Нейросетевая модель **tf.keras.Model** для получения оценок по нейросетевым признакам
        self._text_model_nn: Optional[tf.keras.Model] = None
        self._text_model_b5: Optional[tf.keras.Model] = None

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

        self._bert_tokenizer: Optional[BertTokenizer] = None
        self._bert_model: Optional[TFBertModel] = None

        self._path_to_transriber = "openai/whisper-base"

        self._processor: Optional[AutoProcessor] = None
        self._model_transcriptions: Optional[WhisperForConditionalGeneration] = None

        # ----------------------- Только для внутреннего использования внутри класса

        # Названия мультимодальных корпусов
        self.__multi_corpora: List[str] = ["fi", "mupta"]

        self.__lang_traslate: List[str] = ["ru", "en"]
        self.lang_traslate: List[str] = self.__lang_traslate

        self.__parse_text_features: Optional[FunctionType] = None  # Парсинг экспертных признаков
        self.__category_text_features: List[str] = []  # Словарь с экспертными признаками

        # Поддерживаемые текстовые форматы
        self.__supported_text_formats: List[str] = ["txt"]

        self.__contractions_dict = {
            "ain't": "are not",
            "'s": " is",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "‘cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how're": "how are",
            "i'd": "i would",
            "i'd've": "i would have",
            "i'll": "i will",
            "i 'll": "i will",
            "i'll've": "i will have",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "that'd": "that would",
            "that'll": "that will",
            "that'd've": "that would have",
            "there'd": "there would",
            "there'd've": "there would have",
            "there'll": "there will",
            "there're": "there are",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what've": "what have",
            "what'd": "what would",
            "when've": "when have",
            "where'd": "where did",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who've": "who have",
            "who'd": "who would",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "why'd": "why would",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "ya'll": "you all",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you would",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you will have",
            "you're": "you are",
            "you've": "you have",
        }

        self.__forced_decoder_ids: Optional[List[Tuple[int, int]]] = None

        self.__text_pred: str = ""

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
    def text_model_hc_(self) -> Optional[tf.keras.Model]:
        """Получение нейросетевой модели **tf.keras.Model** для получения оценок по экспертным признакам

        Returns:
            Optional[tf.keras.Model]: Нейросетевая модель **tf.keras.Model** или None
        """

        return self._text_model_hc

    @property
    def text_model_nn_(self) -> Optional[tf.keras.Model]:
        """Получение нейросетевой модели **tf.keras.Model** для получения оценок по нейросетевым признакам

        Returns:
            Optional[tf.keras.Model]: Нейросетевая модель **tf.keras.Model** или None
        """

        return self._text_model_nn

    @property
    def text_model_b5_(self) -> Optional[tf.keras.Model]:
        """Получение нейросетевой модели **tf.keras.Model** для получения оценок персональных качеств

        Returns:
            Optional[tf.keras.Model]: Нейросетевая модель **tf.keras.Model** или None
        """

        return self._text_model_b5

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

    def __translate_and_extract_features(
        self,
        text: str,
        lang: str,
        show_text: bool = False,
        last: bool = False,
        out: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Извлечение признаков из текста

        .. note::
            private (приватный метод)

        Args:
            text (str): Текст
            lang (str): Язык
            show_text (bool): Отображение текста
            last (bool): Замена последнего сообщения
            out (bool): Отображение

        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж с двумя np.ndarray:

                1. np.ndarray с экспертными признаками
                2. np.ndarray с нейросетевыми признаками
        """

        contractions_re = re.compile("(%s)" % "|".join(self.__contractions_dict.keys()))

        expand_contractions = lambda s: contractions_re.sub(lambda match: self.__contractions_dict[match.group(0)], s)

        get_norm_text = lambda text: re.sub(
            r"(?<=[.,])(?=[^\s])",
            " ",
            re.sub(
                r"\[[^\[\]]+\]",
                "",
                expand_contractions(
                    re.sub(
                        r'[.,"\'?:!/;]',
                        "",
                        re.sub(r"((?<=^)(\s*?(\-)??))|(((\-)??\s*?)(?=$))", "", text.lower().strip()),
                    )
                ),
            ),
        )

        norm_features = lambda feature, length: np.pad(
            feature[:length, :], ((0, max(0, length - feature.shape[0])), (0, 0)), "constant"
        )

        if lang == self.__lang_traslate[0]:
            if len(text) > 700:
                translation = ""
                for sentence in text.split(".")[:-1]:
                    input_ids = self._tokenizer.encode(sentence + ".", return_tensors="pt")
                    outputs = self._traslate_model.generate(input_ids.to(self._device), max_new_tokens=4000)
                    translation += self._tokenizer.decode(outputs[0], skip_special_tokens=True) + " "
            else:
                input_ids = self._tokenizer.encode(text, return_tensors="pt")
                outputs = self._traslate_model.generate(input_ids.to(self._device), max_new_tokens=4000)
                translation = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            translation = re.sub(r"(?<=[.,])(?=[^\s])", r" ", translation)
        else:
            translation = text

        text = get_norm_text(text)
        translation = get_norm_text(translation)

        encoded_input = self._bert_tokenizer(text, return_tensors="tf")
        dict_new = {}
        if encoded_input["input_ids"].shape[1] > 512:
            dict_new["input_ids"] = encoded_input["input_ids"][:, :512]
            dict_new["token_type_ids"] = encoded_input["token_type_ids"][:, :512]
            dict_new["attention_mask"] = encoded_input["attention_mask"][:, :512]
            encoded_input = dict_new
        features_bert = self._bert_model(encoded_input)[0][0]

        features_liwc = []
        for i in translation.split(" "):
            curr_f = np.zeros((1, 64))
            for j in self.__parse_text_features(i):
                curr_f[:, self.__category_text_features.index(j)] += 1
            features_liwc.extend(curr_f)

        features_liwc = np.array(features_liwc)

        if lang == self.__lang_traslate[0]:
            features_bert = norm_features(features_bert, 414)
            features_liwc = norm_features(features_liwc, 365)
        elif lang == self.__lang_traslate[1]:
            features_bert = norm_features(features_bert, 104)
            features_liwc = norm_features(features_liwc, 89)

        if not show_text:
            text = None

        if last is False:
            # Статистика извлеченных признаков из текста
            self._stat_text_features(
                last=last,
                out=out,
                shape_hc_features=np.array(features_liwc).shape,
                shape_nn_features=np.array(features_bert).shape,
                text=text,
            )

        return features_liwc, features_bert

    def __process_audio_and_extract_features(
        self, path: str, win: int, lang: str, show_text: bool, last: bool, out: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._processor:
            self._processor = AutoProcessor.from_pretrained(self._path_to_transriber)
            self._model_transcriptions = WhisperForConditionalGeneration.from_pretrained(self._path_to_transriber).to(
                self._device
            )

        if lang == self.__lang_traslate[0]:
            self.__forced_decoder_ids = self._processor.get_decoder_prompt_ids(language=lang, task="transcribe")

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
                    wav, sr = torchaudio.load(path_to_wav)

                    if wav.size(0) > 1:
                        wav = wav.mean(dim=0, keepdim=True)

                    if sr != 16000:
                        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                        wav = transform(wav)
                        sr = 16000

                    wav = wav.squeeze(0)

                    for start in range(0, len(wav), win):
                        inputs = self._processor(wav[start : start + win], sampling_rate=16000, return_tensors="pt")
                        input_features = inputs.input_features.to(self._device)
                        if lang == self.__lang_traslate[0]:
                            generated_ids = self._model_transcriptions.generate(
                                input_features=input_features,
                                forced_decoder_ids=self.__forced_decoder_ids,
                                max_new_tokens=448,
                            )
                        elif lang == self.__lang_traslate[1]:
                            generated_ids = self._model_transcriptions.generate(
                                input_features=input_features, max_new_tokens=448
                            )
                        transcription = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        self.__text_pred += transcription

                    return self.__translate_and_extract_features(self.__text_pred, lang, show_text, last, out)
        else:
            wav, sr = torchaudio.load(path_to_wav)

            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)

            if sr != 16000:
                transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = transform(wav)
                sr = 16000

            wav = wav.squeeze(0)

            for start in range(0, len(wav), win):
                inputs = self._processor(wav[start : start + win], sampling_rate=16000, return_tensors="pt")
                input_features = inputs.input_features.to(self._device)
                if lang == self.__lang_traslate[0]:
                    generated_ids = self._model_transcriptions.generate(
                        input_features=input_features, forced_decoder_ids=self.__forced_decoder_ids
                    )
                elif lang == self.__lang_traslate[1]:
                    generated_ids = self._model_transcriptions.generate(input_features=input_features)
                transcription = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                self.__text_pred += transcription

            return self.__translate_and_extract_features(self.__text_pred, lang, show_text, last, out)

    def __load_text_model_b5(self, show_summary: bool = False, out: bool = True) -> Optional[tf.keras.Model]:
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
            self._inv_args(__class__.__name__, self.__load_text_model_b5.__name__, out=out)
            return None
        else:
            input_1 = tf.keras.Input(shape=(5,))
            input_2 = tf.keras.Input(shape=(5,))
            X = tf.keras.backend.concatenate((input_1, input_2), axis=1)
            X = tf.keras.layers.Dense(5, activation="sigmoid")(X)

            model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=X)

            if show_summary and out:
                model.summary()

            return model

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    def _get_text_features(
        self,
        path: str,
        asr: bool = False,
        lang: str = "ru",
        show_text: bool = False,
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
            lang (str): Язык
            show_text (bool): Отображение текста
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
                (type(path) is not str or not path) and (type(path) is not gradio.utils.NamedString)
                or type(asr) is not bool
                or not isinstance(lang, str)
                or lang not in self.__lang_traslate
                or type(last) is not bool
                or type(show_text) is not bool
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

            win = 400000

            try:
                self.__text_pred = ""

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
                    self.__text_pred = path.strip()

                    return self.__translate_and_extract_features(self.__text_pred, lang, show_text, last, out)

                except Exception:
                    self._other_error(self._unknown_err, last=last, out=out)
                    return np.empty([]), np.empty([])
                else:
                    try:
                        with open(path_to_text, "r", encoding="utf-8") as file:
                            lines = file.readlines()
                            self.__text_pred = " ".join(line.strip() for line in lines)
                    except Exception:
                        self._other_error(self._unknown_err, last=last, out=out)
                        return np.empty([]), np.empty([])
                    else:
                        try:
                            if not self.__text_pred:
                                raise ValueError
                        except ValueError:
                            self._other_error(
                                self._text_is_empty.format(self._info_wrapper(path_to_text)), last=last, out=out
                            )

                            return np.empty([]), np.empty([])
                        else:
                            return np.empty([]), np.empty([])
            except Exception:
                self._other_error(self._unknown_err, last=last, out=out)
                return np.empty([]), np.empty([])
            else:
                try:
                    path_to_text = os.path.join(
                        str(Path(path).parent), Path(path).stem + "." + self.__supported_text_formats[0]
                    )

                    if os.path.isfile(path_to_text) is False:
                        raise FileNotFoundError  # Не текстовый файл
                except FileNotFoundError:
                    return self.__process_audio_and_extract_features(path, win, lang, show_text, last, out)
                except Exception:
                    self._other_error(self._unknown_err, last=last, out=out)
                    return np.empty([]), np.empty([])
                else:
                    try:
                        with open(path_to_text, "r", encoding="utf-8") as file:
                            lines = file.readlines()
                            self.__text_pred = " ".join(line.strip() for line in lines)
                    except Exception:
                        self._other_error(self._unknown_err, last=last, out=out)
                        return np.empty([]), np.empty([])
                    else:
                        try:
                            if not self.__text_pred:
                                raise ValueError
                        except ValueError:
                            self._other_error(
                                self._text_is_empty.format(self._info_wrapper(path_to_text)), last=last, out=out
                            )
                            return np.empty([]), np.empty([])
                        else:
                            if asr:
                                return self.__process_audio_and_extract_features(path, win, lang, show_text, last, out)
                            else:
                                return self.__translate_and_extract_features(
                                    self.__text_pred, lang, show_text, last, out
                                )
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

            input_lstm = tf.keras.Input(shape=input_shape, name="model_hc_input")

            x_lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True), name="model_hc_bilstm_1"
            )(input_lstm)

            x_attention = Attention(use_scale=False, score_mode="dot", name="model_hc_attention")(x_lstm, x_lstm)

            x_dence = tf.keras.layers.Dense(32 * 2, name="model_hc_dence_2", activation="relu")(input_lstm)
            x_dence = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True), name="model_hc_bilstm_2"
            )(x_dence)

            x = tf.keras.layers.Add()([x_lstm, x_attention, x_dence])

            # m = tf.reduce_mean(x, axis=1)
            # s = tf.reduce_std(x, axis=1)
            # m = tf.keras.backend.mean(x, axis=1)
            # s = tf.keras.backend.std(x, axis=1)
        # print('add', K.concatenate((m, s), axis=1).shape)
            # x = tf.concate((m, s), axis=1)

            # print(x.shape)

            x = Addition(name="model_hc_add")(x)

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

            input_lstm = tf.keras.Input(shape=input_shape, name="model_nn_input")

            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True), name="model_nn_bilstm_1"
            )(input_lstm)

            x = Attention(use_scale=False, score_mode="dot", name="model_nn_attention")(x, x)

            x = tf.keras.layers.Dense(128, name="model_nn_dence_2")(x)
            x = Addition(name="model_nn_add")(x)
            x = tf.keras.layers.Dense(128, name="model_nn_dence_3")(x)

            x = tf.keras.layers.Dense(5, activation="sigmoid")(x)
            self._text_model_nn = tf.keras.Model(input_lstm, outputs=x, name="model_nn")

            if show_summary and out:
                self._text_model_nn.summary()

            if runtime:
                self._r_end(out=out)

            return True

    def load_text_model_b5(
        self, show_summary: bool = False, out: bool = True, runtime: bool = True, run: bool = True
    ) -> bool:
        """Формирование нейросетевой архитектуры модели для получения результатов оценки персональных качеств

        Args:
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
                type(show_summary) is not bool
                or type(out) is not bool
                or type(runtime) is not bool
                or type(run) is not bool
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.load_text_model_b5.__name__, out=out)
            return False
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return False

            if runtime:
                self._r_start()

            # Информационное сообщение
            self._info(self._formation_text_model_b5, last=False, out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            self._text_model_b5 = self.__load_text_model_b5()

            if show_summary and out:
                self._text_model_b5.summary()

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
                self._text_model_hc = tf.keras.models.Model(
                    inputs=self._text_model_hc.input,
                    outputs=[self._text_model_hc.output, self._text_model_hc.get_layer("model_hc_add").output],
                )

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
                self._text_model_nn = tf.keras.models.Model(
                    inputs=self._text_model_nn.input,
                    outputs=[self._text_model_nn.output, self._text_model_nn.get_layer("model_nn_dence_3").output],
                )
            except Exception:
                self._error(self._model_text_nn_not_formation, out=out)
                return False
            else:
                return True
            finally:
                if runtime:
                    self._r_end(out=out)

        return False

    def load_text_model_weights_b5(
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

        if self.__load_model_weights(url, force_reload, self._load_text_model_weights_b5, out, False, run) is True:
            try:
                self._text_model_b5.load_weights(self._url_last_filename)
            except Exception:
                self._error(self._model_text_not_formation, out=out)
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
                    self._other_error(self._unknown_err, out=out)
                    return False
                else:
                    # Файл распакован
                    if res_unzip is True:
                        try:
                            self._bert_tokenizer = BertTokenizer.from_pretrained(Path(self._url_last_filename).stem)
                            self._bert_model = TFBertModel.from_pretrained(Path(self._url_last_filename).stem)
                        except Exception:
                            self._other_error(self._unknown_err, out=out)
                            return False
                        else:
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
        lang: str = "ru",
        show_text: bool = False,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ):
        """Извлечение признаков из текста

        Args:
            path (str): Путь к видеофайлу или текст
            asr (bool): Автоматическое распознавание речи
            lang (str): Язык
            show_text (bool): Отображение текста
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
            lang=lang,
            show_text=show_text,
            last=False,
            out=out,
            runtime=runtime,
            run=run,
        )

    def get_text_union_predictions(
        self,
        depth: int = 1,
        recursive: bool = False,
        asr: bool = False,
        lang: str = "ru",
        accuracy=True,
        url_accuracy: str = "",
        logs: bool = True,
        out: bool = True,
        runtime: bool = True,
        run: bool = True,
    ) -> bool:
        """Получения прогнозов по тексту

        Args:
            depth (int): Глубина иерархии для получения данных
            recursive (bool): Рекурсивный поиск данных
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
                or type(out) is not bool
                or type(recursive) is not bool
                or type(asr) is not bool
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
            self._inv_args(__class__.__name__, self.get_text_union_predictions.__name__, out=out)
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
                    get_text_union_predictions_info = self._get_union_predictions_info + self._get_accuracy_info
                else:
                    get_text_union_predictions_info = self._get_union_predictions_info

                get_text_union_predictions_info += self._text_modality

                # Вычисление точности
                if accuracy is True:
                    # Информационное сообщение
                    self._info(get_text_union_predictions_info, out=out)

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
                            get_text_union_predictions_info,
                            i,
                            self.__local_path(curr_path),
                            self.__len_paths,
                            True,
                            last,
                            out,
                        )

                        hc_features, nn_features = self.get_text_features(
                            path=str(curr_path.resolve()),  # Путь к видеофайлу
                            asr=asr,  # Распознавание речи
                            lang=lang,  # Выбор языка
                            show_text=True,  # Отображение текста
                            out=False,  # Отображение
                            runtime=False,  # Подсчет времени выполнения
                            run=run,  # Блокировка выполнения
                        )

                        hc_features = np.expand_dims(hc_features, axis=0)
                        nn_features = np.expand_dims(nn_features, axis=0)

                        # Признаки из текста извлечены
                        if len(hc_features) > 0 and len(nn_features) > 0:
                            # Коды ошибок нейросетевых моделей
                            code_error_pred_hc = -1
                            code_error_pred_nn = -1

                            try:
                                # Оправка экспертных признаков в нейросетевую модель
                                pred_hc, _ = self.text_model_hc_(np.array(hc_features, dtype=np.float16))
                            except TypeError:
                                code_error_pred_hc = 1
                            except Exception:
                                code_error_pred_hc = 2

                            try:
                                # Отправка нейросетевых признаков в нейросетевую модель
                                pred_nn, _ = self.text_model_nn_(np.array(nn_features, dtype=np.float16))
                            except TypeError:
                                code_error_pred_nn = 1
                            except Exception:
                                code_error_pred_nn = 2

                            if code_error_pred_hc != -1 and code_error_pred_nn != -1:
                                self._error(self._model_text_not_formation, out=out)
                                return False

                            if code_error_pred_hc != -1:
                                self._error(self._model_text_hc_not_formation, out=out)
                                return False

                            if code_error_pred_nn != -1:
                                self._error(self._model_text_nn_not_formation, out=out)
                                return False

                            # pred_hc = pred_hc.numpy()[0]
                            # pred_nn = pred_nn.numpy()[0]

                            final_pred = self._text_model_b5([pred_hc, pred_nn]).numpy()[0].tolist()

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
                        get_text_union_predictions_info,
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

                        name_logs_file = self.get_text_union_predictions.__name__

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
