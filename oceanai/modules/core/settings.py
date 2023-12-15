#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Настройки
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
import re  # Регулярные выражения

from typing import List

# Персональные
from oceanai.modules.core.messages import Messages  # Сообщения

# ######################################################################################################################
# Константы
# ######################################################################################################################

COLOR_INFO: str = "#1776D2"  # Цвет текста содержащего информацию (шестнадцатеричный код)
COLOR_SIMPLE: str = "#666"  # Цвет обычного текста (шестнадцатеричный код)
COLOR_ERR: str = "#FF0000"  # Цвет текста содержащего ошибку (шестнадцатеричный код)
COLOR_TRUE: str = "#008001"  # Цвет текста содержащего положительную информацию (шестнадцатеричный код)
BOLD_TEXT: bool = True  # Жирное начертание текста
CHUNK_SIZE: int = 1000000  # Размер загрузки файла из сети за 1 шаг
EXT: List[str] = []  # Расширения искомых файлов
IGNORE_DIRS: List[str] = []  # Директории не входящие в выборку
# Названия ключей для DataFrame набора данных
KEYS_DATASET: List[str] = ["Path", "Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Non-Neuroticism"]
NUM_TO_DF_DISPLAY: int = 30  # Количество строк для отображения в таблицах
PATH_TO_DATASET: str = ""  # Директория набора данных
PATH_TO_SAVE: str = "./models"  # Директория для сохранения данных
PATH_TO_LOGS: str = "./logs"  # Директория для сохранения LOG файлов
TEXT_RUNTIME: str = ""  # Текст времени выполнения


# ######################################################################################################################
# Настройки
# ######################################################################################################################
@dataclass
class Settings(Messages):
    """Класс для настроек

    Args:
        lang (str): Смотреть :attr:`~oceanai.modules.core.language.Language.lang`
        color_simple (str): Цвет обычного текста (шестнадцатеричный код)
        color_info (str): Цвет текста содержащего информацию (шестнадцатеричный код)
        color_err (str): Цвет текста содержащего ошибку (шестнадцатеричный код)
        color_true (str): Цвет текста содержащего положительную информацию (шестнадцатеричный код)
        bold_text (bool): Жирное начертание текста
        num_to_df_display (int): Количество строк для отображения в таблицах
        text_runtime (str): Текст времени выполнения
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    # Цвет текстов
    color_simple: str = COLOR_SIMPLE
    """
    str: Цвет обычного текста (шестнадцатеричный код)

    .. dropdown:: Примеры

        :bdg-success:`Верно` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 1
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings()
            print(settings.color_simple, settings.color_simple_)

        .. output-cell::
            :execution-count: 1
            :linenos:

            #666 #666

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 2
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_simple = '#666')
            print(settings.color_simple, settings.color_simple_)

        .. output-cell::
            :execution-count: 2
            :linenos:

            #666 #666

        :bdg-light:`-- 3 --`

        .. code-cell:: python
            :execution-count: 3
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_simple = '#222')
            print(settings.color_simple, settings.color_simple_)

        .. output-cell::
            :execution-count: 3
            :linenos:

            #222 #222

        :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 4
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_simple = 1)
            print(settings.color_simple, settings.color_simple_)

        .. output-cell::
            :execution-count: 4
            :linenos:

            #666 #666

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 5
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_simple = {1, 2, 3})
            print(settings.color_simple, settings.color_simple_)

        .. output-cell::
            :execution-count: 5
            :linenos:

            #666 #666
    """

    color_info: str = COLOR_INFO
    """
    str: Цвет текста содержащего информацию (шестнадцатеричный код)

    .. dropdown:: Примеры

        :bdg-success:`Верно` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 1
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings()
            print(settings.color_info, settings.color_info_)

        .. output-cell::
            :execution-count: 1
            :linenos:

            #1776D2 #1776D2

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 2
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_info = '#1776D2')
            print(settings.color_info, settings.color_info_)

        .. output-cell::
            :execution-count: 2
            :linenos:

            #1776D2 #1776D2

        :bdg-light:`-- 3 --`

        .. code-cell:: python
            :execution-count: 3
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_info = '#42F')
            print(settings.color_info, settings.color_info_)

        .. output-cell::
            :execution-count: 3
            :linenos:

            #42F #42F

        :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 4
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_info = 1)
            print(settings.color_info, settings.color_info_)

        .. output-cell::
            :execution-count: 4
            :linenos:

            #1776D2 #1776D2

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 5
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_info = [])
            print(settings.color_info, settings.color_info_)

        .. output-cell::
            :execution-count: 5
            :linenos:

            #1776D2 #1776D2
    """

    color_err: str = COLOR_ERR
    """
    str: Цвет текста содержащего ошибку (шестнадцатеричный код)

    .. dropdown:: Примеры

        :bdg-success:`Верно` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 1
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings()
            print(settings.color_err, settings.color_err_)

        .. output-cell::
            :execution-count: 1
            :linenos:

            #FF0000 #FF0000

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 2
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_err = 'FF0000')
            print(settings.color_err, settings.color_err_)

        .. output-cell::
            :execution-count: 2
            :linenos:

            #FF0000 #FF0000

        :bdg-light:`-- 3 --`

        .. code-cell:: python
            :execution-count: 3
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_err = '#FF0')
            print(settings.color_err, settings.color_err_)

        .. output-cell::
            :execution-count: 3
            :linenos:

            #FF0 #FF0

        :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 4
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_err = 1)
            print(settings.color_err, settings.color_err_)

        .. output-cell::
            :execution-count: 4
            :linenos:

            #FF0000 #FF0000

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 5
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_err = [])
            print(settings.color_err, settings.color_err_)

        .. output-cell::
            :execution-count: 5
            :linenos:

            #FF0000 #FF0000
    """

    color_true: str = COLOR_TRUE
    """
    str: Цвет текста содержащего положительную информацию (шестнадцатеричный код)

    .. dropdown:: Примеры

        :bdg-success:`Верно` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 1
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings()
            print(settings.color_true, settings.color_true_)

        .. output-cell::
            :execution-count: 1
            :linenos:

            #008001 #008001

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 2
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_true = '#003332')
            print(settings.color_true, settings.color_true_)

        .. output-cell::
            :execution-count: 2
            :linenos:

            #003332 #003332

        :bdg-light:`-- 3 --`

        .. code-cell:: python
            :execution-count: 3
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_true = '#42F')
            print(settings.color_true, settings.color_true_)

        .. output-cell::
            :execution-count: 3
            :linenos:

            #42F #42F

        :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 4
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_true = 1)
            print(settings.color_true, settings.color_true_)

        .. output-cell::
            :execution-count: 4
            :linenos:

            #008001 #008001

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 5
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(color_true = [])
            print(settings.color_true, settings.color_true_)

        .. output-cell::
            :execution-count: 5
            :linenos:

            #008001 #008001
    """

    bold_text: bool = BOLD_TEXT
    """
    bool: Жирное начертание текста

    .. dropdown:: Примеры

        :bdg-success:`Верно` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 1
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(lang = 'ru')
            print(settings.bold_text, settings.bold_text_)

        .. output-cell::
            :execution-count: 1
            :linenos:

            True True

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 2
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(lang = 'ru', bold_text = True)
            print(settings.bold_text, settings.bold_text_)

        .. output-cell::
            :execution-count: 2
            :linenos:

            True True

        :bdg-light:`-- 3 --`

        .. code-cell:: python
            :execution-count: 3
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(lang = 'ru', bold_text = False)
            print(settings.bold_text, settings.bold_text_)

        .. output-cell::
            :execution-count: 3
            :linenos:

            False False

        :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 4
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(lang = 'ru', bold_text = 1)
            print(settings.bold_text, settings.bold_text_)

        .. output-cell::
            :execution-count: 4
            :linenos:

            True True

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 5
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(lang = 'ru', bold_text = 'какой-то_текст')
            print(settings.bold_text, settings.bold_text_)

        .. output-cell::
            :execution-count: 5
            :linenos:

            True True
    """

    text_runtime: str = TEXT_RUNTIME
    """
    str: Текст времени выполнения

    .. dropdown:: Примеры

        :bdg-success:`Верно` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 1
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings()
            print(settings.text_runtime, settings.text_runtime_)

        .. output-cell::
            :execution-count: 1
            :linenos:

            Время выполнения Время выполнения

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 2
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(text_runtime = 'Код выполнился за')
            print(settings.text_runtime, settings.text_runtime_)

        .. output-cell::
            :execution-count: 2
            :linenos:

            Код выполнился за Код выполнился за

        :bdg-light:`-- 3 --`

        .. code-cell:: python
            :execution-count: 3
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(text_runtime = 'Время выполнения')
            print(settings.text_runtime, settings.text_runtime_)

        .. output-cell::
            :execution-count: 3
            :linenos:

            Время выполнения Время выполнения

        :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 4
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(text_runtime = 1)
            print(settings.text_runtime, settings.text_runtime_)

        .. output-cell::
            :execution-count: 4
            :linenos:

            Время выполнения Время выполнения

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 5
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(text_runtime = {1, 2, 3})
            print(settings.text_runtime, settings.text_runtime_)

        .. output-cell::
            :execution-count: 5
            :linenos:

            Время выполнения Время выполнения
    """

    num_to_df_display: int = NUM_TO_DF_DISPLAY
    """
    int: Количество строк для отображения в таблицах

    .. dropdown:: Примеры

        :bdg-success:`Верно` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 1
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings()
            print(settings.num_to_df_display, settings.num_to_df_display_)

        .. output-cell::
            :execution-count: 1
            :linenos:

            30 30

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 2
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(num_to_df_display = 30)
            print(settings.num_to_df_display, settings.num_to_df_display_)

        .. output-cell::
            :execution-count: 2
            :linenos:

            30 30

        :bdg-light:`-- 3 --`

        .. code-cell:: python
            :execution-count: 3
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(num_to_df_display = 50)
            print(settings.num_to_df_display, settings.num_to_df_display_)

        .. output-cell::
            :execution-count: 3
            :linenos:

            50 50

        :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 4
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(num_to_df_display = 0)
            print(settings.num_to_df_display, settings.num_to_df_display_)

        .. output-cell::
            :execution-count: 4
            :linenos:

            30 30

        :bdg-light:`-- 2 --`

        .. code-cell:: python
            :execution-count: 5
            :linenos:

            from oceanai.modules.core.settings import Settings

            settings = Settings(num_to_df_display = 'какой-то_текст')
            print(settings.num_to_df_display, settings.num_to_df_display_)

        .. output-cell::
            :execution-count: 5
            :linenos:

            30 30
    """

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self.__re_search_color: str = r"^#(?:[0-9a-fA-F]{3}){1,2}$"  # Регулярное выражение для корректности ввода цвета

        # Цвет текстов
        self.__color_simple_true: int = 0  # Счетчик изменения текста
        self.color_simple_: str = self.color_simple  # Обычный текст

        self.__color_info_true: int = 0  # Счетчик изменения текста
        self.color_info_: str = self.color_info  # Цвет текста содержащего информацию

        self.__color_true_true: int = 0  # Счетчик изменения текста
        self.color_true_: str = self.color_true  # Цвет текста содержащего положительную информацию

        self.__color_err_true: int = 0  # Счетчик изменения текста
        self.color_err_: str = self.color_err  # Цвет текста содержащего ошибку

        self.__bold_text_true: int = 0  # Счетчик изменения начертания текста
        self.bold_text_: bool = self.bold_text  # Жирное начертание текста

        self.__text_runtime_true: int = 0  # Счетчик изменения текста
        self.text_runtime_: str = self.text_runtime  # Текст времени выполнения

        self.__num_to_df_display_true: int = 0  # Счетчик изменения количества строк для отображения в таблицах
        self.num_to_df_display_: int = self.num_to_df_display  # Количество строк для отображения в таблицах

        self.chunk_size_: int = CHUNK_SIZE  # Размер загрузки файла из сети за 1 шаг

        self.path_to_save_: str = PATH_TO_SAVE  # Директория для сохранения данных
        self.path_to_dataset_: str = PATH_TO_DATASET  # Директория набора данных
        self.path_to_logs_: str = PATH_TO_LOGS  # Директория для сохранения LOG файлов

        self.ext_: List[str] = EXT  # Расширения искомых файлов

        self.ignore_dirs_: List[str] = IGNORE_DIRS  # Директории не входящие в выборку

        self.keys_dataset_: List[str] = KEYS_DATASET  # Названия ключей для DataFrame набора данных

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def color_simple_(self) -> str:
        """Получение/установка цвета обычного текста

        Args:
            (str): Шестнадцатеричный код

        Returns:
            str: Шестнадцатеричный код

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings(color_simple = '#111')
                print(settings.color_simple_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                #111

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.color_simple_ = '#444'
                print(settings.color_simple_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                #444

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.color_simple_ = 1
                print(settings.color_simple_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                #666

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.color_simple_ = ()
                print(settings.color_simple_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                #666
        """

        return self.color_simple

    @color_simple_.setter
    def color_simple_(self, color: str) -> None:
        """Установка цвета обычного текста"""

        try:
            # Проверка аргументов
            match = re.search(self.__re_search_color, color)
            if not match:
                raise TypeError
        except TypeError:
            if self.__color_simple_true == 0:
                self.color_simple = COLOR_SIMPLE
        else:
            self.color_simple = color
            self.__color_simple_true += 1  # Увеличение счетчика изменения цвета текста

    @property
    def color_info_(self) -> str:
        """Получение/установка цвета текста содержащего информацию

        Args:
            (str): Шестнадцатеричный код

        Returns:
            str: Шестнадцатеричный код

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings(color_info = '#1776D2')
                print(settings.color_info_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                #1776D2

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.color_info_ = '#42F'
                print(settings.color_info_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                #42F

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.color_info_ = 1
                print(settings.color_info_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                #1776D2

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.color_info_ = ()
                print(settings.color_info_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                #1776D2
        """

        return self.color_info

    @color_info_.setter
    def color_info_(self, color: str) -> None:
        """Установка цвета текста содержащего информацию"""

        try:
            # Проверка аргументов
            match = re.search(self.__re_search_color, color)
            if not match:
                raise TypeError
        except TypeError:
            if self.__color_info_true == 0:
                self.color_info = COLOR_INFO
        else:
            self.color_info = color
            self.__color_info_true += 1  # Увеличение счетчика изменения цвета текста

    @property
    def color_true_(self) -> str:
        """Получение/установка цвета текста содержащего положительную информацию

        Args:
            (str): Шестнадцатеричный код

        Returns:
            str: Шестнадцатеричный код

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings(color_true = '#008001')
                print(settings.color_true_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                #008001

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.color_true_ = '#42F'
                print(settings.color_true_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                #42F

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.color_true = 1
                print(settings.color_true)

            .. output-cell::
                :execution-count: 3
                :linenos:

                #008001

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.color_true_ = ()
                print(settings.color_true_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                #008001
        """

        return self.color_true

    @color_true_.setter
    def color_true_(self, color: str) -> None:
        """Установка цвета текста содержащего положительную информацию"""

        try:
            # Проверка аргументов
            match = re.search(self.__re_search_color, color)
            if not match:
                raise TypeError
        except TypeError:
            if self.__color_true_true == 0:
                self.color_true = COLOR_TRUE
        else:
            self.color_true = color
            self.__color_true_true += 1  # Увеличение счетчика изменения цвета текста

    @property
    def color_err_(self) -> str:
        """Получение/установка цвета текста содержащего ошибку

        Args:
            (str): Шестнадцатеричный код

        Returns:
            str: Шестнадцатеричный код

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings(color_err = '#C22931')
                print(settings.color_err_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                #C22931

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.color_err_ = '#FF0'
                print(settings.color_err_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                #FF0

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.color_err_ = 1
                print(settings.color_err_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                #FF0000

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.color_err_ = {}
                print(settings.color_err_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                #FF0000
        """

        return self.color_err

    @color_err_.setter
    def color_err_(self, color: str) -> None:
        """Установка цвета текста содержащего ошибку"""

        try:
            # Проверка аргументов
            match = re.search(self.__re_search_color, color)
            if not match:
                raise TypeError
        except TypeError:
            if self.__color_err_true == 0:
                self.color_err = COLOR_ERR
        else:
            self.color_err = color
            self.__color_err_true += 1  # Увеличение счетчика изменения цвета текста

    @property
    def bold_text_(self) -> bool:
        """Получение/установка жирного начертания текста

        Args:
            (bool): **True** или **False**

        Returns:
            bool: **True** или **False**

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings(lang = 'ru', bold_text = True)
                print(settings.bold_text_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                True

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings(lang = 'ru', bold_text = True)
                settings.bold_text_ = False
                print(settings.bold_text_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                False

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings(lang = 'en', bold_text = False)
                settings.bold_text_ = 1
                print(settings.bold_text_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                False

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings(lang = 'ru', bold_text = True)
                settings.bold_text_ = 'какой-то_текст'
                print(settings.bold_text_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                True
        """

        return self.bold_text

    @bold_text_.setter
    def bold_text_(self, bold: bool) -> None:
        """Установка жирного начертания текста"""

        try:
            # Проверка аргументов
            if type(bold) is not bool:
                raise TypeError
        except TypeError:
            if self.__bold_text_true == 0:
                self.bold_text = BOLD_TEXT
        else:
            self.bold_text = bold
            self.__bold_text_true += 1  # Увеличение счетчика изменения начертания текста

    @property
    def text_runtime_(self) -> str:
        """Получение/установка текста времени выполнения

        Args:
            (str): Текст

        Returns:
            str: Текст

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings(text_runtime = 'Время выполнения')
                print(settings.text_runtime_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                Время выполнения

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.text_runtime_ = 'Код выполнился за'
                print(settings.text_runtime_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                Код выполнился за

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.text_runtime_ = 1
                print(settings.text_runtime_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                Время выполнения

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.text_runtime_ = ()
                print(settings.text_runtime_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                Время выполнения
        """

        return self.text_runtime

    @text_runtime_.setter
    def text_runtime_(self, text: str) -> None:
        """Установка текста времени выполнения"""

        try:
            # Проверка аргументов
            if type(text) is not str or len(text) < 1:
                raise TypeError
        except TypeError:
            if self.__text_runtime_true == 0:
                self.text_runtime = self._text_runtime
        else:
            self.text_runtime = text
            self.__text_runtime_true += 1  # Увеличение счетчика изменения текста времени выполнения

    @property
    def num_to_df_display_(self) -> int:
        """Получение/установка количества строк для отображения в таблицах

        Args:
            (int): Количество строк

        Returns:
            int: Количество строк

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings(num_to_df_display = 30)
                print(settings.num_to_df_display_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                30

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.num_to_df_display_ = 50
                print(settings.num_to_df_display_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                50

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.num_to_df_display_ = 0
                print(settings.num_to_df_display_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                30

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.num_to_df_display_ = ()
                print(settings.num_to_df_display_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                30
        """

        return self.num_to_df_display

    # Установка количества строк для отображения в таблицах
    @num_to_df_display_.setter
    def num_to_df_display_(self, num: int) -> None:
        """Установка количества строк для отображения в таблицах"""

        try:
            # Проверка аргументов
            if type(num) is not int or num < 1 or num > 50:
                raise TypeError
        except TypeError:
            if self.__num_to_df_display_true == 0:
                self.num_to_df_display = NUM_TO_DF_DISPLAY
        else:
            self.num_to_df_display = num
            # Увеличение счетчика изменения количества строк для отображения в таблицах
            self.__num_to_df_display_true += 1

    @property
    def path_to_save_(self) -> str:
        """Получение/установка директории для сохранения данных

        Args:
            (str): Директория для сохранения данных

        Returns:
            str: Директория для сохранения данных

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                print(settings.path_to_save_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                models

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.path_to_save_ = './models/Audio'
                print(settings.path_to_save_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                models/Audio

            :bdg-light:`-- 3 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.path_to_save_ = ''
                print(settings.path_to_save_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                .

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.path_to_save_ = [2, []]
                print(settings.path_to_save_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                models

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 5
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.path_to_save_ = {'a': 1, 'b': 2}
                print(settings.path_to_save_)

            .. output-cell::
                :execution-count: 5
                :linenos:

                models
        """

        return self._path_to_save

    @path_to_save_.setter
    def path_to_save_(self, path: str) -> None:
        """Установка директории для сохранения данных"""

        if type(path) is str:
            self._path_to_save = os.path.normpath(path)

    @property
    def path_to_logs_(self) -> str:
        """Получение/установка директории для сохранения LOG файлов

        Args:
            (str): Директория для сохранения LOG файлов

        Returns:
            str: Директория для сохранения LOG файлов

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                print(settings.path_to_logs_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                logs

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.path_to_logs_ = './logs/DF'
                print(settings.path_to_logs_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                logs/DF

            :bdg-light:`-- 3 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.path_to_logs_ = ''
                print(settings.path_to_logs_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                .

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.path_to_logs_ = [2, []]
                print(settings.path_to_logs_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                logs

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 5
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.path_to_logs_ = {'a': 1, 'b': 2}
                print(settings.path_to_logs_)

            .. output-cell::
                :execution-count: 5
                :linenos:

                logs
        """

        return self._path_to_logs

    @path_to_logs_.setter
    def path_to_logs_(self, path: str) -> None:
        """Установка директории для сохранения LOG файлов"""

        if type(path) is str:
            self._path_to_logs = os.path.normpath(path)

    @property
    def chunk_size_(self) -> int:
        """Получение/установка размера загрузки файла из сети за 1 шаг

        Args:
            (int): Размер загрузки файла из сети за 1 шаг

        Returns:
            int: Размер загрузки файла из сети за 1 шаг

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                print(settings.chunk_size_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                1000000

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.chunk_size_ = 2000000
                print(settings.chunk_size_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                2000000

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.chunk_size_ = -1
                print(settings.chunk_size_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                1000000

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.chunk_size_ = False
                print(settings.chunk_size_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                1000000

            :bdg-light:`-- 3 --`

            .. code-cell:: python
                :execution-count: 5
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.chunk_size_ = 'какой-то_текст'
                print(settings.chunk_size_)

            .. output-cell::
                :execution-count: 5
                :linenos:

                1000000
        """

        return self._chunk_size

    @chunk_size_.setter
    def chunk_size_(self, size: int) -> None:
        """Установка директории для сохранения данных"""

        if type(size) is int and size > 0:
            self._chunk_size = size

    @property
    def path_to_dataset_(self) -> str:
        """Получение/установка директории набора данных

        Args:
            (str): Директория набора данных

        Returns:
            str: Директория набора данных

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                print(settings.path_to_dataset_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                .

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.path_to_dataset_ = './dataset'
                print(settings.path_to_dataset_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                dataset

            :bdg-light:`-- 3 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.path_to_dataset_ = ''
                print(settings.path_to_dataset_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                .

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.path_to_dataset_ = [2, []]
                print(settings.path_to_dataset_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                .

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 5
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.path_to_dataset_ = 1
                print(settings.path_to_dataset_)

            .. output-cell::
                :execution-count: 5
                :linenos:

                .
        """

        return self._path_to_dataset

    @path_to_dataset_.setter
    def path_to_dataset_(self, path: str) -> None:
        """Установка директории набора данных"""

        if type(path) is str:
            self._path_to_dataset = os.path.normpath(path)

    @property
    def keys_dataset_(self):
        """Получение/установка названий ключей набора данных

        Args:
            (List[str]): Список с названиями ключей набора данных

        Returns:
            List[str]: Список с названиями ключей набора данных

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                print(settings.keys_dataset_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                [
                    'Path',
                    'Openness',
                    'Conscientiousness',
                    'Extraversion',
                    'Agreeableness',
                    'Non-Neuroticism'
                ]

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.keys_dataset_ = ['P', 'O', 'C', 'E', 'A', 'N']
                print(settings.keys_dataset_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                ['P', 'O', 'C', 'E', 'A', 'N']

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.keys_dataset_ = [{}, [], 1]
                print(settings.keys_dataset_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                [
                    'Path',
                    'Openness',
                    'Conscientiousness',
                    'Extraversion',
                    'Agreeableness',
                    'Non-Neuroticism'
                ]

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.keys_dataset_ = ['P', 'O']
                print(settings.keys_dataset_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                [
                    'Path',
                    'Openness',
                    'Conscientiousness',
                    'Extraversion',
                    'Agreeableness',
                    'Non-Neuroticism'
                ]

            :bdg-light:`-- 3 --`

            .. code-cell:: python
                :execution-count: 5
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.keys_dataset_ = []
                print(settings.keys_dataset_)

            .. output-cell::
                :execution-count: 5
                :linenos:

                [
                    'Path',
                    'Openness',
                    'Conscientiousness',
                    'Extraversion',
                    'Agreeableness',
                    'Non-Neuroticism'
                ]
        """

        return self._keys_dataset

    # Установка названий ключей набора данных
    @keys_dataset_.setter
    def keys_dataset_(self, keys: List[str]) -> None:
        """Установка названий ключей набора данных"""

        if type(keys) is list and len(keys) == len(KEYS_DATASET):
            try:
                # .capitalize()
                self._keys_dataset = [x for x in keys]
            except Exception:
                pass

        if type(keys) is list and len(keys) == len(KEYS_DATASET) - 1:
            try:
                for x in keys:
                    if type(x) is not str or not x:
                        raise TypeError
                # .capitalize()
                self._keys_dataset[1:] = [x for x in keys]
            except Exception:
                pass

    @property
    def ignore_dirs_(self) -> List[str]:
        """Получение/установка списка с директориями не входящими в выборку

        Args:
            (List[str]): Список с директориями

        Returns:
            List[str]: Список с директориями

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                print(settings.ignore_dirs_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                []

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.ignore_dirs_ = ['test', 'test_2']
                print(settings.ignore_dirs_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                ['test', 'test_2']

            :bdg-light:`-- 3 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.ignore_dirs_ = []
                print(settings.ignore_dirs_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                []

            :bdg-light:`-- 4 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.ext_ = ['1_a', '2_b']
                print(settings.ext_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                ['1_a', '2_b']

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 5
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.ignore_dirs_ = [2, []]
                print(settings.ignore_dirs_)

            .. output-cell::
                :execution-count: 5
                :linenos:

                []
        """

        return self._ignore_dirs

    @ignore_dirs_.setter
    def ignore_dirs_(self, l: List[str]) -> None:
        """Установка списка с директориями не входящими в выборку"""

        if type(l) is list:
            try:
                self._ignore_dirs = [x.lower() for x in l]
            except Exception:
                pass

    @property
    def ext_(self) -> List[str]:
        """Получение/установка расширений искомых файлов

        Args:
            (List[str]): Список с расширениями искомых файлов

        Returns:
            List[str]: Список с расширениями искомых файлов

        .. dropdown:: Примеры

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                print(settings.ext_)

            .. output-cell::
                :execution-count: 1
                :linenos:

                []

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.ext_ = ['.mp4']
                print(settings.ext_)

            .. output-cell::
                :execution-count: 2
                :linenos:

                ['.mp4']

            :bdg-light:`-- 3 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.ext_ = ['.mp3', '.wav']
                print(settings.ext_)

            .. output-cell::
                :execution-count: 3
                :linenos:

                ['.mp3', '.wav']

            :bdg-light:`-- 4 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.ext_ = []
                print(settings.ext_)

            .. output-cell::
                :execution-count: 4
                :linenos:

                []

            :bdg-warning:`Лучше так не делать` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 5
                :linenos:

                from oceanai.modules.core.settings import Settings

                settings = Settings()
                settings.ext_ = [2, []]
                print(settings.ext_)

            .. output-cell::
                :execution-count: 5
                :linenos:

                []
        """

        return self._ext

    @ext_.setter
    def ext_(self, ext: List[str]) -> None:
        """Установка расширений искомых файлов"""

        if type(ext) is list:
            try:
                self._ext = [x.lower() for x in ext]
            except Exception:
                pass
