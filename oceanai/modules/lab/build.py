#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Сборка
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################

import warnings

# Подавление Warning
for warn in [UserWarning, FutureWarning]:
    warnings.filterwarnings("ignore", category=warn)

from dataclasses import dataclass  # Класс данных

# Персональные
from oceanai.modules.lab.prediction import Prediction  # Объединение аудио и видео


# ######################################################################################################################
# Сборка
# ######################################################################################################################
@dataclass
class Run(Prediction):
    """Класс для сборки

    Args:
        lang (str): Смотреть :attr:`~oceanai.modules.core.language.Language.lang`
        color_simple (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.color_simple`
        color_info (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.color_info`
        color_err (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.color_err`
        color_true (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.color_true`
        bold_text (bool): Смотреть :attr:`~oceanai.modules.core.settings.Settings.bold_text`
        num_to_df_display (int): Смотреть :attr:`~oceanai.modules.core.settings.Settings.num_to_df_display`
        text_runtime (str): Смотреть :attr:`~oceanai.modules.core.settings.Settings.text_runtime`
        metadata (bool): Отображение информации о библиотеке
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    metadata: bool = True  # Информация об библиотеке
    """
    bool: Отображение информации о библиотеке
    """

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        if self.is_notebook_ is True and type(self.metadata) is bool and self.metadata is True:
            self._metadata_info()
