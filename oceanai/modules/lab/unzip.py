#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Обработка архивов
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings

for warn in [UserWarning, FutureWarning]:
    warnings.filterwarnings("ignore", category=warn)

from dataclasses import dataclass  # Класс данных

import os  # Взаимодействие с файловой системой
from zipfile import ZipFile, BadZipFile  # Работа с ZIP архивами
from pathlib import Path  # Работа с путями в файловой системе
import shutil  # Набор функций высокого уровня для обработки файлов, групп файлов, и папок

from typing import List, Optional  # Типы данных

from IPython.display import clear_output

# Персональные
from oceanai.modules.core.core import Core  # Ядро

# ######################################################################################################################
# Константы
# ######################################################################################################################
EXTS_ZIP: List[str] = ["zip"]  # Поддерживаемые расширения архивов


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class UnzipMessages(Core):
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

        self._automatic_unzip: str = self._('Разархивирование архива "{}" ...')
        self._download_precent: str = " {}% ..."
        self._automatic_unzip_progress: str = self._automatic_unzip + " {}% ..."
        self._error_unzip: str = self._oh + self._('не удалось разархивировать архив "{}" ...')
        self._error_rename: str = self._oh + self._('не удалось переименовать директорию из "{}" в "{}" ...')


# ######################################################################################################################
# Обработка архивов
# ######################################################################################################################
class Unzip(UnzipMessages):
    """Класс для обработки архивов

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

        self._path_to_unzip: str = ""  # Имя директории для разархивирования

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def path_to_unzip(self) -> str:
        """Получение директории для разархивирования

        Returns:
            str: Директория для разархивирования
        """

        return self._path_to_unzip

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------------------------------------------------------

    # Индикатор выполнения
    def __progressbar_unzip(
        self, path_to_zipfile: str, progress: float, clear_out: bool = True, last: bool = False, out: bool = True
    ) -> None:
        """Индикатор выполнения

        .. note::
            private (приватный метод)

        Args:
            path_to_zipfile (str): Путь до архива
            progress (float): Процент выполнения (от **0.0** до **100.0**)
            clear_out (bool): Очистка области вывода
            last (bool): Замена последнего сообщения
            out (bool): Отображение
        """

        if clear_out is False and last is True:
            clear_out, last = last, clear_out
        elif clear_out is False and last is False:
            clear_out = True

        if clear_out is True:
            clear_output(True)

        try:
            # Проверка аргументов
            if (
                type(path_to_zipfile) is not str
                or not path_to_zipfile
                or type(progress) is not float
                or not (0 <= progress <= 100)
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__progressbar_unzip.__name__, out=out)
            return None

        self._info(
            self._automatic_unzip.format(self._info_wrapper(path_to_zipfile)) + self._download_precent.format(progress),
            last=last,
            out=False,
        )
        if out:
            self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    def _unzip(
        self, path_to_zipfile: str, new_name: Optional[str] = None, force_reload: bool = True, out: bool = True
    ) -> bool:
        """Разархивирование архива (без очистки истории вывода сообщений в ячейке Jupyter)

        .. note::
            protected (защищенный метод)

        Args:
            path_to_zipfile (str): Полный путь до архива
            new_name (str): Имя директории для разархивирования
            force_reload (bool): Принудительное разархивирование
            out (bool): Отображение

        Returns:
            bool: **True** если разархивирование прошло успешно, в обратном случае **False**
        """

        try:
            if new_name is None:
                new_name = path_to_zipfile  # Имя директории для разархивирования не задана

            # Проверка аргументов
            if (
                type(path_to_zipfile) is not str
                or not path_to_zipfile
                or type(new_name) is not str
                or not new_name
                or type(force_reload) is not bool
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.unzip.__name__, out=out)
            return False
        else:
            # Нормализация путей
            path_to_zipfile = os.path.normpath(path_to_zipfile)
            new_name = os.path.normpath(new_name)

            # Информационное сообщение
            self._info(self._automatic_unzip.format(self._info_wrapper(Path(path_to_zipfile).name)), out=False)
            if out:
                self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

            # Имя директории для разархивирования
            if path_to_zipfile == new_name:
                self._path_to_unzip = str(Path(path_to_zipfile).with_suffix(""))
            else:
                self._path_to_unzip = os.path.join(self.path_to_save_, Path(new_name).name)

            try:
                # Расширение файла неверное
                if (Path(path_to_zipfile).suffix.replace(".", "") in EXTS_ZIP) is False:
                    raise TypeError
            except TypeError:
                self._error(
                    self._wrong_extension.format(self._info_wrapper(", ".join(x for x in EXTS_ZIP))),
                    out=out,
                )
                return False
            else:
                # Принудительное разархивирование отключено
                if force_reload is False:
                    # Каталог уже существует
                    if os.path.isdir(self._path_to_unzip):
                        return True
                try:
                    # Файл не найден
                    if os.path.isfile(path_to_zipfile) is False:
                        raise FileNotFoundError
                except FileNotFoundError:
                    self._error(
                        self._file_not_found.format(self._info_wrapper(Path(path_to_zipfile).name)),
                        out=out,
                    )
                    return False
                except Exception:
                    self._other_error(self._unknown_err, out=out)
                    return False
                else:
                    extracted_size = 0  # Объем извлеченной информации

                    try:
                        # Процесс разархивирования
                        with ZipFile(path_to_zipfile, "r") as zf:
                            # Индикатор выполнения
                            self.__progressbar_unzip(
                                Path(path_to_zipfile).name, 0.0, clear_out=True, last=True, out=out
                            )

                            uncompress_size = sum((file.file_size for file in zf.infolist()))  # Общий размер

                            # Проход по всем файлам, которые необходимо разархивировать
                            for file in zf.infolist():
                                extracted_size += file.file_size  # Увеличение общего объема
                                zf.extract(file, self.path_to_save_)  # Извлечение файла из архива

                                # Индикатор выполнения
                                self.__progressbar_unzip(
                                    Path(path_to_zipfile).name,
                                    round(extracted_size * 100 / uncompress_size, 2),
                                    clear_out=True,
                                    last=True,
                                    out=out,
                                )

                            # Индикатор выполнения
                            self.__progressbar_unzip(
                                Path(path_to_zipfile).name, 100.0, clear_out=True, last=True, out=out
                            )
                    except BadZipFile:
                        self._error(self._error_unzip.format(self._info_wrapper(Path(path_to_zipfile).name)), out=out)
                        return False
                    except Exception:
                        self._other_error(self._unknown_err, out=out)
                        return False
                    else:
                        # Переименовывать директорию не нужно
                        if path_to_zipfile == new_name:
                            return True

                        try:
                            # Принудительное разархивирование включено и каталог уже существует
                            if force_reload is True and os.path.isdir(self._path_to_unzip):
                                # Удаление директории
                                try:
                                    shutil.rmtree(self._path_to_unzip)
                                except OSError:
                                    os.remove(self._path_to_unzip)
                                except Exception:
                                    raise Exception
                        except Exception:
                            self._other_error(self._unknown_err, out=out)
                            return False
                        else:
                            try:
                                # Переименование
                                os.rename(Path(path_to_zipfile).with_suffix(""), self._path_to_unzip)
                            except Exception:
                                self._error(
                                    self._error_rename.format(
                                        self._info_wrapper(Path(path_to_zipfile).with_suffix("")),
                                        self._info_wrapper(Path(new_name).name),
                                    ),
                                    out=out,
                                )
                                return False
                            else:
                                return True

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def unzip(
        self, path_to_zipfile: str, new_name: Optional[str] = None, force_reload: bool = True, out: bool = True
    ) -> bool:
        """Разархивирование архива

        Args:
            path_to_zipfile (str): Полный путь до архива
            new_name (str): Имя директории для разархивирования
            force_reload (bool): Принудительное разархивирование
            out (bool): Отображение

        Returns:
            bool: **True** если разархивирование прошло успешно, в обратном случае **False**
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        return self._unzip(path_to_zipfile=path_to_zipfile, new_name=new_name, force_reload=force_reload, out=out)
