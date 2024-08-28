#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Загрузка файлов
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
import numpy as np  # Научные вычисления
import requests  # Отправка HTTP запросов
import re  # Регулярные выражения
import shutil  # Набор функций высокого уровня для обработки файлов, групп файлов, и папок

from pathlib import Path  # Работа с путями в файловой системе

from IPython.display import clear_output

# Персональные
from oceanai.modules.lab.unzip import Unzip  # Обработка архивов
from oceanai.modules.core.exceptions import InvalidContentLength


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class DownloadMessages(Unzip):
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

        self._could_not_process_url = self._oh + self._("не удалось обработать указанный URL ...")
        self._url_incorrect = self._oh + self._("URL указан некорректно ...")
        self._url_incorrect_content_length = self._oh + self._("Не определен размер файла для загрузки ...")
        self._automatic_download: str = self._('Загрузка файла "{}"')
        self._url_error_code_http: str = self._(" (ошибка {})")
        self._url_error_http: str = self._oh + self._('не удалось скачать файл "{}"{} ...')


# ######################################################################################################################
# Загрузка файлов
# ######################################################################################################################
class Download(DownloadMessages):
    """Класс для загрузки файлов

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

        self._headers: str = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/89.0.4389.90 Safari/537.36"
        )  # User-Agent

        self._url_last_filename: str = ""  # Имя последнего загруженного файла

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)
    # ------------------------------------------------------------------------------------------------------------------

    def __progressbar_download_file_from_url(
        self, url_filename: str, progress: float, clear_out: bool = True, last: bool = False, out: bool = True
    ) -> None:
        """Индикатор выполнения загрузки файла из URL

        .. note::
            private (приватный метод)

        Args:
            url_filename (str): Путь до файла
            progress (float): Процент выполнения (от **0.0** до **100.0**)
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

                import numpy as np
                from oceanai.modules.lab.download import Download

                download = Download()

                for progress in np.arange(0., 101, 25):
                    download._Download__progressbar_download_file_from_url(
                        url_filename = 'https://clck.ru/32Nwdk',
                        progress = float(progress),
                        clear_out = False,
                        last = False, out = True
                    )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-16 16:58:51] Загрузка файла "https://clck.ru/32Nwdk" (0.0%) ...

                [2022-10-16 16:58:51] Загрузка файла "https://clck.ru/32Nwdk" (25.0%) ...

                [2022-10-16 16:58:51] Загрузка файла "https://clck.ru/32Nwdk" (50.0%) ...

                [2022-10-16 16:58:51] Загрузка файла "https://clck.ru/32Nwdk" (75.0%) ...

                [2022-10-16 16:58:51] Загрузка файла "https://clck.ru/32Nwdk" (100.0%) ...

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                import numpy as np
                from oceanai.modules.lab.download import Download

                download = Download()

                for progress in np.arange(0., 101, 25):
                    download._Download__progressbar_download_file_from_url(
                        url_filename = 'https://clck.ru/32Nwdk',
                        progress = float(progress),
                        clear_out = True,
                        last = True, out = True
                    )

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-16 16:59:41] Загрузка файла "https://clck.ru/32Nwdk" (100.0%) ...

            :bdg-danger:`Ошибка` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                import numpy as np
                from oceanai.modules.lab.download import Download

                download = Download()

                for progress in np.arange(0., 101, 25):
                    download._Download__progressbar_download_file_from_url(
                        url_filename = 'https://clck.ru/32Nwdk',
                        progress = 101,
                        clear_out = True,
                        last = False, out = True
                    )

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-16 17:00:11] Неверные типы или значения аргументов в "Download.__progressbar_download_file_from_url" ...
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
                type(url_filename) is not str
                or not url_filename
                or type(progress) is not float
                or not (0 <= progress <= 100)
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__progressbar_download_file_from_url.__name__, out=out)
            return None

        self._info(
            self._automatic_download.format(self._info_wrapper(url_filename)) + self._download_precent.format(progress),
            last=last,
            out=False,
        )
        if out:
            self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    def _download_file_from_url(
        self, url: str, force_reload: bool = True, out: bool = True, runtime: bool = True, run: bool = True
    ) -> int:
        """Загрузка файла из URL (без очистки истории вывода сообщений в ячейке Jupyter)

        .. note::
            protected (защищенный метод)

        Args:
            url (str): Полный путь к файлу
            force_reload (bool): Принудительная загрузка файла из сети
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            int: Код статуса ответа:

                * ``200`` - Файл загружен
                * ``400`` - Ошибка при проверке аргументов
                * ``403`` - Выполнение заблокировано пользователем
                * ``404`` - Не удалось скачать файл

        .. dropdown:: Примеры
            :class-body: sd-pr-5

            :bdg-success:`Верно` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 1
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.download import Download

                download = Download()

                download.path_to_save_ = './models'
                download.chunk_size_ = 2000000

                res_download_file_from_url = download._download_file_from_url(
                    url = 'https://download.sberdisk.ru/download/file/400635799?token=MMRrak8fMsyzxLE&filename=weights_2022-05-05_11-27-55.h5',
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )

            .. output-cell::
                :execution-count: 1
                :linenos:

                [2022-10-16 20:23:25] Загрузка файла "weights_2022-05-05_11-27-55.h5" (100.0%) ...

                --- Время выполнения: 0.373 сек. ---

                200

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 2
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.download import Download

                download = Download()

                download.path_to_save_ = './models'
                download.chunk_size_ = 2000000

                res_download_file_from_url = download._download_file_from_url(
                    url = 'https://clck.ru/32Nwdk',
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = False
                )
                res_download_file_from_url

            .. output-cell::
                :execution-count: 2
                :linenos:

                [2022-10-16 19:33:05] Выполнение заблокировано пользователем ...

                403

            :bdg-danger:`Ошибки` :bdg-light:`-- 1 --`

            .. code-cell:: python
                :execution-count: 3
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.download import Download

                download = Download()

                download.path_to_save_ = './models'
                download.chunk_size_ = 2000000

                res_download_file_from_url = download._download_file_from_url(
                    url = 1,
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )
                res_download_file_from_url

            .. output-cell::
                :execution-count: 3
                :linenos:

                [2022-10-16 19:33:01] Неверные типы или значения аргументов в "Download._download_file_from_url" ...

                400

            :bdg-light:`-- 2 --`

            .. code-cell:: python
                :execution-count: 4
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.download import Download

                download = Download()

                download.path_to_save_ = './models'
                download.chunk_size_ = 2000000

                res_download_file_from_url = download._download_file_from_url(
                    url = 'https://',
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )
                res_download_file_from_url

            .. output-cell::
                :execution-count: 4
                :linenos:

                [2022-10-16 19:33:10] Что-то пошло не так ... не удалось обработать указанный URL ...

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/download.py
                    Линия: 257
                    Метод: _download_file_from_url
                    Тип ошибки: InvalidURL

                --- Время выполнения: 0.061 сек. ---

                404

            :bdg-light:`-- 3 --`

            .. code-cell:: python
                :execution-count: 5
                :linenos:
                :tab-width: 8

                from oceanai.modules.lab.download import Download

                download = Download()

                download.path_to_save_ = './models'
                download.chunk_size_ = 2000000

                res_download_file_from_url = download._download_file_from_url(
                    url = 'https://www.iconfinder.com/icons/4375050/download/svg/4096',
                    force_reload = True,
                    out = True,
                    runtime = True,
                    run = True
                )
                res_download_file_from_url

            .. output-cell::
                :execution-count: 5
                :linenos:

                [2022-10-16 19:33:15] Загрузка файла "4375050_logo_python_icon.svg"

                [2022-10-16 19:33:15] Что-то пошло не так ... Не определен размер файла для загрузки ...

                    Файл: /Users/dl/GitHub/oceanai/oceanai/modules/lab/download.py
                    Линия: 324
                    Метод: _download_file_from_url
                    Тип ошибки: InvalidContentLength

                --- Время выполнения: 0.386 сек. ---

                404
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
            self._inv_args(__class__.__name__, self._download_file_from_url.__name__, out=out)
            return 400
        else:
            # Блокировка выполнения
            if run is False:
                self._error(self._lock_user, out=out)
                return 403

            if runtime:
                self._r_start()

            try:
                # Отправка GET запроса для получения файла
                r = requests.get(url, headers={"user-agent": self._headers}, stream=True)
            except (
                # https://requests.readthedocs.io/en/master/_modules/requests/exceptions/
                requests.exceptions.MissingSchema,
                requests.exceptions.InvalidSchema,
                # requests.exceptions.ConnectionError,
                requests.exceptions.InvalidURL,
            ):
                self._other_error(self._could_not_process_url, out=out)
                return 404
            except requests.exceptions.ConnectionError:
                url_filename = url.split("=")[-1]
                local_file = os.path.join(self.path_to_save_, url_filename)
                self._url_last_filename = local_file
                return 200
            
            except Exception:
                self._other_error(self._unknown_err, out=out)
                return 404

            else:
                # Имя файла
                if "Content-Disposition" in r.headers.keys():
                    try:
                        url_filename = re.findall('(?<=[\(\{\["]).+(?=[\)\}\]"])', r.headers["Content-Disposition"])[0]
                    except IndexError:
                        url_filename = re.findall(
                            r'filename\*?=[\'"]?(?:UTF-\d[\'"]*)?([^;\r\n"\']*)[\'"]?;?',
                            r.headers["Content-Disposition"],
                        )[0]
                else:
                    url_filename = url.split("/")[-1]

                try:
                    # URL файл невалидный
                    if not url_filename or not Path(url_filename).suffix:
                        if not Path(url_filename).stem.lower():
                            raise requests.exceptions.InvalidURL

                        if r.headers["Content-Type"] == "image/jpeg":
                            ext = "jpg"
                        elif r.headers["Content-Type"] == "image/png":
                            ext = "png"
                        elif r.headers["Content-Type"] == "text/plain":
                            ext = "txt"
                        elif r.headers["Content-Type"] == "text/csv":
                            ext = "csv"
                        elif r.headers["Content-Type"] == "video/mp4":
                            ext = "mp4"
                        else:
                            raise requests.exceptions.InvalidHeader

                        url_filename = Path(url_filename).stem + "." + ext
                except (requests.exceptions.InvalidURL, requests.exceptions.InvalidHeader):
                    self._other_error(self._url_incorrect, out=out)
                    return 404
                except Exception:
                    self._other_error(self._unknown_err, out=out)
                    return 404
                else:
                    # Информационное сообщение
                    self._info(self._automatic_download.format(self._info_wrapper(url_filename)), out=False)
                    if out:
                        self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

                    # Директория для сохранения файла
                    if not os.path.exists(self.path_to_save_):
                        os.makedirs(self.path_to_save_)

                    local_file = os.path.join(self.path_to_save_, url_filename)  # Путь к файлу

                    try:
                        # Принудительная загрузка файла из сети
                        if force_reload is True:
                            # Файл найден
                            if os.path.isfile(local_file) is True:
                                # Удаление файла
                                try:
                                    shutil.rmtree(local_file)
                                except OSError:
                                    os.remove(local_file)
                                except Exception:
                                    raise Exception
                    except Exception:
                        self._other_error(self._unknown_err, out=out)
                        return 404
                    else:
                        # Файл с указанным именем найден локально и принудительная загрузка файла из сети не указана
                        if Path(local_file).is_file() is True and force_reload is False:
                            self._url_last_filename = local_file
                            return 200
                        else:
                            # Ответ получен
                            if r.status_code == 200:
                                total_length = int(r.headers.get("content-length", 0))  # Длина файла

                                try:
                                    if total_length == 0:
                                        raise InvalidContentLength
                                except InvalidContentLength:
                                    self._other_error(self._url_incorrect_content_length, out=out)
                                    return 404
                                else:
                                    num_bars = int(np.ceil(total_length / self.chunk_size_))  # Количество загрузок

                                    try:
                                        # Открытие файла для записи
                                        with open(local_file, "wb") as f:
                                            # Индикатор выполнения
                                            self.__progressbar_download_file_from_url(
                                                url_filename, 0.0, clear_out=True, last=True, out=out
                                            )

                                            # Сохранение файла по частям
                                            for i, chunk in enumerate(r.iter_content(chunk_size=self.chunk_size_)):
                                                f.write(chunk)  # Запись в файл
                                                f.flush()

                                                # Индикатор выполнения
                                                self.__progressbar_download_file_from_url(
                                                    url_filename,
                                                    round(i * 100 / num_bars, 2),
                                                    clear_out=True,
                                                    last=True,
                                                    out=out,
                                                )

                                            # Индикатор выполнения
                                            self.__progressbar_download_file_from_url(
                                                url_filename, 100.0, clear_out=True, last=True, out=out
                                            )
                                    except Exception:
                                        self._other_error(self._unknown_err, out=out)
                                        return 404
                                    else:
                                        self._url_last_filename = local_file
                                        return 200
                            else:
                                self._error(
                                    self._url_error_http.format(
                                        self._info_wrapper(url_filename),
                                        self._url_error_code_http.format(self._error_wrapper(str(r.status_code))),
                                    ),
                                    out=out,
                                )
                                return 404
            finally:
                if runtime:
                    self._r_end(out=out)

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def download_file_from_url(
        self, url: str, force_reload: bool = True, out: bool = True, runtime: bool = True, run: bool = True
    ) -> int:
        """Загрузка файла из URL

        Args:
            url (str): Полный путь к файлу
            force_reload (bool): Принудительная загрузка файла из сети
            out (bool): Отображение
            runtime (bool): Подсчет времени выполнения
            run (bool): Блокировка выполнения

        Returns:
            int: Код статуса ответа:

                * ``200`` - Файл загружен
                * ``400`` - Ошибка при проверке аргументов
                * ``403`` - Выполнение заблокировано пользователем
                * ``404`` - Не удалось скачать файл

        :bdg-link-light:`Пример <../../user_guide/notebooks/Download-download_file_from_url.ipynb>`
        """

        self._clear_notebook_history_output()  # Очистка истории вывода сообщений в ячейке Jupyter

        return self._download_file_from_url(url=url, force_reload=force_reload, out=out, runtime=runtime, run=run)
