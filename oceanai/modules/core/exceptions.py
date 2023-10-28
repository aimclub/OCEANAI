#!/usr/bin/env python
# -*- coding: utf-8 -*-


class CustomException(Exception):
    """Класс для всех пользовательских исключений

    .. dropdown:: Пример

        :bdg-success:`Верно` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 1
            :linenos:

            from oceanai.modules.core.exceptions import CustomException

            message = 'Пользовательское исключение'

            try: raise CustomException(message)
            except CustomException as ex: print(ex)

        .. output-cell::
            :execution-count: 1
            :linenos:

            Пользовательское исключение
    """

    pass


class IsSmallWindowSizeError(CustomException):
    """Указан слишком маленький размер окна сегмента сигнала

    .. dropdown:: Пример

        :bdg-success:`Верно` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 1
            :linenos:

            from oceanai.modules.core.exceptions import IsSmallWindowSizeError

            message = 'Указан слишком маленький размер окна сегмента сигнала'

            try: raise IsSmallWindowSizeError(message)
            except IsSmallWindowSizeError as ex: print(ex)

        .. output-cell::
            :execution-count: 1
            :linenos:

            Указан слишком маленький размер окна сегмента сигнала

    """

    pass


class InvalidContentLength(CustomException):
    """Не определен размер файла для загрузки

    .. dropdown:: Пример

        :bdg-success:`верно` :bdg-light:`-- 1 --`

        .. code-cell:: python
            :execution-count: 1
            :linenos:

            from oceanai.modules.core.exceptions import InvalidContentLength

            message = 'Не определен размер файла для загрузки'

            try: raise InvalidContentLength(message)
            except InvalidContentLength as ex: print(ex)

        .. output-cell::
            :execution-count: 1
            :linenos:

            Не определен размер файла для загрузки

    """

    pass
