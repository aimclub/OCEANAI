"""
File: oceanai_init.py
Author: Elena Ryumina and Dmitry Ryumin
Description: OceanAI initialization.
License: MIT License
"""

from oceanai.modules.lab.build import Run


def oceanai_initialization():
    out = False

    # Создание экземпляра класса
    _b5 = Run(lang="en", metadata=out)

    # Настройка ядра
    _b5.path_to_save_ = "./models"  # Директория для сохранения файла
    _b5.chunk_size_ = 2000000  # Размер загрузки файла из сети за 1 шаг

    corpus = "fi"
    disk = "googledisk"

    # Формирование аудиомоделей
    _ = _b5.load_audio_model_hc(out=out)
    _ = _b5.load_audio_model_nn(out=out)

    # Загрузка весов аудиомоделей
    url = _b5.weights_for_big5_["audio"][corpus]["hc"][disk]
    _ = _b5.load_audio_model_weights_hc(url=url, out=out)

    url = _b5.weights_for_big5_["audio"][corpus]["nn"][disk]
    _ = _b5.load_audio_model_weights_nn(url=url, out=out)

    # Формирование видеомоделей
    _ = _b5.load_video_model_hc(lang="en", out=out)
    _ = _b5.load_video_model_deep_fe(out=out)
    _ = _b5.load_video_model_nn(out=out)

    # Загрузка весов видеомоделей
    url = _b5.weights_for_big5_["video"][corpus]["hc"][disk]
    _ = _b5.load_video_model_weights_hc(url=url, out=out)

    url = _b5.weights_for_big5_["video"][corpus]["fe"][disk]
    _ = _b5.load_video_model_weights_deep_fe(url=url, out=out)

    url = _b5.weights_for_big5_["video"][corpus]["nn"][disk]
    _ = _b5.load_video_model_weights_nn(url=url, out=out)

    # Загрузка словаря с экспертными признаками (текстовая модальность)
    _ = _b5.load_text_features(out=out)

    # Формирование текстовых моделей
    _ = _b5.setup_translation_model()  # только для русского языка
    _ = _b5.setup_bert_encoder(force_reload=False, out=out)
    _ = _b5.load_text_model_hc(corpus=corpus, out=out)
    _ = _b5.load_text_model_nn(corpus=corpus, out=out)

    # Загрузка весов текстовых моделей
    url = _b5.weights_for_big5_["text"][corpus]["hc"][disk]
    _ = _b5.load_text_model_weights_hc(url=url, out=out)

    url = _b5.weights_for_big5_["text"][corpus]["nn"][disk]
    _ = _b5.load_text_model_weights_nn(url=url, out=out)

    # Формирование модели для мультимодального объединения информации
    _ = _b5.load_avt_model_b5(out=out)

    # Загрузка весов модели для мультимодального объединения информации
    url = _b5.weights_for_big5_["avt"][corpus]["b5"][disk]
    _ = _b5.load_avt_model_weights_b5(url=url, out=out)

    return _b5


b5 = oceanai_initialization()
