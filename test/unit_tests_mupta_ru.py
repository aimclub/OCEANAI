#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from oceanai.modules.lab.build import Run

# The test videos were taken from the MuPTA corpus
# Please click the link https://hci.nw.ru/ru/pages/mupta-corpus
# for access to all videos from the corpus

PATH_TO_DIR = os.path.normpath("./video_MuPTA_2/")
PATH_SAVE_VIDEO = os.path.normpath("./video_MuPTA_2/test/")
PATH_SAVE_MODELS = os.path.normpath("./models")

CHUNK_SIZE = 2000000
FILENAME_1 = "speaker_01_center_83.mov"
FILENAME_2 = "speaker_10_center_83.mov"

corpus = "mupta"
lang = "ru"

_b5 = Run()

_b5.path_to_save_ = PATH_SAVE_VIDEO
_b5.chunk_size_ = CHUNK_SIZE

_b5.path_to_dataset_ = PATH_TO_DIR
_b5.ignore_dirs_ = []
_b5.keys_dataset_ = ["Path", "Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Non-Neuroticism"]
_b5.ext_ = [".mov"]
_b5.path_to_logs_ = "./logs"

URL_ACCURACY = _b5.true_traits_[corpus]["sberdisk"]

_b5.download_file_from_url(
    url="https://download.sberdisk.ru/download/file/477995980?token=jGPtBPS69uzFU6Y&filename=" + FILENAME_1, out=False
)
_b5.download_file_from_url(
    url="https://download.sberdisk.ru/download/file/477995963?token=bTpo96qNDPcwGqb&filename=" + FILENAME_2, out=False
)


def test_get_acoustic_features():
    hc_features, melspectrogram_features = _b5.get_acoustic_features(
        path=os.path.join(PATH_SAVE_VIDEO, FILENAME_1), out=False
    )

    assert np.asarray(hc_features).shape[1] == 196
    assert np.asarray(hc_features).shape[2] == 25
    assert np.asarray(melspectrogram_features).shape[1] == 224
    assert np.asarray(melspectrogram_features).shape[2] == 224


def test_get_visual_features():
    _b5.path_to_save_ = PATH_SAVE_MODELS

    _b5.load_video_model_deep_fe(out=False)
    url = _b5.weights_for_big5_["video"][corpus]["fe"]["sberdisk"]
    _b5.load_video_model_weights_deep_fe(url=url, out=False)

    hc_features, nn_features = _b5.get_visual_features(
        path=os.path.join(PATH_SAVE_VIDEO, FILENAME_1), lang=lang, out=False
    )

    assert hc_features.shape[1] == 10
    assert hc_features.shape[2] == 109
    assert nn_features.shape[2] == 512


def test_get_text_features():
    _b5.load_text_features(out=False)
    _b5.setup_translation_model(out=False)
    _b5.setup_bert_encoder(out=False, force_reload=False)

    _b5.path_to_save_ = PATH_SAVE_MODELS

    hc_features, nn_features = _b5.get_text_features(
        path=os.path.join(PATH_SAVE_VIDEO, FILENAME_1), asr=True, out=False, lang=lang
    )

    assert hc_features.shape[0] == 365
    assert nn_features.shape[0] == 414


def test_get_text_union_predictions():
    _b5.load_text_features(out=False)
    _b5.setup_translation_model(out=False)
    _b5.setup_bert_encoder(force_reload=False, out=False)

    _b5.load_text_model_hc(corpus=corpus, out=False)
    url = _b5.weights_for_big5_["text"][corpus]["hc"]["sberdisk"]
    _b5.load_text_model_weights_hc(url=url, out=False)

    _b5.load_text_model_nn(corpus=corpus, out=False)
    url = _b5.weights_for_big5_["text"][corpus]["nn"]["sberdisk"]
    _b5.load_text_model_weights_nn(url=url, out=False)

    _b5.load_text_model_b5(out=False)
    url = _b5.weights_for_big5_["text"][corpus]["b5"]["sberdisk"]
    _b5.load_text_model_weights_b5(url=url, out=False)

    _b5.get_text_union_predictions(lang=lang, url_accuracy=URL_ACCURACY, out=False)

    assert _b5.df_accuracy_["Mean"].values[0] <= 0.2
    assert _b5.df_accuracy_["Mean"].values[1] >= 0.8


def test_get_avt_predictions():
    _b5.path_to_save_ = PATH_SAVE_MODELS

    _b5.load_audio_model_hc(out=False)
    _b5.load_audio_model_nn(out=False)

    url = _b5.weights_for_big5_["audio"][corpus]["hc"]["sberdisk"]
    _b5.load_audio_model_weights_hc(url=url, out=False)

    url = _b5.weights_for_big5_["audio"][corpus]["nn"]["sberdisk"]
    _b5.load_audio_model_weights_nn(url=url, out=False)

    _b5.load_video_model_hc(lang=lang, out=False)
    _b5.load_video_model_deep_fe(out=False)
    _b5.load_video_model_nn(out=False)

    url = _b5.weights_for_big5_["video"][corpus]["hc"]["sberdisk"]
    _b5.load_video_model_weights_hc(url=url, out=False)

    url = _b5.weights_for_big5_["video"][corpus]["fe"]["sberdisk"]
    _b5.load_video_model_weights_deep_fe(url=url, out=False)

    url = _b5.weights_for_big5_["video"][corpus]["nn"]["sberdisk"]
    _b5.load_video_model_weights_nn(url=url, out=False)

    _b5.load_text_features(out=False)
    _b5.setup_translation_model(out=False)
    _b5.setup_bert_encoder(force_reload=False, out=False)

    _b5.load_text_model_hc(corpus=corpus, out=False)
    url = _b5.weights_for_big5_["text"][corpus]["hc"]["sberdisk"]
    _b5.load_text_model_weights_hc(url=url, out=False)

    _b5.load_text_model_nn(corpus=corpus, out=False)
    url = _b5.weights_for_big5_["text"][corpus]["nn"]["sberdisk"]
    _b5.load_text_model_weights_nn(url=url, out=False)

    _b5.load_avt_model_b5(out=False)
    url = _b5.weights_for_big5_["avt"][corpus]["b5"]["sberdisk"]
    _b5.load_avt_model_weights_b5(url=url, out=False)

    _b5.get_avt_predictions(url_accuracy=URL_ACCURACY, lang=lang, out=False)

    assert _b5.df_accuracy_["Mean"].values[0] <= 0.2
    assert _b5.df_accuracy_["Mean"].values[1] >= 0.8
