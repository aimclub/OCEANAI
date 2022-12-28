#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from oceanai.modules.lab.build import Run

# The test videos were taken from the First Impression v2 corpus. 
# Please click the link https://chalearnlap.cvc.uab.cat/dataset/24/description/
# for access to all videos from the corpus.

PATH_TO_DIR = os.path.normpath('./video_FI/')
PATH_SAVE_VIDEO = os.path.normpath('./video_FI/test/')
PATH_SAVE_MODELS = os.path.normpath('./models')

CHUNK_SIZE = 2000000
FILENAME_1 = 'glgfB3vFewc.004.mp4'
FILENAME_2 = '6V807Mf_gHM.003.mp4'

_b5 = Run()

_b5.path_to_save_ = PATH_SAVE_VIDEO
_b5.chunk_size_ = CHUNK_SIZE

_b5.path_to_dataset_ = PATH_TO_DIR
_b5.ignore_dirs_ = []
_b5.keys_dataset_ = ['Path', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
_b5.ext_ = ['.mp4']
_b5.path_to_logs_ = './logs'

URL_ACCURACY = _b5.true_traits_['sberdisk']

_b5.download_file_from_url(
    url = 'https://download.sberdisk.ru/download/file/425715491?token=EyJnTnsfmJIlsbO&filename=' + FILENAME_1,
    out = False
)
_b5.download_file_from_url(
    url = 'https://download.sberdisk.ru/download/file/425715497?token=00tfwHAaMeMRZMi&filename=' + FILENAME_2,
    out = False
)

def test_get_acoustic_features():
    hc_features, melspectrogram_features = _b5.get_acoustic_features(
        path = os.path.join(PATH_SAVE_VIDEO, FILENAME_1),
        out = False
    )

    assert np.asarray(hc_features).shape[1] == 196
    assert np.asarray(hc_features).shape[2] == 25
    assert np.asarray(melspectrogram_features).shape[1] == 224
    assert np.asarray(melspectrogram_features).shape[2] == 224

def test_get_visual_features():
    _b5.path_to_save_ = PATH_SAVE_MODELS

    _b5.load_video_model_deep_fe(out = False)
    url = _b5.weights_for_big5_['video']['fe']['sberdisk']
    _b5.load_video_model_weights_deep_fe(url = url, out = False)

    hc_features, nn_features = _b5.get_visual_features(
        path = os.path.join(PATH_SAVE_VIDEO, FILENAME_1),
        out = False
    )

    assert hc_features.shape[1] == 10
    assert hc_features.shape[2] == 115
    assert nn_features.shape[2] == 512

def test_get_audio_union_predictions():
    _b5.path_to_save_ = PATH_SAVE_MODELS

    _b5.load_audio_model_hc(out = True)
    url = _b5.weights_for_big5_['audio']['hc']['sberdisk']
    _b5.load_audio_model_weights_hc(url = url, out = True)

    _b5.load_audio_model_nn()
    url = _b5.weights_for_big5_['audio']['nn']['sberdisk']
    _b5.load_audio_model_weights_nn(url = url, out = True)

    _b5.load_audio_models_b5(out = True)
    url_openness = _b5.weights_for_big5_['audio']['b5']['openness']['sberdisk']
    url_conscientiousness = _b5.weights_for_big5_['audio']['b5']['conscientiousness']['sberdisk']
    url_extraversion = _b5.weights_for_big5_['audio']['b5']['extraversion']['sberdisk']
    url_agreeableness = _b5.weights_for_big5_['audio']['b5']['agreeableness']['sberdisk']
    url_neuroticism = _b5.weights_for_big5_['audio']['b5']['neuroticism']['sberdisk']
    _b5.load_audio_models_weights_b5(
        url_openness = url_openness,
        url_conscientiousness = url_conscientiousness,
        url_extraversion = url_extraversion,
        url_agreeableness = url_agreeableness,
        url_neuroticism = url_neuroticism,
        out = True
    )

    _b5.get_audio_union_predictions(url_accuracy = URL_ACCURACY, out = True)

    assert _b5.df_accuracy_['Mean'].values[0] <= 0.1
    assert _b5.df_accuracy_['Mean'].values[1] >= 0.9

def test_get_video_union_predictions():
    _b5.path_to_save_ = PATH_SAVE_MODELS

    _b5.load_video_model_hc(out = False)
    url = _b5.weights_for_big5_['video']['hc']['sberdisk']
    _b5.load_video_model_weights_hc(url = url, out = False)

    _b5.load_video_model_deep_fe(out = False)
    url = _b5.weights_for_big5_['video']['fe']['sberdisk']
    _b5.load_video_model_weights_deep_fe(url = url, out = False)

    _b5.load_video_model_nn(out = False)
    url = _b5.weights_for_big5_['video']['nn']['sberdisk']
    _b5.load_video_model_weights_nn(url = url, out = False)

    _b5.load_video_models_b5(out = False)
    url_openness = _b5.weights_for_big5_['video']['b5']['openness']['sberdisk']
    url_conscientiousness = _b5.weights_for_big5_['video']['b5']['conscientiousness']['sberdisk']
    url_extraversion = _b5.weights_for_big5_['video']['b5']['extraversion']['sberdisk']
    url_agreeableness = _b5.weights_for_big5_['video']['b5']['agreeableness']['sberdisk']
    url_neuroticism = _b5.weights_for_big5_['video']['b5']['neuroticism']['sberdisk']
    _b5.load_video_models_weights_b5(
        url_openness = url_openness,
        url_conscientiousness = url_conscientiousness,
        url_extraversion = url_extraversion,
        url_agreeableness = url_agreeableness,
        url_neuroticism = url_neuroticism,
        out = False
    )

    _b5.get_video_union_predictions(url_accuracy = URL_ACCURACY, out = False)
    
    assert _b5.df_accuracy_['Mean'].values[0] <= 0.1
    assert _b5.df_accuracy_['Mean'].values[1] >= 0.9

def test_get_av_union_predictions():
    _b5.path_to_save_ = PATH_SAVE_MODELS

    _b5.load_audio_model_hc()
    _b5.load_audio_model_nn()

    url = _b5.weights_for_big5_['audio']['hc']['sberdisk']
    _b5.load_audio_model_weights_hc(url = url)

    url = _b5.weights_for_big5_['audio']['nn']['sberdisk']
    _b5.load_audio_model_weights_nn(url = url)

    _b5.load_video_model_hc()
    _b5.load_video_model_deep_fe()
    _b5.load_video_model_nn()

    url = _b5.weights_for_big5_['video']['hc']['sberdisk']
    _b5.load_video_model_weights_hc(url = url)

    url = _b5.weights_for_big5_['video']['fe']['sberdisk']
    _b5.load_video_model_weights_deep_fe(url = url)

    url = _b5.weights_for_big5_['video']['nn']['sberdisk']
    _b5.load_video_model_weights_nn(url = url)

    _b5.load_av_models_b5()
    url_openness = _b5.weights_for_big5_['av']['b5']['openness']['sberdisk']
    url_conscientiousness = _b5.weights_for_big5_['av']['b5']['conscientiousness']['sberdisk']
    url_extraversion = _b5.weights_for_big5_['av']['b5']['extraversion']['sberdisk']
    url_agreeableness = _b5.weights_for_big5_['av']['b5']['agreeableness']['sberdisk']
    url_neuroticism = _b5.weights_for_big5_['av']['b5']['neuroticism']['sberdisk']
    _b5.load_av_models_weights_b5(
        url_openness = url_openness,
        url_conscientiousness = url_conscientiousness,
        url_extraversion = url_extraversion,
        url_agreeableness = url_agreeableness,
        url_neuroticism = url_neuroticism,
        out = False
    )

    _b5.get_av_union_predictions(url_accuracy = URL_ACCURACY, out = False)

    assert _b5.df_accuracy_['Mean'].values[0] <= 0.1
    assert _b5.df_accuracy_['Mean'].values[1] >= 0.9
