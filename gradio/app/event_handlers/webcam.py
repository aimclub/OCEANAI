"""
File: webcam.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Event handler for webcam.
License: MIT License
"""

import gradio as gr
from pathlib import Path

# Importing necessary components for the Gradio app
from app.config import config_data
from app.components import html_message, video_create_ui, button
from app.utils import get_language_settings, webm2mp4


def event_handler_webcam(language, webcam, pt_scores):
    lang_id, _ = get_language_settings(language)

    if not webcam:
        return (
            html_message(
                (
                    config_data.InformationMessages_NOTI_VIDEOS[lang_id].split("(")[0]
                    if lang_id == 0
                    else config_data.InformationMessages_NOTI_VIDEOS[lang_id]
                ),
                False,
                True,
                "notifications",
            ),
            video_create_ui(label=config_data.OtherMessages_VIDEO_PLAYER[lang_id]),
            gr.Video(value=None),
            button(
                config_data.OtherMessages_CALCULATE_PT_SCORES[lang_id],
                False,
                3,
                "./images/calculate_pt_scores.ico",
                True,
                "calculate_oceanai",
            ),
            button(
                config_data.OtherMessages_CLEAR_APP[lang_id],
                False,
                1,
                "./images/clear.ico",
                True,
                "clear_oceanai",
            ),
        )

    if webcam.split(".")[-1] == "webm":
        webcam = webm2mp4(webcam)

    return (
        html_message(
            config_data.OtherMessages_NOTI_CALCULATE[lang_id],
            True,
            False if pt_scores.shape[1] >= 7 else True,
            "notifications",
        ),
        video_create_ui(
            value=webcam,
            label=config_data.OtherMessages_VIDEO_PLAYER[lang_id],
            file_name=Path(Path(webcam).name).name,
        ),
        gr.Video(value=None),
        button(
            config_data.OtherMessages_CALCULATE_PT_SCORES[lang_id],
            True,
            3,
            "./images/calculate_pt_scores.ico",
            True,
            "calculate_oceanai",
        ),
        button(
            config_data.OtherMessages_CLEAR_APP[lang_id],
            True,
            1,
            "./images/clear.ico",
            True,
            "clear_oceanai",
        ),
    )
