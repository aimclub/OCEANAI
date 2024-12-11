"""
File: files.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Event handler for files.
License: MIT License
"""

import gradio as gr
from pathlib import Path

# Importing necessary components for the Gradio app
from app.config import config_data
from app.components import html_message, video_create_ui, button
from app.utils import get_language_settings


def event_handler_files(language, files, video, pt_scores):
    lang_id, _ = get_language_settings(language)

    if not files:
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

    if video not in files:
        video = files[0]

    return (
        html_message(
            config_data.OtherMessages_NOTI_CALCULATE[lang_id],
            True,
            False if pt_scores.shape[1] >= 7 else True,
            "notifications",
        ),
        video_create_ui(
            value=video,
            label=config_data.OtherMessages_VIDEO_PLAYER[lang_id],
            file_name=Path(Path(video).name).name,
        ),
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


def event_handler_files_select(language, files, evt: gr.SelectData):
    lang_id, _ = get_language_settings(language)

    return video_create_ui(
        value=files[evt.index],
        label=config_data.OtherMessages_VIDEO_PLAYER[lang_id],
        file_name=evt.value,
    )


def event_handler_files_delete(language, files, video, evt: gr.DeletedFileData):
    global block_event_handler_files

    lang_id, _ = get_language_settings(language)

    if video == evt.file.path:
        video = files[0]

    return video_create_ui(
        value=video,
        label=config_data.OtherMessages_VIDEO_PLAYER[lang_id],
        file_name=Path(Path(video).name).name,
    )
