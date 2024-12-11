"""
File: examples_blocks.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Event handler for the addition of examples to the Gradio app.
License: MIT License
"""

import re
import gradio as gr
from pathlib import Path

# Importing necessary components for the Gradio app
from app.config import config_data
from app.description_steps import STEP_2
from app.practical_tasks import supported_practical_tasks
from app.components import (
    html_message,
    files_create_ui,
    video_create_ui,
    button,
    dataframe,
    radio_create_ui,
    number_create_ui,
    dropdown_create_ui,
    textbox_create_ui,
)
from app.utils import get_language_settings


def event_handler_examples_blocks(language, type_modes):
    lang_id, _ = get_language_settings(language)

    first_practical_task = next(iter(supported_practical_tasks[lang_id]))

    videos_dir = Path("videos")
    video_files = sorted(
        (str(p) for p in videos_dir.glob("*.mp4")),
        key=lambda x: int(re.search(r"\d+", Path(x).stem).group()),
    )

    if type_modes == config_data.Settings_TYPE_MODES[0]:
        files_ui = files_create_ui(
            value=video_files,
            label="{} ({})".format(
                config_data.OtherMessages_VIDEO_FILES[lang_id],
                ", ".join(config_data.Settings_SUPPORTED_VIDEO_EXT),
            ),
            file_types=[f".{ext}" for ext in config_data.Settings_SUPPORTED_VIDEO_EXT],
        )
    elif type_modes == config_data.Settings_TYPE_MODES[1]:
        files_ui = files_create_ui(
            label="{} ({})".format(
                config_data.OtherMessages_VIDEO_FILES[lang_id],
                ", ".join(config_data.Settings_SUPPORTED_VIDEO_EXT),
            ),
            file_types=[f".{ext}" for ext in config_data.Settings_SUPPORTED_VIDEO_EXT],
            interactive=False,
            visible=False,
        )

    return (
        html_message(
            config_data.OtherMessages_NOTI_CALCULATE[lang_id],
            True,
            True,
            "notifications",
        ),
        files_ui,
        video_create_ui(
            value=video_files[0],
            label=config_data.OtherMessages_VIDEO_PLAYER[lang_id],
            file_name=Path(Path(video_files[0]).name).name,
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
        dataframe(visible=False),
        files_create_ui(
            None,
            "single",
            [".csv"],
            config_data.OtherMessages_EXPORT_PT_SCORES[lang_id],
            True,
            False,
            False,
            "csv-container",
        ),
        gr.HTML(value=STEP_2[lang_id], visible=False),
        gr.Column(visible=False),
        radio_create_ui(
            first_practical_task,
            config_data.Labels_PRACTICAL_TASKS_LABEL[lang_id],
            list(map(str, supported_practical_tasks[lang_id].keys())),
            config_data.InformationMessages_PRACTICAL_TASKS_INFO[lang_id],
            True,
            True,
        ),
        radio_create_ui(
            supported_practical_tasks[lang_id][first_practical_task][0],
            config_data.Labels_PRACTICAL_SUBTASKS_LABEL[lang_id],
            supported_practical_tasks[lang_id][first_practical_task],
            config_data.InformationMessages_PRACTICAL_SUBTASKS_INFO[lang_id],
            True,
            True,
        ),
        gr.JSON(
            value={
                str(task): supported_practical_tasks[index].get(task, [None])[0]
                for index in range(len(supported_practical_tasks))
                for task in supported_practical_tasks[index].keys()
            },
            visible=False,
            render=True,
        ),
        gr.Column(visible=False),
        dropdown_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        dropdown_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        dropdown_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        number_create_ui(visible=False),
        gr.Row(visible=False),
        gr.Column(visible=False),
        dataframe(visible=False),
        files_create_ui(
            None,
            "single",
            [".csv"],
            config_data.OtherMessages_EXPORT_PS,
            True,
            False,
            False,
            "csv-container",
        ),
        gr.Accordion(visible=False),
        gr.HTML(visible=False),
        dataframe(visible=False),
        gr.Column(visible=False),
        video_create_ui(visible=False),
        gr.Column(visible=False),
        gr.Row(visible=False),
        gr.Row(visible=False),
        gr.Image(visible=False),
        textbox_create_ui(visible=False),
        gr.Row(visible=False),
        gr.Image(visible=False),
        textbox_create_ui(visible=False),
        gr.Row(visible=False),
        gr.Row(visible=False),
        gr.Image(visible=False),
        textbox_create_ui(visible=False),
        gr.Row(visible=False),
        gr.Image(visible=False),
        textbox_create_ui(visible=False),
        html_message(config_data.InformationMessages_NOTI_IN_DEV, False, False),
    )
