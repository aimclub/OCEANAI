"""
File: languages.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Selected language event handlers for Gradio app.
License: MIT License
"""

import gradio as gr
from pathlib import Path

# Importing necessary components for the Gradio app
from app.description import DESCRIPTIONS
from app.description_steps import STEP_1, STEP_2
from app.config import config_data
from app.components import (
    files_create_ui,
    video_create_ui,
    dropdown_create_ui,
    button,
    html_message,
    dataframe,
    radio_create_ui,
)
from app.utils import get_language_settings
from app.practical_tasks import supported_practical_tasks


def event_handler_languages(
    languages,
    files,
    video,
    type_modes,
    pt_scores,
    csv_pt_scores,
    practical_tasks,
    practical_subtasks,
):
    lang_id, choices = get_language_settings(languages)
    lang_id_inverse = {0: 1, 1: 0}.get(lang_id, None)

    if type_modes == config_data.Settings_TYPE_MODES[0]:
        files_ui = files_create_ui(
            label="{} ({})".format(
                config_data.OtherMessages_VIDEO_FILES[
                    config_data.AppSettings_DEFAULT_LANG_ID
                ],
                ", ".join(config_data.Settings_SUPPORTED_VIDEO_EXT),
            ),
            file_types=[f".{ext}" for ext in config_data.Settings_SUPPORTED_VIDEO_EXT],
        )
        webcam = gr.Video(interactive=False, visible=False)
        switching_modes = button(
            config_data.OtherMessages_SWITCHEHG_MODES_ONLINE[lang_id],
            True,
            1,
            "./images/webcam.ico",
            True,
            "switching_modes",
        )
        examples = button(
            config_data.OtherMessages_EXAMPLES_APP[lang_id],
            True,
            1,
            "./images/examples.ico",
            True,
            "examples_oceanai",
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
        webcam = gr.Video(interactive=True, visible=True)
        switching_modes = button(
            config_data.OtherMessages_SWITCHEHG_MODES_OFFLINE[lang_id],
            True,
            1,
            "./images/videos.ico",
            True,
            "switching_modes",
        )
        examples = button(
            config_data.OtherMessages_EXAMPLE_APP[lang_id],
            True,
            1,
            "./images/examples.ico",
            True,
            "examples_oceanai",
        )

    if not video:
        video = video_create_ui(label=config_data.OtherMessages_VIDEO_PLAYER[lang_id])
        noti_videos = html_message(
            (
                config_data.InformationMessages_NOTI_VIDEOS[lang_id].split("(")[0]
                if lang_id == 0
                else config_data.InformationMessages_NOTI_VIDEOS[lang_id]
            ),
            False,
            True,
            "notifications",
        )
    else:
        video = video_create_ui(
            value=video,
            label=config_data.OtherMessages_VIDEO_PLAYER[lang_id],
            file_name=Path(video).name,
        )
        noti_videos = html_message(
            config_data.OtherMessages_NOTI_CALCULATE[lang_id],
            True,
            False if pt_scores.shape[1] >= 7 else True,
            "notifications",
        )

    csv_pt_scores = files_create_ui(
        csv_pt_scores if pt_scores.shape[1] >= 7 else None,
        "single",
        [".csv"],
        config_data.OtherMessages_EXPORT_PT_SCORES[lang_id],
        True,
        False,
        True if pt_scores.shape[1] >= 7 else False,
        "csv-container",
    )
    step_2 = gr.HTML(
        value=STEP_2[lang_id], visible=True if pt_scores.shape[1] >= 7 else False
    )

    practical_tasks_column = gr.Column(
        visible=True if pt_scores.shape[1] >= 7 else False
    )

    if pt_scores.shape[1] >= 7:
        pt_scores = dataframe(
            headers=(config_data.Dataframes_PT_SCORES[lang_id]),
            values=pt_scores.values.tolist(),
            visible=True,
        )
    else:
        pt_scores = dataframe(visible=False)

    current_lang_tasks = list(map(str, supported_practical_tasks[lang_id].keys()))
    inverse_lang_tasks = list(
        map(str, supported_practical_tasks[lang_id_inverse].keys())
    )

    if practical_tasks in inverse_lang_tasks:
        practical_task = current_lang_tasks[inverse_lang_tasks.index(practical_tasks)]
    else:
        practical_task = next(iter(supported_practical_tasks[lang_id]))

    # print(current_lang_tasks, "\n")
    # print(inverse_lang_tasks, "\n")
    # print(practical_tasks, "\n")
    # print(supported_practical_tasks, "\n")
    # print(practical_subtasks, "\n")

    return (
        gr.Markdown(value=DESCRIPTIONS[lang_id]),
        gr.HTML(value=STEP_1[lang_id]),
        gr.Image(
            value=config_data.StaticPaths_IMAGES + config_data.Images_LANGUAGES[lang_id]
        ),
        dropdown_create_ui(
            label=None,
            info=None,
            choices=choices,
            value=choices[lang_id],
            visible=True,
            show_label=False,
            elem_classes="dropdown-language-container",
        ),
        gr.Tab(config_data.Labels_APP_LABEL[lang_id]),
        gr.Tab(config_data.Labels_ABOUT_APP_LABEL[lang_id]),
        gr.Tab(config_data.Labels_ABOUT_AUTHORS_LABEL[lang_id]),
        gr.Tab(config_data.Labels_REQUIREMENTS_LABEL[lang_id]),
        files_ui,
        webcam,
        switching_modes,
        video,
        examples,
        button(
            config_data.OtherMessages_CALCULATE_PT_SCORES[lang_id],
            True if files else False,
            3,
            "./images/calculate_pt_scores.ico",
            True,
            "calculate_oceanai",
        ),
        button(
            config_data.OtherMessages_CLEAR_APP[lang_id],
            True if files else False,
            1,
            "./images/clear.ico",
            True,
            "clear_oceanai",
        ),
        noti_videos,
        pt_scores,
        csv_pt_scores,
        step_2,
        practical_tasks_column,
        radio_create_ui(
            practical_task,
            config_data.Labels_PRACTICAL_TASKS_LABEL[lang_id],
            current_lang_tasks,
            config_data.InformationMessages_PRACTICAL_TASKS_INFO[lang_id],
            True,
            True,
        ),
        radio_create_ui(
            supported_practical_tasks[lang_id][practical_task][0],
            config_data.Labels_PRACTICAL_SUBTASKS_LABEL[lang_id],
            supported_practical_tasks[lang_id][practical_task],
            config_data.InformationMessages_PRACTICAL_SUBTASKS_INFO[lang_id],
            True,
            True,
        ),
    )
