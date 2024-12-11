"""
File: switching_modes.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Event handler for switching modes.
License: MIT License
"""

import gradio as gr

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


def event_handler_switching_modes(language, type_modes):
    lang_id, _ = get_language_settings(language)

    first_practical_task = next(iter(supported_practical_tasks[lang_id]))

    if type_modes == config_data.Settings_TYPE_MODES[0]:
        notifications = html_message(
            (
                config_data.InformationMessages_NOTI_VIDEOS[lang_id].split("(")[0]
                if lang_id == 0
                else config_data.InformationMessages_NOTI_VIDEOS[lang_id]
            ),
            False,
            True,
            "notifications",
        )
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
        type_modes_ui = gr.Radio(
            choices=config_data.Settings_TYPE_MODES,
            value=config_data.Settings_TYPE_MODES[1],
        )
        examples = button(
            config_data.OtherMessages_EXAMPLE_APP[lang_id],
            True,
            1,
            "./images/examples.ico",
            True,
            "examples_oceanai",
        )
    elif type_modes == config_data.Settings_TYPE_MODES[1]:
        notifications = html_message(
            config_data.InformationMessages_NOTI_VIDEOS[lang_id],
            False,
            True,
            "notifications",
        )
        files_ui = files_create_ui(
            label="{} ({})".format(
                config_data.OtherMessages_VIDEO_FILES[lang_id],
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
        type_modes_ui = gr.Radio(
            choices=config_data.Settings_TYPE_MODES,
            value=config_data.Settings_TYPE_MODES[0],
        )
        examples = button(
            config_data.OtherMessages_EXAMPLES_APP[lang_id],
            True,
            1,
            "./images/examples.ico",
            True,
            "examples_oceanai",
        )

    return (
        notifications,
        files_ui,
        webcam,
        switching_modes,
        type_modes_ui,
        video_create_ui(),
        examples,
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
