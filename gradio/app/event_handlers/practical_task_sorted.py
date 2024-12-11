"""
File: practical_task_sorted.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Event handler for the practical task sorted to the Gradio app.
License: MIT License
"""

import gradio as gr
from pathlib import Path

# Importing necessary components for the Gradio app
from app.config import config_data
from app.video_metadata import video_metadata
from app.components import video_create_ui, textbox_create_ui


def event_handler_practical_task_sorted(
    type_modes, files, video, practical_task_sorted, evt_data: gr.SelectData
):
    if type_modes == config_data.Settings_TYPE_MODES[0]:
        person_id = (
            int(
                practical_task_sorted.iloc[evt_data.index[0]][
                    config_data.Dataframes_PT_SCORES[0][0]
                ]
            )
            - 1
        )
    elif type_modes == config_data.Settings_TYPE_MODES[1]:
        files = [video]

        person_id = 0

    if evt_data.index[0] == 0:
        label = "Best"
    else:
        label = ""
    label += " " + config_data.Dataframes_PT_SCORES[0][0]

    try:
        is_filename = Path(files[person_id]).name in video_metadata
    except IndexError:
        is_filename = False
        person_id = 0

    if is_filename:
        person_metadata_list = video_metadata[Path(files[person_id]).name]

        person_metadata = (
            gr.Column(visible=True),
            gr.Row(visible=True),
            gr.Row(visible=True),
            gr.Image(visible=True),
            textbox_create_ui(
                person_metadata_list[0],
                "text",
                "First name",
                None,
                None,
                1,
                True,
                False,
                True,
                False,
                1,
                False,
            ),
            gr.Row(visible=True),
            gr.Image(visible=True),
            textbox_create_ui(
                person_metadata_list[1],
                "text",
                "Last name",
                None,
                None,
                1,
                True,
                False,
                True,
                False,
                1,
                False,
            ),
            gr.Row(visible=True),
            gr.Row(visible=True),
            gr.Image(visible=True),
            textbox_create_ui(
                person_metadata_list[2],
                "email",
                "Email",
                None,
                None,
                1,
                True,
                False,
                True,
                False,
                1,
                False,
            ),
            gr.Row(visible=True),
            gr.Image(visible=True),
            textbox_create_ui(
                person_metadata_list[3],
                "text",
                "Phone number",
                None,
                None,
                1,
                True,
                False,
                True,
                False,
                1,
                False,
            ),
        )
    else:
        person_metadata = (
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
        )

    existing_tuple = (
        gr.Column(visible=True),
        video_create_ui(
            value=files[person_id],
            file_name=Path(files[person_id]).name,
            label=f"{label} - " + str(person_id + 1),
            visible=True,
            elem_classes="video-sorted-container",
        ),
    )

    return existing_tuple + person_metadata
