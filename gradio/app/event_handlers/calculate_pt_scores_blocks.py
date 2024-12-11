"""
File: clear_blocks.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Event handler for clearing Gradio app blocks and components.
License: MIT License
"""

import gradio as gr

# Importing necessary components for the Gradio app
from app.oceanai_init import b5
from app.config import config_data
from app.description_steps import STEP_2
from app.utils import get_language_settings
from app.practical_tasks import supported_practical_tasks
from app.components import (
    html_message,
    button,
    dataframe,
    files_create_ui,
    radio_create_ui,
    number_create_ui,
    dropdown_create_ui,
    video_create_ui,
    textbox_create_ui,
)


def event_handler_calculate_pt_scores_blocks(
    language, type_modes, files, video, evt_data: gr.EventData
):
    _ = evt_data.target.__class__.__name__

    lang_id, _ = get_language_settings(language)

    out = False

    try:
        b5.get_avt_predictions_gradio(
            paths=(
                files if type_modes == config_data.Settings_TYPE_MODES[0] else [video]
            ),
            url_accuracy="",
            accuracy=False,
            lang="en",
            out=out,
        )
    except TypeError:
        out = True

    if type_modes == config_data.Settings_TYPE_MODES[0]:
        supported_practical_tasks_ren = supported_practical_tasks
    elif type_modes == config_data.Settings_TYPE_MODES[1]:
        rename_map = {
            "Ranking potential candidates by professional responsibilities": "Estimating professional abilities",
            "Ранжирование потенциальных кандидатов по профессиональным обязанностям": "Определить профессиональные возможности",
        }

        supported_practical_tasks_ren = [
            {rename_map.get(k, k): v for k, v in d.items()}
            for d in supported_practical_tasks
        ]

    first_practical_task = next(iter(supported_practical_tasks_ren[lang_id]))

    if out or len(b5.df_files_) == 0:
        gr.Warning(config_data.OtherMessages_CALCULATE_PT_SCORES_ERR[lang_id])

        return (
            html_message(
                config_data.OtherMessages_CALCULATE_PT_SCORES_ERR[lang_id],
                False,
                "notifications",
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
                config_data.Labels_PRACTICAL_TASKS_LABEL,
                list(map(str, supported_practical_tasks_ren[lang_id].keys())),
                config_data.InformationMessages_PRACTICAL_TASKS_INFO[lang_id],
                True,
                True,
            ),
            radio_create_ui(
                supported_practical_tasks_ren[lang_id][first_practical_task][0],
                config_data.Labels_PRACTICAL_SUBTASKS_LABEL[lang_id],
                supported_practical_tasks_ren[lang_id][first_practical_task],
                config_data.InformationMessages_PRACTICAL_SUBTASKS_INFO[lang_id],
                True,
                True,
            ),
            gr.JSON(
                value={
                    str(task): supported_practical_tasks_ren[index].get(task, [None])[0]
                    for index in range(len(supported_practical_tasks_ren))
                    for task in supported_practical_tasks_ren[index].keys()
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
            button(
                config_data.OtherMessages_CALCULATE_PRACTICAL_TASK,
                True,
                1,
                "./images/pt.ico",
                False,
                "calculate_practical_task",
            ),
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

    b5.df_files_.to_csv(config_data.Filenames_PT_SCORES)

    df_files = b5.df_files_.copy()
    df_files.reset_index(inplace=True)

    if type_modes == config_data.Settings_TYPE_MODES[0]:
        practical_tasks_choices = list(
            map(str, supported_practical_tasks_ren[lang_id].keys())
        )
    elif type_modes == config_data.Settings_TYPE_MODES[1]:
        practical_tasks_choices = [
            value
            for i, value in enumerate(
                map(str, supported_practical_tasks_ren[lang_id].keys())
            )
            if i not in {1}
        ]

    return (
        html_message(
            config_data.InformationMessages_NOTI_VIDEOS[lang_id],
            False,
            False,
            "notifications",
        ),
        dataframe(
            headers=(config_data.Dataframes_PT_SCORES[lang_id]),
            values=df_files.values.tolist(),
            visible=True,
        ),
        files_create_ui(
            config_data.Filenames_PT_SCORES,
            "single",
            [".csv"],
            config_data.OtherMessages_EXPORT_PT_SCORES[lang_id],
            True,
            False,
            True,
            "csv-container",
        ),
        gr.HTML(value=STEP_2[lang_id], visible=True),
        gr.Column(visible=True),
        radio_create_ui(
            first_practical_task,
            config_data.Labels_PRACTICAL_TASKS_LABEL[lang_id],
            practical_tasks_choices,
            config_data.InformationMessages_PRACTICAL_TASKS_INFO[lang_id],
            True,
            True,
        ),
        radio_create_ui(
            supported_practical_tasks_ren[lang_id][first_practical_task][0],
            config_data.Labels_PRACTICAL_SUBTASKS_LABEL[lang_id],
            supported_practical_tasks_ren[lang_id][first_practical_task],
            config_data.InformationMessages_PRACTICAL_SUBTASKS_INFO[lang_id],
            True,
            True,
        ),
        gr.JSON(
            value={
                str(task): supported_practical_tasks_ren[index].get(task, [None])[0]
                for index in range(len(supported_practical_tasks_ren))
                for task in supported_practical_tasks_ren[index].keys()
            },
            visible=False,
            render=True,
        ),
        gr.Column(
            visible=True if type_modes == config_data.Settings_TYPE_MODES[0] else False
        ),
        dropdown_create_ui(
            label=f"Potential candidates by Personality Type of MBTI ({len(config_data.Settings_DROPDOWN_MBTI)})",
            info=config_data.InformationMessages_DROPDOWN_MBTI_INFO,
            choices=config_data.Settings_DROPDOWN_MBTI,
            value=config_data.Settings_DROPDOWN_MBTI[0],
            visible=True if type_modes == config_data.Settings_TYPE_MODES[0] else False,
            elem_classes="dropdown-container",
        ),
        number_create_ui(
            value=0.5,
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            label=config_data.Labels_THRESHOLD_MBTI_LABEL,
            info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
            show_label=True,
            interactive=True,
            visible=True if type_modes == config_data.Settings_TYPE_MODES[0] else False,
            render=True,
            elem_classes="number-container",
        ),
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
        button(
            config_data.OtherMessages_CALCULATE_PRACTICAL_TASK,
            True,
            1,
            "./images/pt.ico",
            True,
            "calculate_practical_task",
        ),
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
