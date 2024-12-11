"""
File: tabs.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Gradio app tabs - Contains the definition of various tabs for the Gradio app interface.
License: MIT License
"""

import gradio as gr

# Importing necessary components for the Gradio app
from app.description import DESCRIPTIONS
from app.description_steps import STEP_1, STEP_2
from app.mbti_description import MBTI_DESCRIPTION, MBTI_DATA
from app.app import APP
from app.authors import AUTHORS
from app.data_init import weights_professions, interactive_professions
from app.requirements_app import read_requirements_to_df
from app.config import config_data
from app.practical_tasks import supported_practical_tasks
from app.utils import read_csv_file
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


def app_tab():
    description = gr.Markdown(
        value=DESCRIPTIONS[config_data.AppSettings_DEFAULT_LANG_ID]
    )

    step_1 = gr.HTML(value=STEP_1[config_data.AppSettings_DEFAULT_LANG_ID])

    with gr.Row(elem_classes="media-container"):
        with gr.Column():
            files = files_create_ui(
                label="{} ({})".format(
                    config_data.OtherMessages_VIDEO_FILES[
                        config_data.AppSettings_DEFAULT_LANG_ID
                    ],
                    ", ".join(config_data.Settings_SUPPORTED_VIDEO_EXT),
                ),
                file_types=[
                    f".{ext}" for ext in config_data.Settings_SUPPORTED_VIDEO_EXT
                ],
            )

            webcam = gr.Video(
                label="{} ({})".format(
                    config_data.OtherMessages_VIDEO_FILES[
                        config_data.AppSettings_DEFAULT_LANG_ID
                    ],
                    ", ".join(config_data.Settings_SUPPORTED_VIDEO_EXT),
                ),
                show_label=True,
                interactive=False,
                visible=False,
                mirror_webcam=False,
                include_audio=True,
                elem_classes="webcam",
                autoplay=False,
            )

            switching_modes = button(
                config_data.OtherMessages_SWITCHEHG_MODES_ONLINE[
                    config_data.AppSettings_DEFAULT_LANG_ID
                ],
                True,
                1,
                "./images/webcam.ico",
                True,
                "switching_modes",
            )

            type_modes = gr.Radio(
                choices=config_data.Settings_TYPE_MODES,
                value=config_data.Settings_TYPE_MODES[0],
                label=None,
                info=None,
                show_label=False,
                container=False,
                interactive=False,
                visible=False,
                render=True,
                elem_classes="type_modes",
            )

        video = video_create_ui()

    with gr.Row():
        examples = button(
            config_data.OtherMessages_EXAMPLES_APP[
                config_data.AppSettings_DEFAULT_LANG_ID
            ],
            True,
            1,
            "./images/examples.ico",
            True,
            "examples_oceanai",
        )
        calculate_pt_scores = button(
            config_data.OtherMessages_CALCULATE_PT_SCORES[
                config_data.AppSettings_DEFAULT_LANG_ID
            ],
            False,
            3,
            "./images/calculate_pt_scores.ico",
            True,
            "calculate_oceanai",
        )
        clear_app = button(
            config_data.OtherMessages_CLEAR_APP[
                config_data.AppSettings_DEFAULT_LANG_ID
            ],
            False,
            1,
            "./images/clear.ico",
            True,
            "clear_oceanai",
        )

    notifications = html_message(
        config_data.InformationMessages_NOTI_VIDEOS[
            config_data.AppSettings_DEFAULT_LANG_ID
        ],
        False,
        "notifications",
    )

    pt_scores = dataframe(visible=False)

    csv_pt_scores = files_create_ui(
        None,
        "single",
        [".csv"],
        config_data.OtherMessages_EXPORT_PT_SCORES[
            config_data.AppSettings_DEFAULT_LANG_ID
        ],
        True,
        False,
        False,
        "csv-container",
    )

    step_2 = gr.HTML(
        value=STEP_2[config_data.AppSettings_DEFAULT_LANG_ID], visible=False
    )

    first_practical_task = next(
        iter(supported_practical_tasks[config_data.AppSettings_DEFAULT_LANG_ID])
    )

    with gr.Column(scale=1, visible=False, render=True) as practical_tasks_column:
        practical_tasks = radio_create_ui(
            first_practical_task,
            config_data.Labels_PRACTICAL_TASKS_LABEL,
            list(
                map(
                    str,
                    supported_practical_tasks[
                        config_data.AppSettings_DEFAULT_LANG_ID
                    ].keys(),
                )
            ),
            config_data.InformationMessages_PRACTICAL_TASKS_INFO[
                config_data.AppSettings_DEFAULT_LANG_ID
            ],
            True,
            True,
        )

        practical_subtasks = radio_create_ui(
            supported_practical_tasks[config_data.AppSettings_DEFAULT_LANG_ID][
                first_practical_task
            ][0],
            config_data.Labels_PRACTICAL_SUBTASKS_LABEL[
                config_data.AppSettings_DEFAULT_LANG_ID
            ],
            supported_practical_tasks[config_data.AppSettings_DEFAULT_LANG_ID][
                first_practical_task
            ],
            config_data.InformationMessages_PRACTICAL_SUBTASKS_INFO[
                config_data.AppSettings_DEFAULT_LANG_ID
            ],
            True,
            True,
        )

        with gr.Row(
            visible=False,
            render=True,
            variant="default",
            elem_classes="settings-container",
        ) as settings_practical_tasks:
            dropdown_mbti = dropdown_create_ui(
                label=f"Potential candidates by Personality Type of MBTI ({len(config_data.Settings_DROPDOWN_MBTI)})",
                info=config_data.InformationMessages_DROPDOWN_MBTI_INFO,
                choices=config_data.Settings_DROPDOWN_MBTI,
                value=config_data.Settings_DROPDOWN_MBTI[0],
                visible=False,
                elem_classes="dropdown-container",
            )

            threshold_mbti = number_create_ui(
                value=0.5,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label=config_data.Labels_THRESHOLD_MBTI_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            threshold_professional_skills = number_create_ui(
                value=0.5,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label=config_data.Labels_THRESHOLD_PROFESSIONAL_SKILLS_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            dropdown_professional_skills = dropdown_create_ui(
                label=f"Professional skills ({len(config_data.Settings_DROPDOWN_PROFESSIONAL_SKILLS)})",
                info=config_data.InformationMessages_DROPDOWN_PROFESSIONAL_SKILLS_INFO,
                choices=config_data.Settings_DROPDOWN_PROFESSIONAL_SKILLS,
                value=config_data.Settings_DROPDOWN_PROFESSIONAL_SKILLS[0],
                visible=False,
                elem_classes="dropdown-container",
            )

            target_score_ope = number_create_ui(
                value=config_data.Values_TARGET_SCORES[0],
                minimum=0.0,
                maximum=1.0,
                step=0.000001,
                label=config_data.Labels_TARGET_SCORE_OPE_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            target_score_con = number_create_ui(
                value=config_data.Values_TARGET_SCORES[1],
                minimum=0.0,
                maximum=1.0,
                step=0.000001,
                label=config_data.Labels_TARGET_SCORE_CON_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            target_score_ext = number_create_ui(
                value=config_data.Values_TARGET_SCORES[2],
                minimum=0.0,
                maximum=1.0,
                step=0.000001,
                label=config_data.Labels_TARGET_SCORE_EXT_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            target_score_agr = number_create_ui(
                value=config_data.Values_TARGET_SCORES[3],
                minimum=0.0,
                maximum=1.0,
                step=0.000001,
                label=config_data.Labels_TARGET_SCORE_AGR_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            target_score_nneu = number_create_ui(
                value=config_data.Values_TARGET_SCORES[4],
                minimum=0.0,
                maximum=1.0,
                step=0.000001,
                label=config_data.Labels_TARGET_SCORE_NNEU_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            equal_coefficient = number_create_ui(
                value=0.5,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label=config_data.Labels_EQUAL_COEFFICIENT_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            df_correlation_coefficients = read_csv_file(
                config_data.Links_CAR_CHARACTERISTICS,
                ["Trait", "Style and performance", "Safety and practicality"],
            )

            number_priority = number_create_ui(
                value=1,
                minimum=1,
                maximum=df_correlation_coefficients.columns.size,
                step=1,
                label=config_data.Labels_NUMBER_PRIORITY_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(
                    1, df_correlation_coefficients.columns.size
                ),
                show_label=True,
                interactive=True,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            number_importance_traits = number_create_ui(
                value=1,
                minimum=1,
                maximum=5,
                step=1,
                label=config_data.Labels_NUMBER_IMPORTANCE_TRAITS_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(1, 5),
                show_label=True,
                interactive=True,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            threshold_consumer_preferences = number_create_ui(
                value=0.55,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label=config_data.Labels_THRESHOLD_CONSUMER_PREFERENCES_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            dropdown_candidates = dropdown_create_ui(
                label=f"Potential candidates by professional responsibilities ({len(config_data.Settings_DROPDOWN_CANDIDATES)})",
                info=config_data.InformationMessages_DROPDOWN_CANDIDATES_INFO,
                choices=config_data.Settings_DROPDOWN_CANDIDATES,
                value=config_data.Settings_DROPDOWN_CANDIDATES[0],
                visible=False,
                elem_classes="dropdown-container",
            )

            number_openness = number_create_ui(
                value=weights_professions[0],
                minimum=config_data.Values_0_100[0],
                maximum=config_data.Values_0_100[1],
                step=1,
                label=config_data.Labels_NUMBER_IMPORTANCE_OPE_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(
                    config_data.Values_0_100[0], config_data.Values_0_100[1]
                ),
                show_label=True,
                interactive=interactive_professions,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            number_conscientiousness = number_create_ui(
                value=weights_professions[1],
                minimum=config_data.Values_0_100[0],
                maximum=config_data.Values_0_100[1],
                step=1,
                label=config_data.Labels_NUMBER_IMPORTANCE_CON_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(
                    config_data.Values_0_100[0], config_data.Values_0_100[1]
                ),
                show_label=True,
                interactive=interactive_professions,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            number_extraversion = number_create_ui(
                value=weights_professions[2],
                minimum=config_data.Values_0_100[0],
                maximum=config_data.Values_0_100[1],
                step=1,
                label=config_data.Labels_NUMBER_IMPORTANCE_EXT_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(
                    config_data.Values_0_100[0], config_data.Values_0_100[1]
                ),
                show_label=True,
                interactive=interactive_professions,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            number_agreeableness = number_create_ui(
                value=weights_professions[3],
                minimum=config_data.Values_0_100[0],
                maximum=config_data.Values_0_100[1],
                step=1,
                label=config_data.Labels_NUMBER_IMPORTANCE_AGR_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(
                    config_data.Values_0_100[0], config_data.Values_0_100[1]
                ),
                show_label=True,
                interactive=interactive_professions,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

            number_non_neuroticism = number_create_ui(
                value=weights_professions[4],
                minimum=config_data.Values_0_100[0],
                maximum=config_data.Values_0_100[1],
                step=1,
                label=config_data.Labels_NUMBER_IMPORTANCE_NNEU_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(
                    config_data.Values_0_100[0], config_data.Values_0_100[1]
                ),
                show_label=True,
                interactive=interactive_professions,
                visible=False,
                render=True,
                elem_classes="number-container",
            )

        calculate_practical_task = button(
            config_data.OtherMessages_CALCULATE_PRACTICAL_TASK,
            True,
            1,
            "./images/pt.ico",
            False,
            "calculate_practical_task",
        )

        with gr.Row(
            visible=False,
            render=True,
            variant="default",
        ) as sorted_videos:
            with gr.Column(scale=1, visible=False, render=True) as sorted_videos_column:
                practical_task_sorted = dataframe(visible=False)

                with gr.Accordion(
                    label=config_data.Labels_NOTE_MBTI_LABEL,
                    open=False,
                    visible=False,
                ) as mbti_accordion:
                    mbti_description = gr.HTML(value=MBTI_DESCRIPTION, visible=False)

                    mbti_description_data = dataframe(
                        headers=MBTI_DATA.columns.tolist(),
                        values=MBTI_DATA.values.tolist(),
                        visible=False,
                        elem_classes="mbti-dataframe",
                    )

                csv_practical_task_sorted = files_create_ui(
                    None,
                    "single",
                    [".csv"],
                    config_data.OtherMessages_EXPORT_PS,
                    True,
                    False,
                    False,
                    "csv-container",
                )

            with gr.Column(
                scale=1,
                visible=False,
                render=True,
                elem_classes="video-column-container",
            ) as video_sorted_column:
                video_sorted = video_create_ui(
                    visible=False, elem_classes="video-sorted-container"
                )

                with gr.Column(scale=1, visible=False, render=True) as metadata:
                    with gr.Row(
                        visible=False, render=True, variant="default"
                    ) as metadata_1:
                        with gr.Row(
                            visible=False,
                            render=True,
                            variant="default",
                            elem_classes="name-container",
                        ) as name_row:
                            name_logo = gr.Image(
                                value="images/name.svg",
                                container=False,
                                interactive=False,
                                show_label=False,
                                visible=False,
                                show_download_button=False,
                                elem_classes="metadata_name-logo",
                                show_fullscreen_button=False,
                            )

                            name = textbox_create_ui(
                                "First name",
                                "text",
                                "First name",
                                None,
                                None,
                                1,
                                True,
                                False,
                                False,
                                False,
                                1,
                                False,
                            )

                        with gr.Row(
                            visible=False,
                            render=True,
                            variant="default",
                            elem_classes="surname-container",
                        ) as surname_row:
                            surname_logo = gr.Image(
                                value="images/name.svg",
                                container=False,
                                interactive=False,
                                show_label=False,
                                visible=False,
                                show_download_button=False,
                                elem_classes="metadata_surname-logo",
                                show_fullscreen_button=False,
                            )

                            surname = textbox_create_ui(
                                "Last name",
                                "text",
                                "Last name",
                                None,
                                None,
                                1,
                                True,
                                False,
                                False,
                                False,
                                1,
                                False,
                            )
                    with gr.Row(
                        visible=False, render=True, variant="default"
                    ) as metadata_2:
                        with gr.Row(
                            visible=False,
                            render=True,
                            variant="default",
                            elem_classes="email-container",
                        ) as email_row:
                            email_logo = gr.Image(
                                value="images/email.svg",
                                container=False,
                                interactive=False,
                                show_label=False,
                                visible=False,
                                show_download_button=False,
                                elem_classes="metadata_email-logo",
                                show_fullscreen_button=False,
                            )

                            email = textbox_create_ui(
                                "example@example.com",
                                "email",
                                "Email",
                                None,
                                None,
                                1,
                                True,
                                False,
                                False,
                                False,
                                1,
                                False,
                            )

                        with gr.Row(
                            visible=False,
                            render=True,
                            variant="default",
                            elem_classes="phone-container",
                        ) as phone_row:
                            phone_logo = gr.Image(
                                value="images/phone.svg",
                                container=False,
                                interactive=False,
                                show_label=False,
                                visible=False,
                                show_download_button=False,
                                elem_classes="metadata_phone-logo",
                                show_fullscreen_button=False,
                            )

                            phone = textbox_create_ui(
                                "+1 (555) 123-4567",
                                "text",
                                "Phone number",
                                None,
                                None,
                                1,
                                True,
                                False,
                                False,
                                False,
                                1,
                                False,
                            )

    practical_subtasks_selected = gr.JSON(
        value={
            str(task): supported_practical_tasks[index].get(task, [None])[0]
            for index in range(len(supported_practical_tasks))
            for task in supported_practical_tasks[index].keys()
        },
        visible=False,
        render=True,
    )

    in_development = html_message(
        config_data.InformationMessages_NOTI_IN_DEV, False, False
    )

    return (
        description,
        step_1,
        notifications,
        files,
        webcam,
        switching_modes,
        type_modes,
        video,
        examples,
        calculate_pt_scores,
        clear_app,
        pt_scores,
        csv_pt_scores,
        step_2,
        practical_tasks,
        practical_subtasks,
        settings_practical_tasks,
        dropdown_mbti,
        threshold_mbti,
        threshold_professional_skills,
        dropdown_professional_skills,
        target_score_ope,
        target_score_con,
        target_score_ext,
        target_score_agr,
        target_score_nneu,
        equal_coefficient,
        number_priority,
        number_importance_traits,
        threshold_consumer_preferences,
        dropdown_candidates,
        number_openness,
        number_conscientiousness,
        number_extraversion,
        number_agreeableness,
        number_non_neuroticism,
        calculate_practical_task,
        practical_subtasks_selected,
        practical_tasks_column,
        sorted_videos,
        sorted_videos_column,
        practical_task_sorted,
        csv_practical_task_sorted,
        mbti_accordion,
        mbti_description,
        mbti_description_data,
        video_sorted_column,
        video_sorted,
        metadata,
        metadata_1,
        name_row,
        name_logo,
        name,
        surname_row,
        surname_logo,
        surname,
        metadata_2,
        email_row,
        email_logo,
        email,
        phone_row,
        phone_logo,
        phone,
        in_development,
    )


def about_app_tab():
    return gr.HTML(value=APP)


def about_authors_tab():
    return gr.HTML(value=AUTHORS)


def requirements_app_tab():
    requirements_df = read_requirements_to_df()

    return dataframe(
        headers=requirements_df.columns.tolist(),
        values=requirements_df.values.tolist(),
        visible=True,
        elem_classes="requirements-dataframe",
    )
