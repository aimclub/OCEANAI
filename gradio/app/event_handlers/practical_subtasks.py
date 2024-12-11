"""
File: practical_subtasks.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Event handler for Gradio app to filter practical subtasks based on selected practical subtasks.
License: MIT License
"""

import gradio as gr

# Importing necessary components for the Gradio app
from app.config import config_data
from app.utils import read_csv_file, get_language_settings
from app.data_init import weights_professions, interactive_professions
from app.components import number_create_ui, dropdown_create_ui


def event_handler_practical_subtasks(
    language,
    type_modes,
    practical_tasks,
    practical_subtasks,
    practical_subtasks_selected,
):
    lang_id, _ = get_language_settings(language)

    practical_subtasks_selected[practical_tasks] = practical_subtasks

    visible_subtasks = (
        True if type_modes == config_data.Settings_TYPE_MODES[0] else False
    )

    if practical_subtasks.lower() == "16 personality types of mbti":
        return (
            practical_subtasks_selected,
            gr.Column(visible=visible_subtasks),
            dropdown_create_ui(
                label=f"Potential candidates by Personality Type of MBTI ({len(config_data.Settings_DROPDOWN_MBTI)})",
                info=config_data.InformationMessages_DROPDOWN_MBTI_INFO,
                choices=config_data.Settings_DROPDOWN_MBTI,
                value=config_data.Settings_DROPDOWN_MBTI[0],
                visible=visible_subtasks,
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
                visible=visible_subtasks,
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
        )
    elif practical_subtasks.lower() == "professional groups":
        return (
            practical_subtasks_selected,
            gr.Column(visible=visible_subtasks),
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
            dropdown_create_ui(
                label=f"Potential candidates by professional responsibilities ({len(config_data.Settings_DROPDOWN_CANDIDATES)})",
                info=config_data.InformationMessages_DROPDOWN_CANDIDATES_INFO,
                choices=config_data.Settings_DROPDOWN_CANDIDATES,
                value=config_data.Settings_DROPDOWN_CANDIDATES[0],
                visible=visible_subtasks,
                elem_classes="dropdown-container",
            ),
            number_create_ui(
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
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
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
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
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
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
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
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
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
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
        )
    elif practical_subtasks.lower() == "professional skills":
        return (
            practical_subtasks_selected,
            gr.Column(visible=visible_subtasks),
            dropdown_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(
                value=0.45,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label=config_data.Labels_THRESHOLD_PROFESSIONAL_SKILLS_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            dropdown_create_ui(
                label=f"Professional skills ({len(config_data.Settings_DROPDOWN_PROFESSIONAL_SKILLS)})",
                info=config_data.InformationMessages_DROPDOWN_PROFESSIONAL_SKILLS_INFO,
                choices=config_data.Settings_DROPDOWN_PROFESSIONAL_SKILLS,
                value=config_data.Settings_DROPDOWN_PROFESSIONAL_SKILLS[0],
                visible=visible_subtasks,
                elem_classes="dropdown-container",
            ),
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
        )
    elif (
        practical_subtasks.lower() == "finding a suitable junior colleague"
        or practical_subtasks.lower() == "finding a suitable senior colleague"
        or practical_subtasks.lower()
        == "finding a suitable colleague by personality types"
    ):
        return (
            practical_subtasks_selected,
            gr.Column(visible=visible_subtasks),
            dropdown_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
            dropdown_create_ui(visible=False),
            number_create_ui(
                value=config_data.Values_TARGET_SCORES[0],
                minimum=0.0,
                maximum=1.0,
                step=0.000001,
                label=config_data.Labels_TARGET_SCORE_OPE_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
                value=config_data.Values_TARGET_SCORES[1],
                minimum=0.0,
                maximum=1.0,
                step=0.000001,
                label=config_data.Labels_TARGET_SCORE_CON_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
                value=config_data.Values_TARGET_SCORES[2],
                minimum=0.0,
                maximum=1.0,
                step=0.000001,
                label=config_data.Labels_TARGET_SCORE_EXT_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
                value=config_data.Values_TARGET_SCORES[3],
                minimum=0.0,
                maximum=1.0,
                step=0.000001,
                label=config_data.Labels_TARGET_SCORE_AGR_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
                value=config_data.Values_TARGET_SCORES[4],
                minimum=0.0,
                maximum=1.0,
                step=0.000001,
                label=config_data.Labels_TARGET_SCORE_NNEU_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
                value=0.5,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label=(
                    config_data.Labels_THRESHOLD_TARGET_SCORE_LABEL
                    if practical_subtasks.lower()
                    == "finding a suitable colleague by personality types"
                    else config_data.Labels_EQUAL_COEFFICIENT_LABEL
                ),
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
            dropdown_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
        )
    elif (
        practical_subtasks.lower() == "car characteristics"
        or practical_subtasks.lower() == "mobile device application categories"
        or practical_subtasks.lower() == "clothing styles"
    ):
        if practical_subtasks.lower() == "car characteristics":

            df_correlation_coefficients = read_csv_file(
                config_data.Links_CAR_CHARACTERISTICS,
                ["Trait", "Style and performance", "Safety and practicality"],
            )

        elif practical_subtasks.lower() == "mobile device application categories":

            df_correlation_coefficients = read_csv_file(
                config_data.Links_MDA_CATEGORIES
            )

        elif practical_subtasks.lower() == "clothing styles":
            df_correlation_coefficients = read_csv_file(config_data.Links_CLOTHING_SC)

        return (
            practical_subtasks_selected,
            gr.Column(visible=visible_subtasks),
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
            number_create_ui(
                value=1,
                minimum=1,
                maximum=(
                    df_correlation_coefficients.columns.size
                    if practical_subtasks.lower() == "car characteristics"
                    else df_correlation_coefficients.columns.size - 1
                ),
                step=1,
                label=config_data.Labels_NUMBER_PRIORITY_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(
                    1,
                    (
                        df_correlation_coefficients.columns.size
                        if practical_subtasks.lower() == "car characteristics"
                        else df_correlation_coefficients.columns.size - 1
                    ),
                ),
                show_label=True,
                interactive=True,
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
                value=1,
                minimum=1,
                maximum=5,
                step=1,
                label=config_data.Labels_NUMBER_IMPORTANCE_TRAITS_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(1, 5),
                show_label=True,
                interactive=True,
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
                value=0.55,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label=config_data.Labels_THRESHOLD_CONSUMER_PREFERENCES_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(0, 1.0),
                show_label=True,
                interactive=True,
                visible=visible_subtasks,
                render=True,
                elem_classes="number-container",
            ),
            dropdown_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
        )
    else:
        return (
            practical_subtasks_selected,
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
        )
