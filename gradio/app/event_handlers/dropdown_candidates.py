"""
File: dropdown_candidates.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Event handler for Gradio app to filter dropdown candidates based on selected dropdown candidates.
License: MIT License
"""

# Importing necessary components for the Gradio app
from app.config import config_data
from app.utils import extract_profession_weights
from app.data_init import df_traits_priority_for_professions
from app.components import number_create_ui


def event_handler_dropdown_candidates(practical_subtasks, dropdown_candidates):
    if practical_subtasks.lower() == "professional groups":
        weights, interactive = extract_profession_weights(
            df_traits_priority_for_professions,
            dropdown_candidates,
        )

        return (
            number_create_ui(
                value=weights[0],
                minimum=config_data.Values_0_100[0],
                maximum=config_data.Values_0_100[1],
                step=1,
                label=config_data.Labels_NUMBER_IMPORTANCE_OPE_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(
                    config_data.Values_0_100[0], config_data.Values_0_100[1]
                ),
                show_label=True,
                interactive=interactive,
                visible=True,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
                value=weights[1],
                minimum=config_data.Values_0_100[0],
                maximum=config_data.Values_0_100[1],
                step=1,
                label=config_data.Labels_NUMBER_IMPORTANCE_CON_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(
                    config_data.Values_0_100[0], config_data.Values_0_100[1]
                ),
                show_label=True,
                interactive=interactive,
                visible=True,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
                value=weights[2],
                minimum=config_data.Values_0_100[0],
                maximum=config_data.Values_0_100[1],
                step=1,
                label=config_data.Labels_NUMBER_IMPORTANCE_EXT_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(
                    config_data.Values_0_100[0], config_data.Values_0_100[1]
                ),
                show_label=True,
                interactive=interactive,
                visible=True,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
                value=weights[3],
                minimum=config_data.Values_0_100[0],
                maximum=config_data.Values_0_100[1],
                step=1,
                label=config_data.Labels_NUMBER_IMPORTANCE_AGR_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(
                    config_data.Values_0_100[0], config_data.Values_0_100[1]
                ),
                show_label=True,
                interactive=interactive,
                visible=True,
                render=True,
                elem_classes="number-container",
            ),
            number_create_ui(
                value=weights[4],
                minimum=config_data.Values_0_100[0],
                maximum=config_data.Values_0_100[1],
                step=1,
                label=config_data.Labels_NUMBER_IMPORTANCE_NNEU_LABEL,
                info=config_data.InformationMessages_VALUE_FROM_TO_INFO.format(
                    config_data.Values_0_100[0], config_data.Values_0_100[1]
                ),
                show_label=True,
                interactive=interactive,
                visible=True,
                render=True,
                elem_classes="number-container",
            ),
        )
    else:
        return (
            number_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
            number_create_ui(visible=False),
        )
