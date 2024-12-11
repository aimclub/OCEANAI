"""
File: practical_tasks.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Event handler for Gradio app to filter practical tasks based on selected practical tasks.
License: MIT License
"""

# Importing necessary components for the Gradio app
from app.config import config_data
from app.practical_tasks import supported_practical_tasks
from app.components import radio_create_ui
from app.utils import get_language_settings


def event_handler_practical_tasks(
    language, type_modes, practical_tasks, practical_subtasks_selected
):
    lang_id, _ = get_language_settings(language)

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

    return radio_create_ui(
        practical_subtasks_selected[practical_tasks],
        config_data.Labels_PRACTICAL_SUBTASKS_LABEL[lang_id],
        supported_practical_tasks_ren[lang_id][practical_tasks],
        config_data.InformationMessages_PRACTICAL_SUBTASKS_INFO[lang_id],
        True,
        True,
    )
