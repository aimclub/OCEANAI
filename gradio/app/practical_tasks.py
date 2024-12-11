"""
File: practical_tasks.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Utility functions for working with practical tasks data.
License: MIT License
"""

import yaml
from typing import Dict, List

# Importing necessary components for the Gradio app


def load_practical_tasks_data(file_paths: List[str]) -> List[Dict]:
    all_tasks_data = []

    for file_path in file_paths:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file) or []
            all_tasks_data.append(data)

    return all_tasks_data


def transform_practical_tasks_data(data: List[Dict]) -> List[Dict]:
    output_list = []

    for task_data in data:
        output_dict = {item["task"]: item["subtasks"] for item in task_data}
        output_list.append(output_dict)

    return output_list


yaml_file_paths = ["./practical_tasks_en.yaml", "./practical_tasks_ru.yaml"]
practical_tasks_data = load_practical_tasks_data(yaml_file_paths)
supported_practical_tasks = transform_practical_tasks_data(practical_tasks_data)
