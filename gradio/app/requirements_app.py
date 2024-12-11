"""
File: requirements_app.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Project requirements for the Gradio app.
License: MIT License
"""

import pandas as pd

# Importing necessary components for the Gradio app


def read_requirements_to_df(file_path="requirements.txt"):
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []

    pypi = (
        lambda x: f"<a href='https://pypi.org/project/{x}' target='_blank'><img src='https://img.shields.io/pypi/v/{x}' alt='PyPI' /></a>"
    )

    for line in lines:
        line = line.strip()
        if "==" in line:
            library, version = line.split("==")
            data.append(
                {
                    "Library": library,
                    "Recommended Version": version,
                    "Current Version": pypi(library),
                }
            )

    df = pd.DataFrame(data)

    return df
