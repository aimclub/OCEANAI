"""
File: utils.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Utility functions.
License: MIT License
"""

import pandas as pd
import subprocess
from pathlib import Path

# Importing necessary components for the Gradio app
from app.config import config_data


def get_language_settings(language):
    language_mappings = {
        "english": (0, config_data.Settings_LANGUAGES_EN),
        "английский": (0, config_data.Settings_LANGUAGES_EN),
        "russian": (1, config_data.Settings_LANGUAGES_RU),
        "русский": (1, config_data.Settings_LANGUAGES_RU),
    }

    normalized_language = language.lower()

    lang_id, choices = language_mappings.get(
        normalized_language, (0, config_data.Settings_LANGUAGES_EN)
    )

    return lang_id, choices


def preprocess_scores_df(df, name):
    df.index.name = name
    df.index += 1
    df.index = df.index.map(str)

    return df


def read_csv_file(file_path, drop_columns=[]):
    df = pd.read_csv(file_path)

    if len(drop_columns) != 0:
        df = pd.DataFrame(df.drop(drop_columns, axis=1))

    return preprocess_scores_df(df, "ID")


def round_numeric_values(x):
    if isinstance(x, (int, float)):
        return round(x, 4)

    return x


def apply_rounding_and_rename_columns(df):
    df_rounded = df.rename(
        columns={
            "Openness": "OPE",
            "Conscientiousness": "CON",
            "Extraversion": "EXT",
            "Agreeableness": "AGR",
            "Non-Neuroticism": "NNEU",
        }
    )

    columns_to_round = df_rounded.columns[1:]
    df_rounded[columns_to_round] = df_rounded[columns_to_round].applymap(
        round_numeric_values
    )

    return df_rounded


def extract_profession_weights(df, dropdown_candidates):
    try:
        weights_professions = df.loc[df["Profession"] == dropdown_candidates, :].values[
            0
        ][1:]
        interactive_professions = False
    except Exception:
        weights_professions = [0] * 5
        interactive_professions = True
    else:
        weights_professions = list(map(int, weights_professions))

    return weights_professions, interactive_professions


def webm2mp4(input_file):
    input_path = Path(input_file)
    output_path = input_path.with_suffix(".mp4")

    if not output_path.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(input_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-strict",
                "experimental",
                str(output_path),
            ],
            check=True,
        )

    return str(output_path)
