"""
File: data_init.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Data initialization.
License: MIT License
"""

from app.config import config_data
from app.utils import read_csv_file, extract_profession_weights


df_traits_priority_for_professions = read_csv_file(config_data.Links_PROFESSIONS)
weights_professions, interactive_professions = extract_profession_weights(
    df_traits_priority_for_professions,
    config_data.Settings_DROPDOWN_CANDIDATES[0],
)
