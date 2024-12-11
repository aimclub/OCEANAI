"""
File: calculate_practical_tasks.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Event handler for Gradio app to calculate practical tasks.
License: MIT License
"""

from app.oceanai_init import b5
import pandas as pd
import re
import gradio as gr
from pathlib import Path
from bs4 import BeautifulSoup

# Importing necessary components for the Gradio app
from app.config import config_data
from app.video_metadata import video_metadata
from app.mbti_description import MBTI_DESCRIPTION, MBTI_DATA
from app.data_init import df_traits_priority_for_professions
from app.utils import (
    read_csv_file,
    apply_rounding_and_rename_columns,
    preprocess_scores_df,
    get_language_settings,
    extract_profession_weights,
)
from app.components import (
    html_message,
    dataframe,
    files_create_ui,
    video_create_ui,
    textbox_create_ui,
)


def colleague_type(subtask):
    return "minor" if "junior" in subtask.lower() else "major"


def consumer_preferences(subtask):
    return (
        config_data.Filenames_CAR_CHARACTERISTICS
        if "mobile device" in subtask.lower()
        else config_data.Filenames_MDA_CATEGORIES
    )


def remove_parentheses(s):
    return re.sub(r"\s*\([^)]*\)", "", s)


def extract_text_in_parentheses(s):
    result = re.search(r"\(([^)]+)\)", s)
    if result:
        return result.group(1)
    else:
        return None


def compare_strings(original, comparison, prev=False):
    result = []
    prev_class = None

    for orig_char, comp_char in zip(original, comparison):
        curr_class = "true" if orig_char == comp_char else "err"
        if not prev:
            result.append(f"<span class='{curr_class}'>{comp_char}</span>")
        else:
            if curr_class != prev_class:
                result.append("</span>" if prev_class else "")
                result.append(f"<span class='{curr_class}'>")
                prev_class = curr_class
            result.append(comp_char)

    return f"<span class='wrapper_mbti'>{''.join(result + [f'</span>' if prev_class else ''])}</span>"


def create_person_metadata(person_id, files, video_metadata):
    if (
        Path(files[person_id]).name in video_metadata
        and config_data.Settings_SHOW_VIDEO_METADATA
    ):
        person_metadata_list = video_metadata[Path(files[person_id]).name]
        return (
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
        return (
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


def event_handler_calculate_practical_task_blocks(
    language,
    type_modes,
    files,
    video,
    practical_subtasks,
    pt_scores,
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
    number_openness,
    number_conscientiousness,
    number_extraversion,
    number_agreeableness,
    number_non_neuroticism,
):
    lang_id, _ = get_language_settings(language)

    if type_modes == config_data.Settings_TYPE_MODES[1]:
        files = [video]

    if practical_subtasks.lower() == "16 personality types of mbti":
        df_correlation_coefficients = read_csv_file(config_data.Links_MBTI)

        pt_scores_copy = pt_scores.iloc[:, 1:].copy()

        preprocess_scores_df(pt_scores_copy, config_data.Dataframes_PT_SCORES[0][0])

        if type_modes == config_data.Settings_TYPE_MODES[0]:
            b5._professional_match(
                df_files=pt_scores_copy,
                correlation_coefficients=df_correlation_coefficients,
                personality_type=remove_parentheses(dropdown_mbti),
                threshold=threshold_mbti,
                out=False,
            )

            df = apply_rounding_and_rename_columns(b5.df_files_MBTI_job_match_)

            df_hidden = df.drop(
                columns=config_data.Settings_SHORT_PROFESSIONAL_SKILLS
                + config_data.Settings_DROPDOWN_MBTI_DEL_COLS
            )

            df_hidden.rename(
                columns={
                    "Path": "Filename",
                    "MBTI": "Personality Type",
                    "MBTI_Score": "Personality Type Score",
                },
                inplace=True,
            )

            df_copy = df_hidden.copy()
            df_copy["Personality Type"] = df_copy["Personality Type"].apply(
                lambda x: "".join(BeautifulSoup(x, "html.parser").stripped_strings)
            )
            df_copy.to_csv(config_data.Filenames_MBTI_JOB, index=False)

            df_hidden.reset_index(inplace=True)

            person_id = (
                int(df_hidden.iloc[0][config_data.Dataframes_PT_SCORES[0][0]]) - 1
            )

            short_mbti = extract_text_in_parentheses(dropdown_mbti)
            mbti_values = df_hidden["Personality Type"].tolist()

            df_hidden["Personality Type"] = [
                compare_strings(short_mbti, mbti, False) for mbti in mbti_values
            ]

            person_metadata = create_person_metadata(person_id, files, video_metadata)
        elif type_modes == config_data.Settings_TYPE_MODES[1]:
            all_hidden_dfs = []

            for dropdown_mbti in config_data.Settings_DROPDOWN_MBTI:
                b5._professional_match(
                    df_files=pt_scores_copy,
                    correlation_coefficients=df_correlation_coefficients,
                    personality_type=remove_parentheses(dropdown_mbti),
                    threshold=threshold_mbti,
                    out=False,
                )

                df = apply_rounding_and_rename_columns(b5.df_files_MBTI_job_match_)

                df_hidden = df.drop(
                    columns=config_data.Settings_SHORT_PROFESSIONAL_SKILLS
                    + config_data.Settings_DROPDOWN_MBTI_DEL_COLS
                    + config_data.Settings_DROPDOWN_MBTI_DEL_COLS_WEBCAM
                )

                df_hidden.insert(0, "Popular Occupations", dropdown_mbti)

                df_hidden.rename(
                    columns={
                        "MBTI": "Personality Type",
                        "MBTI_Score": "Personality Type Score",
                    },
                    inplace=True,
                )

                short_mbti = extract_text_in_parentheses(dropdown_mbti)
                mbti_values = df_hidden["Personality Type"].tolist()

                df_hidden["Personality Type"] = [
                    compare_strings(short_mbti, mbti, False) for mbti in mbti_values
                ]

                all_hidden_dfs.append(df_hidden)

            df_hidden = pd.concat(all_hidden_dfs, ignore_index=True)

            df_hidden = df_hidden.sort_values(
                by="Personality Type Score", ascending=False
            )

            df_hidden.reset_index(drop=True, inplace=True)

            df_copy = df_hidden.copy()
            df_copy["Personality Type"] = df_copy["Personality Type"].apply(
                lambda x: "".join(BeautifulSoup(x, "html.parser").stripped_strings)
            )
            df_copy.to_csv(config_data.Filenames_MBTI_JOB, index=False)

            person_id = 0

            person_metadata = create_person_metadata(person_id, files, video_metadata)

        existing_tuple = (
            gr.Row(visible=True),
            gr.Column(visible=True),
            dataframe(
                headers=df_hidden.columns.tolist(),
                values=df_hidden.values.tolist(),
                visible=True,
            ),
            files_create_ui(
                config_data.Filenames_MBTI_JOB,
                "single",
                [".csv"],
                config_data.OtherMessages_EXPORT_MBTI,
                True,
                False,
                True,
                "csv-container",
            ),
            gr.Accordion(
                label=config_data.Labels_NOTE_MBTI_LABEL,
                open=False,
                visible=True,
            ),
            gr.HTML(value=MBTI_DESCRIPTION, visible=True),
            dataframe(
                headers=MBTI_DATA.columns.tolist(),
                values=MBTI_DATA.values.tolist(),
                visible=True,
                elem_classes="mbti-dataframe",
            ),
            gr.Column(visible=True),
            video_create_ui(
                value=files[person_id],
                file_name=Path(files[person_id]).name,
                label="Best Person ID - " + str(person_id + 1),
                visible=True,
                elem_classes="video-sorted-container",
            ),
            html_message(config_data.InformationMessages_NOTI_IN_DEV, False, False),
        )

        return existing_tuple[:-1] + person_metadata + existing_tuple[-1:]
    elif practical_subtasks.lower() == "professional groups":
        if type_modes == config_data.Settings_TYPE_MODES[0]:
            sum_weights = sum(
                [
                    number_openness,
                    number_conscientiousness,
                    number_extraversion,
                    number_agreeableness,
                    number_non_neuroticism,
                ]
            )

            if sum_weights != 100:
                gr.Warning(
                    config_data.InformationMessages_SUM_WEIGHTS.format(sum_weights)
                )

                return (
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
                    html_message(
                        config_data.InformationMessages_SUM_WEIGHTS.format(sum_weights),
                        False,
                        True,
                    ),
                )
            else:
                b5._candidate_ranking(
                    df_files=pt_scores.iloc[:, 1:],
                    weigths_openness=number_openness,
                    weigths_conscientiousness=number_conscientiousness,
                    weigths_extraversion=number_extraversion,
                    weigths_agreeableness=number_agreeableness,
                    weigths_non_neuroticism=number_non_neuroticism,
                    out=False,
                )

                df = apply_rounding_and_rename_columns(b5.df_files_ranking_)

                df_hidden = df.drop(
                    columns=config_data.Settings_SHORT_PROFESSIONAL_SKILLS
                )

                df_hidden.to_csv(config_data.Filenames_POTENTIAL_CANDIDATES)

                df_hidden.reset_index(inplace=True)

                person_id = (
                    int(df_hidden.iloc[0][config_data.Dataframes_PT_SCORES[0][0]]) - 1
                )

                person_metadata = create_person_metadata(
                    person_id, files, video_metadata
                )
        elif type_modes == config_data.Settings_TYPE_MODES[1]:
            all_hidden_dfs = []

            for dropdown_candidate in config_data.Settings_DROPDOWN_CANDIDATES[:-1]:
                weights, _ = extract_profession_weights(
                    df_traits_priority_for_professions,
                    dropdown_candidate,
                )

                sum_weights = sum(weights)

                if sum_weights != 100:
                    gr.Warning(
                        config_data.InformationMessages_SUM_WEIGHTS.format(sum_weights)
                    )

                    return (
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
                        html_message(
                            config_data.InformationMessages_SUM_WEIGHTS.format(
                                sum_weights
                            ),
                            False,
                            True,
                        ),
                    )
                else:
                    b5._candidate_ranking(
                        df_files=pt_scores.iloc[:, 1:],
                        weigths_openness=weights[0],
                        weigths_conscientiousness=weights[1],
                        weigths_extraversion=weights[2],
                        weigths_agreeableness=weights[3],
                        weigths_non_neuroticism=weights[4],
                        out=False,
                    )

                    df = apply_rounding_and_rename_columns(b5.df_files_ranking_)

                    df_hidden = df.drop(
                        columns=config_data.Settings_SHORT_PROFESSIONAL_SKILLS
                        + config_data.Settings_DROPDOWN_MBTI_DEL_COLS_WEBCAM
                    )

                    df_hidden.insert(0, "Professional Group", dropdown_candidate)

                    all_hidden_dfs.append(df_hidden)

                df_hidden = pd.concat(all_hidden_dfs, ignore_index=True)

                df_hidden.rename(
                    columns={
                        "Candidate score": "Summary Score",
                    },
                    inplace=True,
                )

                df_hidden = df_hidden.sort_values(by="Summary Score", ascending=False)

                df_hidden.reset_index(drop=True, inplace=True)

                df_hidden.to_csv(
                    config_data.Filenames_POTENTIAL_CANDIDATES, index=False
                )

                person_id = 0

                person_metadata = create_person_metadata(
                    person_id, files, video_metadata
                )

        existing_tuple = (
            gr.Row(visible=True),
            gr.Column(visible=True),
            dataframe(
                headers=df_hidden.columns.tolist(),
                values=df_hidden.values.tolist(),
                visible=True,
            ),
            files_create_ui(
                config_data.Filenames_POTENTIAL_CANDIDATES,
                "single",
                [".csv"],
                config_data.OtherMessages_EXPORT_PG,
                True,
                False,
                True,
                "csv-container",
            ),
            gr.Accordion(visible=False),
            gr.HTML(visible=False),
            dataframe(visible=False),
            gr.Column(visible=True),
            video_create_ui(
                value=files[person_id],
                file_name=Path(files[person_id]).name,
                label="Best Person ID - " + str(person_id + 1),
                visible=True,
                elem_classes="video-sorted-container",
            ),
            html_message(config_data.InformationMessages_NOTI_IN_DEV, False, False),
        )

        return existing_tuple[:-1] + person_metadata + existing_tuple[-1:]
    elif practical_subtasks.lower() == "professional skills":
        df_professional_skills = read_csv_file(config_data.Links_PROFESSIONAL_SKILLS)

        pt_scores_copy = pt_scores.iloc[:, 1:].copy()

        preprocess_scores_df(pt_scores_copy, config_data.Dataframes_PT_SCORES[0][0])

        b5._priority_skill_calculation(
            df_files=pt_scores_copy,
            correlation_coefficients=df_professional_skills,
            threshold=threshold_professional_skills,
            out=False,
        )

        df = apply_rounding_and_rename_columns(b5.df_files_priority_skill_)

        if type_modes == config_data.Settings_TYPE_MODES[0]:
            professional_skills_list = (
                config_data.Settings_DROPDOWN_PROFESSIONAL_SKILLS.copy()
            )

            professional_skills_list.remove(dropdown_professional_skills)

            del_cols = []
        elif type_modes == config_data.Settings_TYPE_MODES[1]:
            professional_skills_list = []
            del_cols = config_data.Settings_DROPDOWN_MBTI_DEL_COLS_WEBCAM

        df_hidden = df.drop(
            columns=config_data.Settings_SHORT_PROFESSIONAL_SKILLS
            + professional_skills_list
            + del_cols
        )

        if type_modes == config_data.Settings_TYPE_MODES[0]:
            df_hidden = df_hidden.sort_values(
                by=[dropdown_professional_skills], ascending=False
            )
            df_hidden.reset_index(inplace=True)
        elif type_modes == config_data.Settings_TYPE_MODES[1]:
            df_hidden = df_hidden.melt(
                var_name="Professional Skill", value_name="Summary Score"
            )
            df_hidden = df_hidden.sort_values(by=["Summary Score"], ascending=False)
            df_hidden.reset_index(drop=True, inplace=True)

        df_hidden.to_csv(config_data.Filenames_PT_SKILLS_SCORES)

        if type_modes == config_data.Settings_TYPE_MODES[0]:
            person_id = (
                int(df_hidden.iloc[0][config_data.Dataframes_PT_SCORES[0][0]]) - 1
            )
        elif type_modes == config_data.Settings_TYPE_MODES[1]:
            person_id = 0

        person_metadata = create_person_metadata(person_id, files, video_metadata)

        existing_tuple = (
            gr.Row(visible=True),
            gr.Column(visible=True),
            dataframe(
                headers=df_hidden.columns.tolist(),
                values=df_hidden.values.tolist(),
                visible=True,
            ),
            files_create_ui(
                config_data.Filenames_PT_SKILLS_SCORES,
                "single",
                [".csv"],
                config_data.OtherMessages_EXPORT_PS,
                True,
                False,
                True,
                "csv-container",
            ),
            gr.Accordion(visible=False),
            gr.HTML(visible=False),
            dataframe(visible=False),
            gr.Column(visible=True),
            video_create_ui(
                value=files[person_id],
                file_name=Path(files[person_id]).name,
                label="Best Person ID - " + str(person_id + 1),
                visible=True,
                elem_classes="video-sorted-container",
            ),
            html_message(config_data.InformationMessages_NOTI_IN_DEV, False, False),
        )

        return existing_tuple[:-1] + person_metadata + existing_tuple[-1:]
    elif (
        practical_subtasks.lower() == "finding a suitable junior colleague"
        or practical_subtasks.lower() == "finding a suitable senior colleague"
        or practical_subtasks.lower()
        == "finding a suitable colleague by personality types"
    ):
        pt_scores_copy = pt_scores.iloc[:, 1:].copy()

        preprocess_scores_df(pt_scores_copy, config_data.Dataframes_PT_SCORES[0][0])

        if (
            practical_subtasks.lower()
            != "finding a suitable colleague by personality types"
        ):
            df_correlation_coefficients = read_csv_file(
                config_data.Links_FINDING_COLLEAGUE, ["ID"]
            )

            b5._colleague_ranking(
                df_files=pt_scores_copy,
                correlation_coefficients=df_correlation_coefficients,
                target_scores=[
                    target_score_ope,
                    target_score_con,
                    target_score_ext,
                    target_score_agr,
                    target_score_nneu,
                ],
                colleague=colleague_type(practical_subtasks),
                equal_coefficients=equal_coefficient,
                out=False,
            )
            df = apply_rounding_and_rename_columns(b5.df_files_colleague_)

            df_hidden = df.drop(columns=config_data.Settings_SHORT_PROFESSIONAL_SKILLS)

            df_hidden.to_csv(
                colleague_type(practical_subtasks)
                + config_data.Filenames_COLLEAGUE_RANKING
            )
        else:
            b5._colleague_personality_type_match(
                df_files=pt_scores_copy,
                correlation_coefficients=None,
                target_scores=[
                    target_score_ope,
                    target_score_con,
                    target_score_ext,
                    target_score_agr,
                    target_score_nneu,
                ],
                threshold=equal_coefficient,
                out=False,
            )
            df = b5.df_files_MBTI_colleague_match_.rename(
                columns={
                    "MBTI": "Personality Type",
                    "MBTI_Score": "Personality Type Score",
                }
            )

            df_hidden = df[["Path", "Personality Type", "Match"]]

            df_hidden.to_csv(config_data.Filenames_COLLEAGUE_RANKING)

        df_hidden.reset_index(inplace=True)

        person_id = (
            int(
                df_hidden.iloc[
                    (
                        0
                        if practical_subtasks.lower()
                        != "finding a suitable colleague by personality types"
                        else 1
                    )
                ][config_data.Dataframes_PT_SCORES[0][0]]
            )
            - 1
        )

        person_metadata = create_person_metadata(person_id, files, video_metadata)

        existing_tuple = (
            gr.Row(visible=True),
            gr.Column(visible=True),
            dataframe(
                headers=df_hidden.columns.tolist(),
                values=df_hidden.values.tolist(),
                visible=True,
            ),
            files_create_ui(
                colleague_type(practical_subtasks)
                + config_data.Filenames_COLLEAGUE_RANKING,
                "single",
                [".csv"],
                config_data.OtherMessages_EXPORT_WT,
                True,
                False,
                True,
                "csv-container",
            ),
            gr.Accordion(visible=False),
            gr.HTML(visible=False),
            dataframe(visible=False),
            gr.Column(visible=True),
            video_create_ui(
                value=files[person_id],
                file_name=Path(files[person_id]).name,
                label="Best Person ID - " + str(person_id + 1),
                visible=True,
                elem_classes="video-sorted-container",
            ),
            html_message(config_data.InformationMessages_NOTI_IN_DEV, False, False),
        )

        return existing_tuple[:-1] + person_metadata + existing_tuple[-1:]
    elif (
        practical_subtasks.lower() == "car characteristics"
        or practical_subtasks.lower() == "mobile device application categories"
        or practical_subtasks.lower() == "clothing styles"
    ):
        if practical_subtasks.lower() == "car characteristics":
            df_correlation_coefficients = read_csv_file(
                config_data.Links_CAR_CHARACTERISTICS,
                ["Style and performance", "Safety and practicality"],
            )
        elif practical_subtasks.lower() == "mobile device application categories":
            df_correlation_coefficients = read_csv_file(
                config_data.Links_MDA_CATEGORIES
            )
        elif practical_subtasks.lower() == "clothing styles":
            df_correlation_coefficients = read_csv_file(config_data.Links_CLOTHING_SC)

        if type_modes == config_data.Settings_TYPE_MODES[1]:
            number_priority = df_correlation_coefficients.columns.size - 1
            number_importance_traits = 5

        pt_scores_copy = pt_scores.iloc[:, 1:].copy()

        preprocess_scores_df(pt_scores_copy, config_data.Dataframes_PT_SCORES[0][0])

        b5._priority_calculation(
            df_files=pt_scores_copy,
            correlation_coefficients=df_correlation_coefficients,
            col_name_ocean="Trait",
            threshold=threshold_consumer_preferences,
            number_priority=number_priority,
            number_importance_traits=number_importance_traits,
            out=False,
        )

        df_files_priority = b5.df_files_priority_.copy()
        df_files_priority.reset_index(inplace=True)

        df = apply_rounding_and_rename_columns(df_files_priority.iloc[:, 1:])

        preprocess_scores_df(df, config_data.Dataframes_PT_SCORES[0][0])

        if type_modes == config_data.Settings_TYPE_MODES[0]:
            del_cols = []
        elif type_modes == config_data.Settings_TYPE_MODES[1]:
            del_cols = config_data.Settings_DROPDOWN_MBTI_DEL_COLS_WEBCAM

        df_hidden = df.drop(
            columns=config_data.Settings_SHORT_PROFESSIONAL_SKILLS + del_cols
        )

        if type_modes == config_data.Settings_TYPE_MODES[1]:
            df_hidden = df_hidden.T

            df_hidden = df_hidden.head(-number_importance_traits)

            df_hidden = df_hidden.reset_index()

            df_hidden.columns = ["Priority", "Category"]

        df_hidden.to_csv(consumer_preferences(practical_subtasks))

        df_hidden.reset_index(
            drop=True if type_modes == config_data.Settings_TYPE_MODES[1] else False,
            inplace=True,
        )

        if type_modes == config_data.Settings_TYPE_MODES[0]:
            person_id = (
                int(df_hidden.iloc[0][config_data.Dataframes_PT_SCORES[0][0]]) - 1
            )
        elif type_modes == config_data.Settings_TYPE_MODES[1]:
            person_id = 0

        person_metadata = create_person_metadata(person_id, files, video_metadata)

        existing_tuple = (
            gr.Row(visible=True),
            gr.Column(visible=True),
            dataframe(
                headers=df_hidden.columns.tolist(),
                values=df_hidden.values.tolist(),
                visible=True,
            ),
            files_create_ui(
                consumer_preferences(practical_subtasks),
                "single",
                [".csv"],
                config_data.OtherMessages_EXPORT_CP,
                True,
                False,
                True,
                "csv-container",
            ),
            gr.Accordion(visible=False),
            gr.HTML(visible=False),
            dataframe(visible=False),
            gr.Column(visible=True),
            video_create_ui(
                value=files[person_id],
                file_name=Path(files[person_id]).name,
                label="Best Person ID - " + str(person_id + 1),
                visible=True,
                elem_classes="video-sorted-container",
            ),
            html_message(config_data.InformationMessages_NOTI_IN_DEV, False, False),
        )

        return existing_tuple[:-1] + person_metadata + existing_tuple[-1:]
    else:
        gr.Info(config_data.InformationMessages_NOTI_IN_DEV)

        return (
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
            html_message(config_data.InformationMessages_NOTI_IN_DEV, False, True),
        )
