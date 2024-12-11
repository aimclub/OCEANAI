"""
File: components.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Utility functions for creating Gradio components.
License: MIT License
"""

import gradio as gr
from typing import Union, List, Callable, Optional, Literal

# Importing necessary components for the Gradio app
from app.config import config_data


def html_message(
    message: str = "",
    error: bool = True,
    visible: bool = True,
    elem_classes: Optional[str] = "html-container",
) -> gr.HTML:
    css_class = "noti_err" if not error else "noti_true"

    return gr.HTML(
        value=f"<h3 class='{css_class}'>{message}</h3>",
        visible=visible,
        elem_classes=elem_classes,
    )


def files_create_ui(
    value: Union[str, List[str], Callable, None] = None,
    file_count: str = "multiple",
    file_types: List = ["video"],
    label: str = config_data.OtherMessages_VIDEO_FILES[
        config_data.AppSettings_DEFAULT_LANG_ID
    ],
    show_label: bool = True,
    interactive: bool = True,
    visible: bool = True,
    elem_classes: Optional[str] = "files-container",
) -> gr.File:
    return gr.File(
        value=value,
        file_count=file_count,
        file_types=file_types,
        label=label,
        show_label=show_label,
        interactive=interactive,
        visible=visible,
        elem_classes=elem_classes,
    )


def video_create_ui(
    value: Optional[str] = None,
    label: str = config_data.OtherMessages_VIDEO_PLAYER[
        config_data.AppSettings_DEFAULT_LANG_ID
    ],
    file_name: Optional[str] = None,
    show_label: bool = True,
    interactive: bool = False,
    visible: bool = True,
    elem_classes: Optional[str] = "video-container",
) -> gr.Video:
    if file_name is not None:
        label += f" ({file_name})"

    return gr.Video(
        value=value,
        label=label,
        show_label=show_label,
        interactive=interactive,
        visible=visible,
        elem_classes=elem_classes,
    )


def dataframe(
    headers: Optional[List] = None,
    values: Optional[List] = None,
    height: int = 500,
    wrap: bool = True,
    visible: bool = True,
    interactive: bool = False,
    elem_classes: Optional[str] = "dataframe",
) -> gr.Dataframe:
    if headers is None or values is None:
        datatype = "str"
    else:
        datatype = ["markdown"] * len(headers)

    return gr.Dataframe(
        value=values,
        headers=headers,
        datatype=datatype,
        max_height=height,
        wrap=wrap,
        visible=visible,
        interactive=interactive,
        elem_classes=elem_classes,
    )


def button(
    value: str = "",
    interactive: bool = True,
    scale: int = 3,
    icon: Optional[str] = None,
    visible: bool = True,
    elem_classes: Optional[str] = None,
) -> gr.Button:
    return gr.Button(
        value=value,
        interactive=interactive,
        scale=scale,
        icon=icon,
        visible=visible,
        elem_classes=elem_classes,
    )


def radio_create_ui(
    value: Union[str, int, float, Callable, None],
    label: str,
    choices: Union[List, None],
    info: str,
    interactive: bool,
    visible: bool,
):
    return gr.Radio(
        value=value,
        label=label,
        choices=choices,
        info=info,
        show_label=True,
        container=True,
        interactive=interactive,
        visible=visible,
    )


def number_create_ui(
    value: float = 0.5,
    minimum: float = 0.0,
    maximum: float = 1.0,
    step: float = 0.01,
    label: Optional[str] = None,
    info: Optional[str] = None,
    show_label: bool = True,
    interactive: bool = True,
    visible: bool = False,
    render: bool = True,
    elem_classes: Optional[str] = "number-container",
):
    return gr.Number(
        value=value,
        minimum=minimum,
        maximum=maximum,
        step=step,
        label=label,
        info=info,
        show_label=show_label,
        interactive=interactive,
        visible=visible,
        render=render,
        elem_classes=elem_classes,
    )


def dropdown_create_ui(
    label: Optional[str] = None,
    info: Optional[str] = None,
    choices: Optional[List[str]] = None,
    value: Optional[List[str]] = None,
    multiselect: bool = False,
    show_label: bool = True,
    interactive: bool = True,
    visible: bool = True,
    render: bool = True,
    elem_classes: Optional[str] = None,
) -> gr.Dropdown:
    return gr.Dropdown(
        choices=choices,
        value=value,
        multiselect=multiselect,
        label=label,
        info=info,
        show_label=show_label,
        interactive=interactive,
        visible=visible,
        render=render,
        elem_classes=elem_classes,
    )


def textbox_create_ui(
    value: Optional[str] = None,
    type: Literal["text", "password", "email"] = "text",
    label: Optional[str] = None,
    placeholder: Optional[str] = None,
    info: Optional[str] = None,
    max_lines: int = 1,
    show_label: bool = True,
    interactive: bool = True,
    visible: bool = True,
    show_copy_button: bool = True,
    scale: int = 1,
    container: bool = False,
):
    return gr.Textbox(
        value=value,
        type=type,
        label=label,
        placeholder=placeholder,
        info=info,
        max_lines=max_lines,
        show_label=show_label,
        interactive=interactive,
        visible=visible,
        show_copy_button=show_copy_button,
        scale=scale,
        container=container,
    )
