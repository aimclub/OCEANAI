"""
File: app.py
Authors: Elena Ryumina and Dmitry Ryumin
Description: OCEANAI App for gradio.
License: MIT License
"""

import gradio as gr

# Importing necessary components for the Gradio app
from app.config import CONFIG_NAME, config_data, load_tab_creators
from app.event_handlers.event_handlers import setup_app_event_handlers
from app import tabs
from app.components import dropdown_create_ui
from app.port import is_port_in_use, free_ports

gr.set_static_paths(paths=[config_data.StaticPaths_IMAGES])


def create_gradio_app() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Default(), css_paths=config_data.AppSettings_CSS_PATH
    ) as gradio_app:
        with gr.Column(
            visible=True,
            render=True,
            variant="default",
            elem_classes="languages-container_wrapper",
        ):
            with gr.Row(
                visible=True,
                render=True,
                variant="default",
                elem_classes="languages-container",
            ) as languages_row:
                country_flags = gr.Image(
                    value=config_data.StaticPaths_IMAGES
                    + config_data.Images_LANGUAGES[0],
                    container=False,
                    interactive=False,
                    show_label=False,
                    visible=True,
                    show_download_button=False,
                    elem_classes="country_flags",
                    show_fullscreen_button=False,
                )

                languages = dropdown_create_ui(
                    label=None,
                    info=None,
                    choices=config_data.Settings_LANGUAGES_EN,
                    value=config_data.Settings_LANGUAGES_EN[0],
                    visible=True,
                    show_label=False,
                    elem_classes="dropdown-language-container",
                    interactive=True,
                )

        tab_results = {}
        ts = []

        available_functions = {
            attr: getattr(tabs, attr)
            for attr in dir(tabs)
            if callable(getattr(tabs, attr)) and attr.endswith("_tab")
        }

        tab_creators = load_tab_creators(CONFIG_NAME, available_functions)

        for tab_name, create_tab_function in tab_creators.items():
            with gr.Tab(tab_name) as tab:
                app_instance = create_tab_function()
                tab_results[tab_name] = app_instance
                ts.append(tab)

        setup_app_event_handlers(
            *tab_results[list(tab_results.keys())[0]],
            *ts,
            languages_row,
            country_flags,
            languages
        )

    return gradio_app


if __name__ == "__main__":
    ports_to_check = [config_data.AppSettings_PORT]

    for port in filter(None, ports_to_check):
        if is_port_in_use(config_data.AppSettings_SERVER_NAME, port):
            free_ports(port)

    create_gradio_app().queue(api_open=False).launch(
        share=False,
        server_name=config_data.AppSettings_SERVER_NAME,
        server_port=config_data.AppSettings_PORT,
    )
