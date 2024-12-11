"""
File: description.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Project description for the Gradio app.
License: MIT License
"""

# Importing necessary components for the Gradio app
from app.config import config_data

TEMPLATE = """\
<h1><a href="https://github.com/aimclub/OCEANAI" target="_blank">OCEAN-AI</a> {description}.</h1>

<div class="app-flex-container">
    <img src="https://img.shields.io/badge/version-v{version}-rc0" alt="{version_label}">
    <a href='https://github.com/DmitryRyumin/OCEANAI' target='_blank'><img src='https://img.shields.io/github/stars/DmitryRyumin/OCEANAI?style=flat' alt='GitHub' /></a>
</div>

The models used in OCEAN-AI were trained on 15-second clips from the ChaLearn First Impression v2 dataset. 
For more reliable predictions, 15-second videos are recommended, but OCEAN-AI can analyze videos of any length. 
Due to limited computational resources on HuggingFace, we provide six 3-second videos as examples.
"""

DESCRIPTIONS = [
    TEMPLATE.format(
        description=config_data.InformationMessages_DESCRIPTIONS[0],
        version=config_data.AppSettings_APP_VERSION,
        version_label=config_data.Labels_APP_VERSION[0],
    ),
    TEMPLATE.format(
        description=config_data.InformationMessages_DESCRIPTIONS[1],
        version=config_data.AppSettings_APP_VERSION,
        version_label=config_data.Labels_APP_VERSION[1],
    ),
]
