"""
File: video_metadata.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Utility functions for working with video metadata.
License: MIT License
"""

import yaml
from typing import List, Dict

# Importing necessary components for the Gradio app


def load_video_metadata(file_path: str) -> Dict[str, List]:
    with open(file_path, "r") as file:
        video_metadata = yaml.safe_load(file) or {}
        result = {}
        for key, value in video_metadata.get("video_metadata", {}).items():
            alias = key.split("_")[0]
            result[key] = value + [f"video{alias}"]
        return result


yaml_file_path = "./video_metadata.yaml"
video_metadata = load_video_metadata(yaml_file_path)
