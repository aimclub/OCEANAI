"""
File: mbti_description.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Personality dimensions description.
License: MIT License
"""

import pandas as pd

# Importing necessary components for the Gradio app

MBTI_DATA = pd.DataFrame(
    {
        "Dimension description": [
            "How we interact with the world and where we direct our energy",
            "The kind of information we naturally notice",
            "How we make decisions",
            "Whether we prefer to live in a more structured way (making decisions) or in a more spontaneous way (taking in information)",
        ],
        "Dimension": [
            "(E) Extraversion - Introversion (I)",
            "(S) Sensing - Intuition (N)",
            "(T) Thinking - Feeling (F)",
            "(J) Judging  - Perceiving (P)",
        ],
    }
)

MBTI_DESCRIPTION = (
    "<h4>Personality types of MBTI are based on four Personality Dimensions</h4>"
)
