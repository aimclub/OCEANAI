"""
File: app.py
Author: Elena Ryumina and Dmitry Ryumin
Description: About the app.
License: MIT License
"""

APP = """
<div class="about_app">
    <div style="max-width: 90%; margin: auto; padding: 20px;">
        <p style="text-align: center;">
            <img src="https://raw.githubusercontent.com/aimclub/OCEANAI/main/docs/source/_static/logo.svg" alt="Logo" style="width: 20%; height: auto; display: block; margin: auto;">
        </p>

        <blockquote>
            <a href="https://oceanai.readthedocs.io/en/latest/">OCEAN-AI</a> is an open-source library consisting of a set of algorithms for intellectual analysis of human behavior based on multimodal data for automatic personality traits (PT) assessment. The library evaluates five PT: <strong>O</strong>penness to experience, <strong>C</strong>onscientiousness, <strong>E</strong>xtraversion, <strong>A</strong>greeableness, Non-<strong>N</strong>euroticism.
        </blockquote>

        <p style="text-align: center;">
            <img src="https://raw.githubusercontent.com/aimclub/OCEANAI/main/docs/source/_static/Pipeline_OCEANAI.en.svg" alt="Pipeline" style="max-width: 60%; height: auto; display: block; margin: auto;">
        </p>

        <hr>

        <h2>OCEAN-AI includes three main algorithms:</h2>
        <ol>
            <li>Audio Information Analysis Algorithm (AIA).</li>
            <li>Video Information Analysis Algorithm (VIA).</li>
            <li>Text Information Analysis Algorithm (TIA).</li>
            <li>Multimodal Information Fusion Algorithm (MIF).</li>
        </ol>

        <p>The AIA, VIA and TIA algorithms implement the functions of strong artificial intelligence (AI) in terms of complexing acoustic, visual and linguistic features built on different principles (hand-crafted and deep features), i.e. these algorithms implement the approaches of composite (hybrid) AI. The necessary pre-processing of audio, video and text information, the calculation of visual, acoustic and linguistic features and the output of predictions of personality traits based on them are carried out in the algorithms.</p>

        <p>The MIF algorithm is a combination of three information analysis algorithms (AIA, VIA and TIA). This algorithm performs feature-level fusion obtained by the AIA, VIA and TIA algorithms.</p>

        <p>In addition to the main task - unimodal and multimodal personality traits assessment, the features implemented in <a href="https://oceanai.readthedocs.io/en/latest/">OCEAN-AI</a> will allow researchers to solve other problems of analyzing human behavior, for example, affective state recognition.</p>

        <p>The library solves practical tasks:</p>
        <ol>
            <li><a href="https://oceanai.readthedocs.io/en/latest/user_guide/notebooks/Pipeline_practical_task_1.html">Ranking of potential candidates by professional responsibilities</a>.</li>
            <li><a href="https://oceanai.readthedocs.io/en/latest/user_guide/notebooks/Pipeline_practical_task_2.html">Predicting consumer preferences for industrial goods</a>.</li>
            <li><a href="https://oceanai.readthedocs.io/ru/latest/user_guide/notebooks/Pipeline_practical_task_3.html">Forming effective work teams</a>.</li>
        </ol>

        <p><a href="https://oceanai.readthedocs.io/en/latest/">OCEAN-AI</a> uses the latest open-source libraries for audio, video and text processing: <a href="https://librosa.org/">librosa</a>, <a href="https://audeering.github.io/opensmile-python/">openSMILE</a>, <a href="https://pypi.org/project/opencv-python/">openCV</a>, <a href="https://google.github.io/mediapipe/getting_started/python">mediapipe</a>, <a href="https://pypi.org/project/transformers">transformers</a>.</p>

        <p><a href="https://oceanai.readthedocs.io/en/latest/">OCEAN-AI</a> is written in the <a href="https://www.python.org/">python programming language</a>. Neural network models are implemented and trained using an open-source library code <a href="https://www.tensorflow.org/">TensorFlow</a>.</p>

        <hr>

        <h2>Research data</h2>

        <p>The <a href="https://oceanai.readthedocs.io/en/latest/">OCEAN-AI</a> library was tested on two corpora:</p>

        <ol>
            <li>The publicly available and large-scale <a href="https://chalearnlap.cvc.uab.cat/dataset/24/description/">First Impressions V2 corpus</a>.</li>
            <li>On the first publicly available Russian-language <a href="https://hci.nw.ru/en/pages/mupta-corpus">Multimodal Personality Traits Assessment (MuPTA) corpus</a>.</li>
        </ol>

        <hr>

        <h2>Publications</h2>

        <h3>Journals</h3>
        <pre>
            <code>
@article{ryumina24_prl,
    author = {Ryumina, Elena and Markitantov, Maxim and Ryumin, Dmitry and Karpov, Alexey},
    title = {Gated Siamese Fusion Network based on Multimodal Deep and Hand-Crafted Features for Personality Traits Assessment},
    journal = {Pattern Recognition Letters},
    volume = {185},
    pages = {45--51},
    year = {2024},
    doi = {<a href="https://doi.org/10.1016/j.patrec.2024.07.004">https://doi.org/10.1016/j.patrec.2024.07.004</a>},
}
@article{ryumina24_eswa,
    author = {Elena Ryumina and Maxim Markitantov and Dmitry Ryumin and Alexey Karpov},
    title = {OCEAN-AI Framework with EmoFormer Cross-Hemiface Attention Approach for Personality Traits Assessment},
    journal = {Expert Systems with Applications},
    volume = {239},
    pages = {122441},
    year = {2024},
    doi = {<a href="https://doi.org/10.1016/j.eswa.2023.122441">https://doi.org/10.1016/j.eswa.2023.122441</a>},
}
@article{ryumina22_neurocomputing,
    author = {Elena Ryumina and Denis Dresvyanskiy and Alexey Karpov},
    title = {In Search of a Robust Facial Expressions Recognition Model: A Large-Scale Visual Cross-Corpus Study},
    journal = {Neurocomputing},
    volume = {514},
    pages = {435-450},
    year = {2022},
    doi = {<a href="https://doi.org/10.1016/j.neucom.2022.10.013">https://doi.org/10.1016/j.neucom.2022.10.013</a>},
}
            </code>
        </pre>

        <h3>Conferences</h3>
        <pre>
            <code>
@inproceedings{ryumina24_interspeech,
    author = {Elena Ryumina and Dmitry Ryumin and and Alexey Karpov},
    title = {OCEAN-AI: Open Multimodal Framework for Personality Traits Assessment and HR-Processes Automatization},
    year = {2024},
    booktitle = {INTERSPEECH},
    pages = {3630--3631},
    doi = {<a href="https://www.isca-archive.org/interspeech_2024/ryumina24_interspeech.html#">https://www.isca-archive.org/interspeech_2024/ryumina24_interspeech.html#</a>},
}
@inproceedings{ryumina23_interspeech,
    author = {Elena Ryumina and Dmitry Ryumin and Maxim Markitantov and Heysem Kaya and Alexey Karpov},
    title = {Multimodal Personality Traits Assessment (MuPTA) Corpus: The Impact of Spontaneous and Read Speech},
    year = {2023},
    booktitle = {INTERSPEECH},
    pages = {4049--4053},
    doi = {<a href="https://doi.org/10.21437/Interspeech.2023-1686">https://doi.org/10.21437/Interspeech.2023-1686</a>},
}
            </code>
        </pre>
    </div>
</div>
"""
