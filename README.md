# eco_well_being
Code extracted and cleaned from the research notebook used for the manuscript “Grassroots Environmental Discourse and the National Project ‘Ecological Well-Being’: Topic Modeling of Russian VKontakte Communities”.

## Contents

`01_preprocess_and_bertopic.py` — text cleaning, lemmatization, BERTopic configuration,
model fitting, and export of topic assignments and topic information.

`02_topic_labels_deepseek.py` — prompt and API workflow used to generate concise
English topic labels from topic keywords and representative posts.

`bertopic_topics_with_examples.xslx` — topics for DeepSeek analysis.

`requirements.txt` — main Python dependencies.

## Important notes

Raw VKontakte post texts are not included in this repo.

API credentials are not included. The DeepSeek script reads the key from the
`DEEPSEEK_API_KEY` environment variable.

