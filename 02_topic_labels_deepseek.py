import os
import pandas as pd
from openai import OpenAI


# ---------------------------------------------------------------------
# Topic-label generation with DeepSeek
# ---------------------------------------------------------------------
# The API key is deliberately not stored in this file.
# Example:
#   export DEEPSEEK_API_KEY="..."
#
# Input table should contain topic keywords and representative examples.
INPUT_XLSX = "bertopic_topics_with_examples.xlsx"
OUTPUT_XLSX = "bertopic_topics_with_deepseek_labels.xlsx"

df = pd.read_excel(INPUT_XLSX)


def build_prompt(keywords, examples):
    return f"""
You are a professional computational linguist.

Given a topic described by keywords and example texts, produce a short,
clear English topic label (up to 10 words).

Keywords:
{keywords}

Examples:
{examples}

Return ONLY the label. The label should be in English. The labels should
be suitable for publishing in a scientific paper.
""".strip()


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


def generate_topic_name(keywords, examples):
    prompt = build_prompt(keywords, examples)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


# Adapt these column names if the exported table uses different names.
KEYWORDS_COLUMN = "keywords"
EXAMPLES_COLUMN = "examples"

if KEYWORDS_COLUMN in df.columns and EXAMPLES_COLUMN in df.columns:
    df["deepseek_label"] = [
        generate_topic_name(keywords, examples)
        for keywords, examples in zip(
            df[KEYWORDS_COLUMN],
            df[EXAMPLES_COLUMN],
        )
    ]
    df.to_excel(OUTPUT_XLSX, index=False)
else:
    raise ValueError(
        "Expected columns 'keywords' and 'examples'. "
        "Adapt KEYWORDS_COLUMN and EXAMPLES_COLUMN to the exported topic table."
    )
