import re
import pickle
import pandas as pd
import nltk

from pymorphy3 import MorphAnalyzer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer


# ---------------------------------------------------------------------
# 1. INPUT
# ---------------------------------------------------------------------
INPUT_CSV = "filtered_posts_q3_331.csv"
TEXT_COLUMN = "ТЕКСТ"

df = pd.read_csv(INPUT_CSV)


# ---------------------------------------------------------------------
# 2. TEXT PREPROCESSING
# ---------------------------------------------------------------------
def clean_text(text):
    """Cleaning procedure used before lemmatization."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)          # mentions
    text = re.sub(r"#\w+", "", text)          # hashtags
    text = re.sub(r"[^\w\s]", "", text)       # punctuation/symbols
    text = re.sub(r"\d+", "", text)            # digits
    text = re.sub(r"\s+", " ", text).strip()  # repeated whitespace
    return text


morph = MorphAnalyzer()


def lemmatize_text(text):
    """Lemmatize Russian tokens with pymorphy3."""
    if pd.isna(text) or not text:
        return ""

    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmas)


df["text_clean"] = df[TEXT_COLUMN].apply(clean_text)
df_clean = df[df["text_clean"].str.len() > 0].copy()

df_clean["text_lemmatized"] = df_clean["text_clean"].apply(lemmatize_text)
df_clean = df_clean[df_clean["text_lemmatized"].str.len() > 0].copy()


# ---------------------------------------------------------------------
# 3. BERTopic CONFIGURATION
# ---------------------------------------------------------------------
embedding_model = SentenceTransformer(
    "paraphrase-multilingual-mpnet-base-v2"
)

nltk.download("stopwords")
russian_stopwords = nltk.corpus.stopwords.words("russian")

extra_stopwords = [
    "это", "так", "все", "уже", "там", "тут", "который", "является"
]
russian_stopwords.extend(extra_stopwords)

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    metric="euclidean",
    cluster_selection_epsilon=0.05,
    prediction_data=True,
)

vectorizer_model = CountVectorizer(
    min_df=5,
    max_df=0.8,
    stop_words=russian_stopwords,
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    language="russian",
    calculate_probabilities=True,
    verbose=True,
)


# ---------------------------------------------------------------------
# 4. MODEL FITTING
# ---------------------------------------------------------------------
topics, probs = topic_model.fit_transform(
    df_clean["text_lemmatized"].tolist()
)

df_clean["topic"] = topics
df_clean["topic_prob"] = probs.max(axis=1) if probs is not None else None

n_topics = len(set(topics)) - (1 if -1 in topics else 0)
n_outliers = sum(topic == -1 for topic in topics)

print(f"Documents: {len(df_clean):,}")
print(f"Topics: {n_topics}")
print(
    f"Outliers: {n_outliers:,} "
    f"({n_outliers / len(df_clean) * 100:.1f}%)"
)


# ---------------------------------------------------------------------
# 5. EXPORT DERIVED RESULTS
# ---------------------------------------------------------------------
topic_info = topic_model.get_topic_info()

# This export contains derived assignments. Remove columns that should not
# be redistributed before public release.
export_columns = [
    col for col in [
        "ССЫЛКА НА ПОСТ",
        "ССЫЛКА НА ВЛАДЕЛЬЦА СТЕНЫ",
        "ДАТА ПУБЛИКАЦИИ",
        "topic",
        "topic_prob",
    ]
    if col in df_clean.columns
]

df_clean[export_columns].to_csv(
    "posts_with_topics.csv",
    index=False,
)

topic_info.to_csv("topic_info.csv", index=False)

topics_words = {
    topic_id: topic_model.get_topic(topic_id)
    for topic_id in set(topics)
    if topic_id != -1
}

with open("topics_words.pkl", "wb") as f:
    pickle.dump(topics_words, f)

# Optional: save the fitted BERTopic model.
# topic_model.save("bertopic_model", save_embedding_model=False)
