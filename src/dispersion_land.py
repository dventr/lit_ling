#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict, Counter

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan

from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec

import plotly.express as px

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

CORPUS_FILE = "cleaned_file.tsv"

OUTPUT_BASE_DIR = "outfiles_land"
CLUSTER_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "clusters")
PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "plots")

EVEN_OUTPUT = os.path.join(OUTPUT_BASE_DIR, "evenly_distributed_words.tsv")
UNEVEN_OUTPUT = os.path.join(OUTPUT_BASE_DIR, "unevenly_distributed_words.tsv")

OUTPUT_DIRS = {
    "bert_kmeans": os.path.join(CLUSTER_OUTPUT_DIR, "bert", "kmeans"),
    "bert_hdbscan": os.path.join(CLUSTER_OUTPUT_DIR, "bert", "hdbscan"),
    "w2v_kmeans": os.path.join(CLUSTER_OUTPUT_DIR, "word2vec", "kmeans"),
    "w2v_hdbscan": os.path.join(CLUSTER_OUTPUT_DIR, "word2vec", "hdbscan"),
}

PLOTS_DIRS = {
    key: path.replace(CLUSTER_OUTPUT_DIR, PLOTS_OUTPUT_DIR)
    for key, path in OUTPUT_DIRS.items()
}

for path in [OUTPUT_BASE_DIR, CLUSTER_OUTPUT_DIR, PLOTS_OUTPUT_DIR] + \
            list(OUTPUT_DIRS.values()) + list(PLOTS_DIRS.values()):
    os.makedirs(path, exist_ok=True)

JUILLAND_THRESHOLD = 0.75
MIN_TOTAL_OCCURRENCES = 5

N_CLUSTERS_KMEANS = 30
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

TSNE_PERPLEXITY = 5

BERT_MODEL_NAME = "distiluse-base-multilingual-cased-v2"
WORD2VEC_MODEL_PATH = "/Users/dventr/Downloads/tp1_compass.model"

# ----------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------

print(f"Loading BERT model: {BERT_MODEL_NAME}")
bert_model = SentenceTransformer(BERT_MODEL_NAME)
print("✅ BERT model loaded.")

print(f"Loading Word2Vec model: {WORD2VEC_MODEL_PATH}")
w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH)
print("✅ Word2Vec model loaded.")

# ----------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------

def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens

def count_words_by_country(df_corpus):
    word_counts = defaultdict(lambda: defaultdict(int))

    for _, row in tqdm(df_corpus.iterrows(), total=len(df_corpus), desc="Counting words"):
        text = str(row["text"])
        land = str(row["land"])
        tokens = tokenize(text)
        for token in tokens:
            word_counts[token][land] += 1

    records = []
    for word, counts in word_counts.items():
        freq_de = counts.get("DE", 0)
        freq_oe = counts.get("OE", 0)
        freq_ch = counts.get("CH", 0)
        total = freq_de + freq_oe + freq_ch
        record = {
            "Word": word,
            "Freq_Land_DE": freq_de,
            "Freq_Land_OE": freq_oe,
            "Freq_Land_CH": freq_ch,
            "Total_Frequency": total
        }
        records.append(record)

    df_counts = pd.DataFrame(records)
    return df_counts

def compute_juillands_d(df_counts):
    country_cols = ["Freq_Land_DE", "Freq_Land_OE", "Freq_Land_CH"]

    def calc_d(row):
        counts = np.array([row[col] for col in country_cols])
        total = counts.sum()
        if total == 0:
            return 1.0
        probs = counts / total
        n = len(probs)
        mean = 1.0 / n
        ssd = np.sum((probs - mean) ** 2)
        denominator = np.sqrt((n - 1) / n)
        if denominator == 0:
            return 1.0
        D = 1 - (np.sqrt(ssd) / denominator)
        return D

    df_counts["Juillands_D"] = df_counts.apply(calc_d, axis=1)
    df_counts["Evenly_Distributed_Countries"] = df_counts["Juillands_D"] >= JUILLAND_THRESHOLD
    return df_counts

def extract_contexts(words, texts, n_contexts=20):
    contexts = defaultdict(list)
    regex_lookup = {
        word: re.compile(r'\b' + re.escape(word) + r'\b', flags=re.IGNORECASE)
        for word in words
    }

    for text in tqdm(texts, desc="Extracting contexts"):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            for word, pattern in regex_lookup.items():
                if pattern.search(sentence):
                    if len(contexts[word]) < n_contexts:
                        contexts[word].append(sentence)
    return contexts

def embed_words_bert(words, contexts):
    vectors = []
    for word in tqdm(words, desc="Embedding words with BERT"):
        word_contexts = contexts.get(word, [])
        if word_contexts:
            embeds = bert_model.encode(word_contexts)
            avg_vec = np.mean(embeds, axis=0)
        else:
            avg_vec = np.zeros(bert_model.get_sentence_embedding_dimension())
        vectors.append(avg_vec)
    return np.array(vectors)

def embed_words_word2vec(words):
    vectors = []
    for word in tqdm(words, desc="Embedding words with Word2Vec"):
        word_lower = word.lower()
        if word_lower in w2v_model.wv:
            vectors.append(w2v_model.wv[word_lower])
        else:
            vectors.append(np.zeros(w2v_model.vector_size))
    return np.array(vectors)

def cluster_kmeans(X, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return model.fit_predict(X)

def cluster_hdbscan(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC
    )
    return clusterer.fit_predict(X)

def plot_semantic_scatter(embeddings_2d, clusters, words, output_file, title):
    df_plot = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "Cluster": clusters,
        "Word": words
    })

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color=df_plot["Cluster"].astype(str),
        text="Word",
        hover_name="Word",
        title=title,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        height=600
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.write_html(output_file)
    print(f"✅ Plot saved: {output_file}")

def save_cluster_results(df, clusters, embeddings, output_dir, plots_dir, algo):
    df["Cluster"] = clusters

    # Save full cluster data to TSV
    df_sorted = df.sort_values(["Cluster", "Word"])
    tsv_file = os.path.join(output_dir, f"clusters_full_{algo}.tsv")
    df_sorted.to_csv(tsv_file, sep="\t", index=False)
    print(f"✅ Full cluster list saved: {tsv_file}")

    # Save cluster summary
    summary_rows = []
    grouped = df.groupby("Cluster")
    for cluster_label, group in grouped:
        cluster_name = "Noise" if cluster_label == -1 else f"Cluster {cluster_label}"
        words_list = group["Word"].tolist()
        words_sample = ", ".join(words_list[:20]) + (" ..." if len(words_list) > 20 else "")

        total_land_freqs = Counter()
        for _, row in group.iterrows():
            for col in ["Freq_Land_DE", "Freq_Land_OE", "Freq_Land_CH"]:
                total_land_freqs[col.replace("Freq_Land_", "")] += row[col]

        top_lands = "; ".join(
            f"{land}: {count}" for land, count in total_land_freqs.items() if count > 0
        )

        summary_rows.append({
            "Cluster": cluster_label,
            "Cluster_Name": cluster_name,
            "Num_Words": len(group),
            "Words_Sample": words_sample,
            "Top_Lands_in_Cluster": top_lands
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(output_dir, f"clusters_summary_{algo}.tsv")
    summary_df.to_csv(summary_file, sep="\t", index=False)
    print(f"✅ Cluster summary saved: {summary_file}")

    if len(df) >= 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=TSNE_PERPLEXITY)
        embeddings_2d = tsne.fit_transform(embeddings)
        plot_file = os.path.join(plots_dir, f"semantic_scatter_{algo}.html")
        plot_semantic_scatter(
            embeddings_2d,
            clusters,
            df["Word"].tolist(),
            plot_file,
            title=f"Semantic Scatter Plot ({algo})"
        )

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    print(f"Loading corpus file: {CORPUS_FILE}")
    df_corpus = pd.read_csv(CORPUS_FILE, sep="\t")
    texts = df_corpus["text"].dropna().astype(str).tolist()

    # Count words
    df_counts = count_words_by_country(df_corpus)
    df_counts = df_counts[df_counts["Total_Frequency"] >= MIN_TOTAL_OCCURRENCES]

    # Compute dispersion
    df_counts = compute_juillands_d(df_counts)

    # Split even/uneven
    df_even = df_counts[df_counts["Evenly_Distributed_Countries"] == True]
    df_uneven = df_counts[df_counts["Evenly_Distributed_Countries"] == False]

    df_even.to_csv(EVEN_OUTPUT, sep="\t", index=False)
    df_uneven.to_csv(UNEVEN_OUTPUT, sep="\t", index=False)

    print(f"✅ Evenly distributed words saved: {EVEN_OUTPUT}")
    print(f"✅ Unevenly distributed words saved: {UNEVEN_OUTPUT}")

    # Cluster both groups
    for label, df in [("even", df_even), ("uneven", df_uneven)]:
        if len(df) < 2:
            print(f"⚠️ Skipping clustering for {label} (too few words)")
            continue

        words = df["Word"].tolist()
        contexts = extract_contexts(words, texts)

        # ---------------- BERT ----------------
        embeddings_bert = embed_words_bert(words, contexts)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embeddings_bert)

        clusters_kmeans = cluster_kmeans(X_scaled, N_CLUSTERS_KMEANS)
        save_cluster_results(
            df.copy(),
            clusters_kmeans,
            embeddings_bert,
            OUTPUT_DIRS["bert_kmeans"],
            PLOTS_DIRS["bert_kmeans"],
            algo=f"bert_kmeans_{label}"
        )

        clusters_hdb = cluster_hdbscan(X_scaled)
        save_cluster_results(
            df.copy(),
            clusters_hdb,
            embeddings_bert,
            OUTPUT_DIRS["bert_hdbscan"],
            PLOTS_DIRS["bert_hdbscan"],
            algo=f"bert_hdbscan_{label}"
        )

        # ---------------- WORD2VEC ----------------
        embeddings_w2v = embed_words_word2vec(words)
        scaler = StandardScaler()
        X_scaled_w2v = scaler.fit_transform(embeddings_w2v)

        pca = PCA(n_components=50)
        X_reduced = pca.fit_transform(X_scaled_w2v)

        clusters_kmeans_w2v = cluster_kmeans(X_reduced, N_CLUSTERS_KMEANS)
        save_cluster_results(
            df.copy(),
            clusters_kmeans_w2v,
            embeddings_w2v,
            OUTPUT_DIRS["w2v_kmeans"],
            PLOTS_DIRS["w2v_kmeans"],
            algo=f"w2v_kmeans_{label}"
        )

        clusters_hdb_w2v = cluster_hdbscan(X_reduced)
        save_cluster_results(
            df.copy(),
            clusters_hdb_w2v,
            embeddings_w2v,
            OUTPUT_DIRS["w2v_hdbscan"],
            PLOTS_DIRS["w2v_hdbscan"],
            algo=f"w2v_hdbscan_{label}"
        )

    print("\n🎉 All done!")

if __name__ == "__main__":
    main()