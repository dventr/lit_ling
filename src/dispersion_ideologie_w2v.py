#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
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

# Input dispersion files
INPUT_FILES = {
    "party_even": "outfiles/party/evenly_distributed_words.tsv",
    "party_uneven": "outfiles/party/unevenly_distributed_words.tsv",
    "block_even": "outfiles/blocks/evenly_distributed_words.tsv",
    "block_uneven": "outfiles/blocks/unevenly_distributed_words.tsv",
}

# Corpus file for extracting contexts
CORPUS_FILE = "cleaned_file.tsv"

# Output structure
OUTPUT_BASE_DIR = "outfiles"
CLUSTER_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "clusters")
PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "plots")

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

for path in list(OUTPUT_DIRS.values()) + list(PLOTS_DIRS.values()):
    os.makedirs(path, exist_ok=True)

# Models
BERT_MODEL_NAME = "distiluse-base-multilingual-cased-v2"
WORD2VEC_MODEL_PATH = "/Users/dventr/Downloads/tp1_compass.model"

# Clustering params
N_CLUSTERS_KMEANS = 30
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

TSNE_PERPLEXITY = 5

# Contexts
N_CONTEXTS_PER_WORD = 20

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
# LOAD CORPUS
# ----------------------------------------------------------

print(f"Loading corpus file: {CORPUS_FILE}")
df_corpus = pd.read_csv(CORPUS_FILE, sep="\t")
texts = df_corpus["text"].dropna().astype(str).tolist()

# ----------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------

def extract_contexts(vocab, texts, n_contexts=N_CONTEXTS_PER_WORD):
    contexts = defaultdict(list)
    regex_lookup = {
        word: re.compile(r'\b' + re.escape(word) + r'\b', flags=re.IGNORECASE)
        for word in vocab
    }

    for text in tqdm(texts, desc="Extracting contexts"):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            for word, pattern in regex_lookup.items():
                if pattern.search(sent):
                    if len(contexts[word]) < n_contexts:
                        contexts[word].append(sent)
    return contexts

def embed_words_bert(vocab, contexts):
    vectors = []
    for word in tqdm(vocab, desc="Embedding words with BERT"):
        word_contexts = contexts.get(word, [])
        if word_contexts:
            embeds = bert_model.encode(word_contexts)
            avg_vec = np.mean(embeds, axis=0)
        else:
            avg_vec = np.zeros(bert_model.get_sentence_embedding_dimension())
        vectors.append(avg_vec)
    return np.array(vectors)

def embed_words_word2vec(words):
    vecs = []
    for word in tqdm(words, desc="Embedding words with Word2Vec"):
        word_lower = word.lower()
        if word_lower in w2v_model.wv:
            vecs.append(w2v_model.wv[word_lower])
        else:
            vecs.append(np.zeros(w2v_model.vector_size))
    return np.array(vecs)

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

def plot_semantic_scatter(embeddings_2d, clusters, words, output_file, title="Semantic Scatter Plot"):
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
    print(f"✅ Semantic scatter plot saved to {output_file}")

def save_cluster_results(df, clusters, embeddings, output_dir, plots_dir, prefix, algo, params):
    df["Cluster"] = clusters

    party_cols = [col for col in df.columns if col.startswith("Freq_")]
    party_usage = []
    for _, row in df.iterrows():
        usage = []
        for col in party_cols:
            if row[col] > 0:
                party_name = col.replace("Freq_", "")
                usage.append(f"{party_name}: {int(row[col])}")
        party_usage.append("; ".join(usage))
    df["Parties_Using_Word"] = party_usage

    clustered_file = os.path.join(output_dir, f"{prefix}_{algo}_clusters.tsv")
    df.to_csv(clustered_file, sep="\t", index=False)
    print(f"✅ Saved clustered data to {clustered_file}")

    summary_rows = []
    grouped = df.groupby("Cluster")
    for cluster_label, group in grouped:
        cluster_name = "Noise" if cluster_label == -1 else f"Cluster {cluster_label}"
        words_list = group["Word"].tolist()
        words_str = ", ".join(words_list[:20]) + (" ..." if len(words_list) > 20 else "")
        dominant_party = "; ".join(group["Dominant_Party"].dropna().unique()) if "Dominant_Party" in group.columns else ""
        dominant_block = "; ".join(group["Dominant_Block"].dropna().unique()) if "Dominant_Block" in group.columns else ""

        total_freqs = Counter()
        for _, row in group.iterrows():
            for col in party_cols:
                total_freqs[col.replace("Freq_", "")] += row[col]

        top_parties = "; ".join(
            f"{party}: {count}" for party, count in total_freqs.most_common() if count > 0
        )

        summary_rows.append({
            "Cluster": cluster_label,
            "Cluster_Name": cluster_name,
            "Num_Words": len(group),
            "Dominant_Parties": dominant_party,
            "Dominant_Blocks": dominant_block,
            "Words_Sample": words_str,
            "Top_Parties_in_Cluster": top_parties
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(output_dir, f"{prefix}_{algo}_clusters_summary.tsv")
    summary_df.to_csv(summary_file, sep="\t", index=False)
    print(f"✅ Saved summary to {summary_file}")

    param_file = os.path.join(output_dir, f"{prefix}_{algo}_params.txt")
    with open(param_file, "w", encoding="utf-8") as f:
        f.write(f"Clustering Algorithm: {algo}\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    # Visualizations
    words_list = df["Word"].tolist()

    if len(df) >= 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=TSNE_PERPLEXITY)
        embeddings_2d = tsne.fit_transform(embeddings)

        semantic_plot_file = os.path.join(plots_dir, f"{prefix}_{algo}_semantic_scatter.html")
        plot_semantic_scatter(
            embeddings_2d,
            clusters,
            words_list,
            semantic_plot_file,
            title=f"Semantic Scatter Plot - {prefix} ({algo})"
        )
    else:
        print(f"⚠️ Skipping plots for {prefix} ({algo}) – not enough points.")

# ----------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------

for key, file_path in INPUT_FILES.items():
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        continue

    print(f"\nProcessing {key} from {file_path}")
    df = pd.read_csv(file_path, sep="\t")

    if "Dominant_Party" not in df.columns:
        df["Dominant_Party"] = ""
    if "Dominant_Block" not in df.columns:
        df["Dominant_Block"] = ""

    vocab = df["Word"].tolist()

    # ---------------- BERT ----------------
    print("🔶 BERT embeddings...")
    contexts = extract_contexts(vocab, texts)
    embeddings_bert = embed_words_bert(vocab, contexts)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings_bert)

    clusters_kmeans = cluster_kmeans(X_scaled, N_CLUSTERS_KMEANS)
    save_cluster_results(
        df.copy(),
        clusters_kmeans,
        embeddings_bert,
        OUTPUT_DIRS["bert_kmeans"],
        PLOTS_DIRS["bert_kmeans"],
        prefix=key,
        algo="bert_kmeans",
        params={"Embedding_Model": BERT_MODEL_NAME, "n_clusters": N_CLUSTERS_KMEANS}
    )

    clusters_hdbscan = cluster_hdbscan(X_scaled)
    save_cluster_results(
        df.copy(),
        clusters_hdbscan,
        embeddings_bert,
        OUTPUT_DIRS["bert_hdbscan"],
        PLOTS_DIRS["bert_hdbscan"],
        prefix=key,
        algo="bert_hdbscan",
        params={
            "Embedding_Model": BERT_MODEL_NAME,
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "min_samples": HDBSCAN_MIN_SAMPLES,
            "metric": HDBSCAN_METRIC
        }
    )

    # ---------------- WORD2VEC ----------------
    print("🔷 Word2Vec embeddings...")
    embeddings_w2v = embed_words_word2vec(vocab)

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
        prefix=key,
        algo="w2v_kmeans",
        params={
            "Embedding_Model": WORD2VEC_MODEL_PATH,
            "n_clusters": N_CLUSTERS_KMEANS
        }
    )

    clusters_hdbscan_w2v = cluster_hdbscan(X_reduced)
    save_cluster_results(
        df.copy(),
        clusters_hdbscan_w2v,
        embeddings_w2v,
        OUTPUT_DIRS["w2v_hdbscan"],
        PLOTS_DIRS["w2v_hdbscan"],
        prefix=key,
        algo="w2v_hdbscan",
        params={
            "Embedding_Model": WORD2VEC_MODEL_PATH,
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "min_samples": HDBSCAN_MIN_SAMPLES,
            "metric": HDBSCAN_METRIC
        }
    )

print("\n🎉 All done!")