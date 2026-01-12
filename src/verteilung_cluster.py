#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

# Input files (these come from your dispersion outputs)
INPUT_FILES = {
    "party_even": "outfiles/party/evenly_distributed_words.tsv",
    "party_uneven": "outfiles/party/unevenly_distributed_words.tsv",
    "block_even": "outfiles/blocks/evenly_distributed_words.tsv",
    "block_uneven": "outfiles/blocks/unevenly_distributed_words.tsv",
}

# Output directory
OUTPUT_DIR = "outfiles/cluster"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Embedding model
EMBEDDING_MODEL_NAME = "distilbert-base-german-cased"

# Clustering
N_CLUSTERS = 50

# TSNE parameters
TSNE_PERPLEXITY = 5

# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("✅ Embedding model loaded.")

# ----------------------------------------------------------
# EMBEDDING + CLUSTERING FUNCTION
# ----------------------------------------------------------

def embed_and_cluster(words_df, prefix, n_clusters=N_CLUSTERS):
    if words_df.empty:
        print(f"⚠️ No words to cluster for {prefix}.")
        return
    
    words = words_df["Word"].tolist()
    
    print(f"Embedding {len(words)} words for {prefix}...")
    embeddings = embedding_model.encode(
        words,
        batch_size=64,
        show_progress_bar=True
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    
    print(f"Clustering with KMeans (n_clusters={n_clusters}) for {prefix}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels
    words_df["Cluster"] = clusters
    
    # Save clustered words
    output_filename = os.path.join(OUTPUT_DIR, f"{prefix}_embedding_clusters.tsv")
    words_df.to_csv(output_filename, sep="\t", index=False)
    print(f"✅ Saved clustered words to {output_filename}")
    
    # Cluster summary
    summary = words_df.groupby("Cluster").agg({
        "Word": lambda x: ", ".join(x[:30]) + (" ..." if len(x) > 30 else ""),
        "Total_Frequency": "sum",
        # If present, dominant parties/blocks
        "Dominant_Party": lambda x: "; ".join(x.dropna().unique()) if "Dominant_Party" in words_df.columns else "",
        "Dominant_Block": lambda x: "; ".join(x.dropna().unique()) if "Dominant_Block" in words_df.columns else "",
    }).reset_index()
    summary_filename = os.path.join(OUTPUT_DIR, f"{prefix}_embedding_clusters_summary.tsv")
    summary.to_csv(summary_filename, sep="\t", index=False)
    print(f"✅ Saved cluster summary to {summary_filename}")
    
    # TSNE plot
    if len(words) >= 2:
        print(f"Creating TSNE plot for {prefix}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=TSNE_PERPLEXITY)
        embeddings_2d = tsne.fit_transform(X_scaled)
        
        plt.figure(figsize=(8,6))
        plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=clusters, cmap="tab20", s=15)
        plt.title(f"t-SNE Visualization - {prefix}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_embedding_clusters_tsne.png"))
        plt.close()
        print(f"✅ Saved TSNE plot for {prefix}")
    else:
        print(f"Skipping TSNE for {prefix}: not enough words.")

# ----------------------------------------------------------
# PROCESS ALL FILES
# ----------------------------------------------------------

for key, file_path in INPUT_FILES.items():
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        continue
    
    print(f"Processing {key} from {file_path}")
    df = pd.read_csv(file_path, sep="\t")
    
    # Add missing columns if not present
    if "Dominant_Party" not in df.columns:
        df["Dominant_Party"] = ""
    if "Dominant_Block" not in df.columns:
        df["Dominant_Block"] = ""
    
    embed_and_cluster(df, prefix=key)

print("\n🎉 All done!")