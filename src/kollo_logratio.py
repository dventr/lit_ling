#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from math import log2
from tqdm import tqdm

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

CORPUS_FILE = "cleaned_file.tsv"

# List your keywords here
KEYWORDS = [
    "weiblich", "weibliche", "weiblichen", "weiblicher", "weibliches"
]

WINDOW_SIZE = 5

# NEW: Toggle lemmatization
USE_LEMMATIZATION = False  # Set to False to use raw tokens instead

PARTY_TO_IDEOLOGY = {
    'sp': 'sozial',
    'spd': 'sozial',
    'spoe': 'sozial',
    'linke': 'sozial',
    'gruene_oe': 'gruen',
    'gruene_de': 'gruen',
    'gruene_ch': 'gruen',
    'mitte': 'mittelinks',
    'fdp': 'mitterechts',
    'neos': 'mittelinks',
    'oevp': 'mitterechts',
    'cdu': 'mitterechts',
    'fpoe': 'rechts',
    'svp': 'rechts',
    'afd': 'rechts'
}

GROUP_FIELDS = ["ideology", "party", "land"]

OUTPUT_BASE_DIR = "outfiles_koll_kwic_ling"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ----------------------------------------------------------
# LOAD SPACY (only if needed)
# ----------------------------------------------------------

nlp = None
if USE_LEMMATIZATION:
    try:
        import spacy
        print("Loading spaCy model for German...")
        nlp = spacy.load("de_core_news_sm")
        print("✅ spaCy loaded successfully")
    except ImportError:
        print("⚠️ spaCy not installed. Install with: pip install spacy")
        print("Falling back to raw tokenization...")
        USE_LEMMATIZATION = False
    except OSError:
        print("⚠️ German spaCy model not found. Install with: python -m spacy download de_core_news_sm")
        print("Falling back to raw tokenization...")
        USE_LEMMATIZATION = False

# ----------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------

def tokenize_text(text):
    """
    Tokenize text with or without lemmatization based on USE_LEMMATIZATION setting.
    Returns a list of tokens (lemmas or raw tokens) preserving German capitalization.
    """
    if USE_LEMMATIZATION and nlp is not None:
        doc = nlp(text)
        return [token.lemma_ for token in doc if token.is_alpha]
    else:
        # Simple regex tokenization - preserve original case
        tokens = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
        return [token for token in tokens if token.strip()]

def extract_kwic_for_keyword(df_corpus, keyword, window_size):
    """
    Extract KWIC lines and collocate counts for one keyword in a corpus subset.
    KWIC output also contains the party and frequency of the keyword in that text.
    """
    kwic_lines = []
    collocate_counts = Counter()
    collocate_texts = defaultdict(set)
    keyword_occurrences = 0
    window_tokens_total = 0

    # Keep original keyword case for matching
    keyword_original = keyword

    for idx, row in df_corpus.iterrows():
        text = str(row["text"])
        partei = str(row["partei"]) if "partei" in row and not pd.isna(row["partei"]) else "unknown"

        tokens = tokenize_text(text)

        # Case-sensitive frequency of keyword in this document
        keyword_freq_in_text = tokens.count(keyword_original)

        for i, token in enumerate(tokens):
            if token == keyword_original:
                left = tokens[max(0, i - window_size):i]
                right = tokens[i + 1:i + 1 + window_size]

                kwic_lines.append({
                    "Left_Context": " ".join(left),
                    "Word": keyword,
                    "Right_Context": " ".join(right),
                    "Partei": partei,
                    "Keyword_Frequency_in_Text": keyword_freq_in_text
                })

                # Count collocates (preserving case)
                context = left + right
                for collocate in context:
                    collocate_counts[collocate] += 1
                    collocate_texts[collocate].add(idx)

                keyword_occurrences += 1
                window_tokens_total += len(context)

    return kwic_lines, collocate_counts, collocate_texts, keyword_occurrences, window_tokens_total

def compute_log_ratio_per_keyword(collocate_counts, corpus_counts, collocate_texts,
                                  window_tokens_total, total_tokens):
    """
    Compute log ratio for collocates for one keyword in one corpus subset.
    """
    colloc_data = []
    for word, O11 in collocate_counts.items():
        word_total = corpus_counts[word]
        expected_freq = (word_total / total_tokens) * window_tokens_total if total_tokens > 0 else 0
        observed_freq = O11
        texts_count = len(collocate_texts[word])

        # Compute relative frequencies
        observed_rate = (O11 / window_tokens_total) if window_tokens_total > 0 else 0.00000001
        expected_rate = (word_total / total_tokens) if total_tokens > 0 else 0.00000001

        if observed_rate > 0 and expected_rate > 0:
            log_ratio = log2(observed_rate / expected_rate)
        else:
            log_ratio = 0.0

        colloc_data.append({
            "Word": word,
            "Total no. in whole corpus": word_total,
            "Expected collocate frequency": round(expected_freq, 3),
            "Observed collocate frequency": observed_freq,
            "In no. of texts": texts_count,
            "Log Ratio (filtered)": round(log_ratio, 3)
        })

    df_colloc = pd.DataFrame(colloc_data)
    if not df_colloc.empty:
        df_colloc.sort_values("Log Ratio (filtered)", ascending=False, inplace=True)

    return df_colloc

def save_kwic(df_kwic, keyword, suffix=""):
    file_suffix = f"_{suffix}" if suffix else ""
    kwic_file = os.path.join(
        OUTPUT_BASE_DIR,
        f"kwic_{keyword}{file_suffix}.tsv"
    )
    df_kwic.to_csv(kwic_file, sep="\t", index=False)
    print(f"✅ KWIC saved: {kwic_file}")

def save_collocations(df_colloc, keyword, suffix=""):
    file_suffix = f"_{suffix}" if suffix else ""
    colloc_file = os.path.join(
        OUTPUT_BASE_DIR,
        f"collocations_{keyword}{file_suffix}.tsv"
    )
    df_colloc.to_csv(colloc_file, sep="\t", index=False)
    print(f"✅ Collocations saved: {colloc_file}")

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    print(f"Loading corpus file: {CORPUS_FILE}")
    print(f"Using lemmatization: {USE_LEMMATIZATION}")
    df_corpus = pd.read_csv(CORPUS_FILE, sep="\t")

    # -------------------------
    # Add derived ideology field
    # -------------------------
    print("Mapping parties to ideology...")
    def map_party(party):
        if pd.isna(party):
            return "unknown"
        party_lower = str(party).lower()
        return PARTY_TO_IDEOLOGY.get(party_lower, "unknown")

    df_corpus["ideology"] = df_corpus["partei"].apply(map_party)

    # Count entire corpus using tokenized text
    tokenization_method = "lemmatized" if USE_LEMMATIZATION else "raw"
    print(f"Counting words in entire corpus ({tokenization_method})...")
    corpus_counts = Counter()
    total_tokens = 0
    for _, row in tqdm(df_corpus.iterrows(), total=len(df_corpus), desc="Counting total corpus"):
        tokens = tokenize_text(str(row["text"]))
        corpus_counts.update(tokens)
        total_tokens += len(tokens)

    # -------------------------
    # Entire corpus analysis
    # -------------------------
    for keyword in KEYWORDS:
        (kwic_lines,
         collocate_counts,
         collocate_texts,
         keyword_occurrences,
         window_tokens_total) = extract_kwic_for_keyword(
            df_corpus,
            keyword,
            WINDOW_SIZE
        )

        # Save KWIC
        df_kwic = pd.DataFrame(kwic_lines)
        if not df_kwic.empty:
            save_kwic(df_kwic, keyword)
        else:
            print(f"⚠️ No KWIC hits for keyword '{keyword}' in entire corpus.")

        # Save collocations
        if collocate_counts:
            df_colloc = compute_log_ratio_per_keyword(
                collocate_counts,
                corpus_counts,
                collocate_texts,
                window_tokens_total,
                total_tokens
            )
            save_collocations(df_colloc, keyword)
        else:
            print(f"⚠️ No collocates for keyword '{keyword}' in entire corpus.")

    # -------------------------
    # Grouped analysis
    # -------------------------
    for field in GROUP_FIELDS:
        if field not in df_corpus.columns:
            print(f"⚠️ Skipping field '{field}' (not in corpus).")
            continue

        unique_values = sorted(df_corpus[field].dropna().unique())
        print(f"Field '{field}' has values: {unique_values}")

        for value in unique_values:
            df_subset = df_corpus[df_corpus[field] == value]
            if df_subset.empty:
                continue

            for keyword in KEYWORDS:
                (kwic_lines,
                 collocate_counts,
                 collocate_texts,
                 keyword_occurrences,
                 window_tokens_total) = extract_kwic_for_keyword(
                    df_subset,
                    keyword,
                    WINDOW_SIZE
                )

                suffix = f"{field}_{value}"

                # Save KWIC
                df_kwic = pd.DataFrame(kwic_lines)
                if not df_kwic.empty:
                    save_kwic(df_kwic, keyword, suffix)
                else:
                    print(f"⚠️ No KWIC hits for keyword '{keyword}' in {field}={value}.")

                # Save collocations
                if collocate_counts:
                    df_colloc = compute_log_ratio_per_keyword(
                        collocate_counts,
                        corpus_counts,
                        collocate_texts,
                        window_tokens_total,
                        total_tokens
                    )
                    save_collocations(df_colloc, keyword, suffix)
                else:
                    print(f"⚠️ No collocates for keyword '{keyword}' in {field}={value}.")

if __name__ == "__main__":
    main()