#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from math import log2
from tqdm import tqdm
import spacy

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

CORPUS_FILE = "cleaned_file.tsv"

# Your keyword list (lemmas, case-sensitive)
KEYWORDS = [
    "leistungsfeindlich", "LGBTIQ"
]

WINDOW_SIZE = 5
KWIC_WINDOW_SIZE = 20      # für die KWIC-Ausgabe
COLLOC_WINDOW_SIZE = 5     # für die Kollokationszählung

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

OUTPUT_BASE_DIR = "outfiles_kwic_ling_LL"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ----------------------------------------------------------
# LOAD SPACY
# ----------------------------------------------------------

print("Loading spaCy model for German...")
nlp = spacy.load("de_core_news_sm")

# ----------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------

def safe_log(x, y):
    return 0 if x == 0 else x * np.log(x / y)

def lemmatize_with_surface(text):
    """
    Returns two lists:
    - surface forms (tokens)
    - lemmas
    Only alphabetic tokens are included.
    """
    doc = nlp(text)
    surface = [token.text for token in doc if token.is_alpha]
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    return surface, lemmas

def extract_kwic_for_keyword(df_corpus, keyword, kwic_window_size, colloc_window_size):
    """
    Extracts KWIC lines using a larger surface window,
    and collocates using a separate smaller lemma window.
    """
    kwic_lines = []
    collocate_counts = Counter()
    collocate_texts = defaultdict(set)
    keyword_occurrences = 0
    window_tokens_total = 0

    for idx, row in df_corpus.iterrows():
        text = str(row["text"])
        partei = str(row["partei"]) if "partei" in row and not pd.isna(row["partei"]) else "unknown"

        tokens_surface, tokens_lemmas = lemmatize_with_surface(text)

        keyword_freq_in_text = tokens_lemmas.count(keyword)

        for i, lemma in enumerate(tokens_lemmas):
            if lemma == keyword:
                # larger window for KWIC
                left_kwic = tokens_surface[max(0, i - kwic_window_size):i]
                right_kwic = tokens_surface[i + 1:i + 1 + kwic_window_size]

                kwic_lines.append({
                    "Left_Context": " ".join(left_kwic),
                    "Word": tokens_surface[i],
                    "Right_Context": " ".join(right_kwic),
                    "Partei": partei,
                    "Keyword_Frequency_in_Text": keyword_freq_in_text
                })

                # smaller window for collocation counting
                left_colloc = tokens_lemmas[max(0, i - colloc_window_size):i]
                right_colloc = tokens_lemmas[i + 1:i + 1 + colloc_window_size]
                context_lemmas = left_colloc + right_colloc

                for collocate in context_lemmas:
                    collocate_counts[collocate] += 1
                    collocate_texts[collocate].add(idx)

                keyword_occurrences += 1
                window_tokens_total += len(context_lemmas)

    return kwic_lines, collocate_counts, collocate_texts, keyword_occurrences, window_tokens_total

def compute_log_likelihood_and_tscore(collocate_counts, corpus_counts, collocate_texts,
                                      window_tokens_total, total_tokens):
    """
    Compute log-likelihood and corrected t-score for collocates.
    """
    colloc_data = []

    for word, O11 in collocate_counts.items():
        word_total = corpus_counts[word]
        O12 = window_tokens_total - O11
        O21 = word_total - O11
        O22 = (total_tokens - word_total) - O12

        # avoid zeros
        O11 = max(O11, 0.00000001)
        O12 = max(O12, 0.00000001)
        O21 = max(O21, 0.00000001)
        O22 = max(O22, 0.00000001)

        N = O11 + O12 + O21 + O22

        E11 = ((O11 + O12) * (O11 + O21)) / N
        E12 = ((O11 + O12) * (O12 + O22)) / N
        E21 = ((O21 + O22) * (O11 + O21)) / N
        E22 = ((O21 + O22) * (O12 + O22)) / N

        LL = 2 * (
            safe_log(O11, E11) +
            safe_log(O12, E12) +
            safe_log(O21, E21) +
            safe_log(O22, E22)
        )

        # corrected t-score
        tscore = (O11 - E11) / np.sqrt(O11) if O11 > 0 else 0.0

        texts_count = len(collocate_texts[word])

        colloc_data.append({
            "Word": word,
            "Total no. in whole corpus": word_total,
            "Observed collocate frequency": O11,
            "Expected collocate frequency": round(E11, 3),
            "In no. of texts": texts_count,
            "Log-Likelihood": round(LL, 3),
            "Corrected t-score": round(tscore, 3)
        })

    df_colloc = pd.DataFrame(colloc_data)
    if not df_colloc.empty:
        df_colloc.sort_values("Log-Likelihood", ascending=False, inplace=True)

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
    df_corpus = pd.read_csv(CORPUS_FILE, sep="\t")

    # Map parties to ideologies
    print("Mapping parties to ideology...")
    def map_party(party):
        if pd.isna(party):
            return "unknown"
        party_lower = str(party).lower()
        return PARTY_TO_IDEOLOGY.get(party_lower, "unknown")

    df_corpus["ideology"] = df_corpus["partei"].apply(map_party)

    # Count entire corpus (lemmatized)
    print("Counting words in entire corpus (lemmatized)...")
    corpus_counts = Counter()
    total_tokens = 0
    for _, row in tqdm(df_corpus.iterrows(), total=len(df_corpus), desc="Counting total corpus"):
        _, tokens_lemmas = lemmatize_with_surface(str(row["text"]))
        corpus_counts.update(tokens_lemmas)
        total_tokens += len(tokens_lemmas)

    # -------------------------
    # Entire corpus analysis
    # -------------------------
    for keyword in KEYWORDS:
        kwic_lines, collocate_counts, collocate_texts, keyword_occurrences, window_tokens_total = \
            extract_kwic_for_keyword(df_corpus, keyword, KWIC_WINDOW_SIZE, COLLOC_WINDOW_SIZE)

        # Save KWIC
        df_kwic = pd.DataFrame(kwic_lines)
        if not df_kwic.empty:
            save_kwic(df_kwic, keyword)
        else:
            print(f"⚠️ No KWIC hits for keyword '{keyword}' in entire corpus.")

        # Save collocations
        if collocate_counts:
            df_colloc = compute_log_likelihood_and_tscore(
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
                kwic_lines, collocate_counts, collocate_texts, keyword_occurrences, window_tokens_total = \
                    extract_kwic_for_keyword(df_subset, keyword, KWIC_WINDOW_SIZE, COLLOC_WINDOW_SIZE)


                suffix = f"{field}_{value}"

                # Save KWIC
                df_kwic = pd.DataFrame(kwic_lines)
                if not df_kwic.empty:
                    save_kwic(df_kwic, keyword, suffix)
                else:
                    print(f"⚠️ No KWIC hits for keyword '{keyword}' in {field}={value}.")

                # Save collocations
                if collocate_counts:
                    df_colloc = compute_log_likelihood_and_tscore(
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