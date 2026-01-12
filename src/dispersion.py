#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import Counter
import os
import re
from tqdm import tqdm

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

# Input file
INPUT_FILENAME = 'cleaned_file.tsv'

# Output directory
OUTPUT_DIR = 'outfiles'

# General thresholds
MIN_TOTAL_FREQ = 5
MIN_VOCAB_FREQ = 5

# Thresholds for defining even/uneven distributions
MAX_JUILLANDS_D_EVEN = 0.75      # ≥ 0.75 = rather evenly distributed
MIN_KL_DIVERGENCE_UNEVEN = 0.3   # ≥ 0.3 = rather unevenly distributed

TOP_N = 1000                      # Number of top uneven words to save

# Define German stopwords
GERMAN_STOPWORDS = set([
    "und", "oder", "aber", "denn", "dass", "die", "der", "das", "ein", "eine",
    "einer", "eines", "dem", "den", "des", "mit", "zu", "auf", "im", "in",
    "für", "von", "an", "bei", "sich", "ist", "war", "wird", "werden", "hat",
    "haben", "sein", "ich", "du", "er", "sie", "es", "wir", "ihr", "sie",
    "nicht", "nur", "auch", "wie", "so", "noch", "schon", "wenn", "als",
    "dann", "doch", "da", "weil", "bis", "zwischen", "nach", "vor", "über",
    "unter", "dies", "dieser", "dieses", "diesem", "diese", "jedoch", "kann",
    "können", "muss", "müssen", "soll", "sollen", "will", "wollen"
])

# ----------------------------------------------------------
# STEP 1: LOAD DATA
# ----------------------------------------------------------

print(f"Loading data from: {INPUT_FILENAME}")
df = pd.read_csv(INPUT_FILENAME, sep='\t')

print(f"Data loaded. Shape: {df.shape}")

# Clean text column
df['text'] = df['text'].fillna("").astype(str)

# Check unique parties
original_parties = sorted(df['partei'].unique())
print(f"\nFound {len(original_parties)} unique parties in the data:")
print(original_parties)

# ----------------------------------------------------------
# STEP 2: AGGREGATE TEXTS BY PARTY
# ----------------------------------------------------------

print("\nAggregating texts per party...")

df_party = df.groupby('partei')['text'].apply(
    lambda texts: ' '.join(
        str(t) for t in texts.dropna()
    )
).reset_index(name='AggregatedText')

# Check that no parties are lost
aggregated_parties = sorted(df_party['partei'].unique())
missing_parties = set(original_parties) - set(aggregated_parties)

if missing_parties:
    print("⚠️ WARNING: The following parties are missing after aggregation:")
    print(missing_parties)
else:
    print("✅ All parties successfully aggregated.")

print(f"Aggregated texts shape: {df_party.shape}")
print(df_party.head())

# ----------------------------------------------------------
# STEP 3: TOKENIZE AND COUNT
# ----------------------------------------------------------

print("\nTokenizing texts and counting frequencies...")

# Tokenization regex: keep words, numbers, umlauts
TOKEN_REGEX = r'\b\w+\b'

total_counter = Counter()
party_counters = {}
part_sizes = []
total_words = 0

for _, row in df_party.iterrows():
    party = row['partei']
    text = row['AggregatedText']
    
    # Tokenize text
    tokens = re.findall(TOKEN_REGEX, text, flags=re.UNICODE)
    tokens = [t for t in tokens if t.strip()]
    
    counter = Counter(tokens)
    party_counters[party] = counter
    total_counter.update(counter)

    part_size = sum(counter.values())
    part_sizes.append(part_size)
    total_words += part_size

# Compute party size proportions
part_size_percentages = [size / total_words for size in part_sizes]

print("\nTokenization complete. Sample token stats:")
for party, size in zip(df_party['partei'], part_sizes):
    top_words = party_counters[party].most_common(10)
    print(f"- {party}: {size} tokens, top words: {top_words}")

print(f"\nVocabulary size before thresholding: {len(total_counter)}")

# Build vocabulary with frequency threshold
vocab = [word for word, freq in total_counter.items() if freq >= MIN_VOCAB_FREQ]
print(f"Vocabulary size after applying MIN_VOCAB_FREQ={MIN_VOCAB_FREQ}: {len(vocab)}")

# ----------------------------------------------------------
# STEP 4: COMPUTE DISPERSION MEASURES
# ----------------------------------------------------------

print("\nComputing dispersion measures...")

results = []

parties = df_party['partei'].tolist()

for word in tqdm(vocab, desc="Processing words"):
    freqs = np.array([party_counters[party][word] for party in parties])
    f = freqs.sum()

    if f == 0:
        continue

    # Relative percentages per party
    p = np.array([
        (freq / size * 100) if size > 0 else 0
        for freq, size in zip(freqs, part_sizes)
    ])

    # DP
    DP = 0.5 * sum(
        abs((v_i / f) - s_i) 
        for v_i, s_i in zip(freqs, part_size_percentages)
    )
    DP_norm = DP / (1 - min(part_size_percentages)) if (1 - min(part_size_percentages)) > 0 else 0

    # KL Divergence
    proportions = freqs / f
    nonzero = proportions > 0
    kl_div = np.sum(
        proportions[nonzero] *
        np.log2(proportions[nonzero] / np.array(part_size_percentages)[nonzero])
    ) if f > 0 else 0.0

    # Juilland’s D
    mean_p = np.mean(p)
    if mean_p == 0:
        Juillands_D = np.nan
    else:
        sd_population_p = np.std(p, ddof=0)
        Juillands_D = 1 - (sd_population_p / mean_p) * (1 / np.sqrt(len(parties)-1)) if len(parties) > 1 else 0

    # Relative frequencies per 10,000 words
    rel_freqs = [
        (freq / size * 10000) if size > 0 else 0
        for freq, size in zip(freqs, part_sizes)
    ]
    max_idx = int(np.argmax(rel_freqs))
    dominant_party = parties[max_idx] if f > 0 else ""

    # Save result
    result = {
        'Word': word,
        'Total_Frequency': f,
        'DP': DP,
        'DP_Norm': DP_norm,
        'KL_Divergence': kl_div,
        'Juillands_D': Juillands_D,
        'Dominant_Party': dominant_party,
    }
    for party, freq, rel in zip(parties, freqs, rel_freqs):
        result[f"Freq_{party}"] = freq
        result[f"RelFreqPer10k_{party}"] = rel

    results.append(result)

# ----------------------------------------------------------
# STEP 5: BUILD RESULTS DATAFRAME
# ----------------------------------------------------------

results_df = pd.DataFrame(results)

# Sort for readability
results_df_sorted = results_df.sort_values('KL_Divergence', ascending=False)

# Save complete file
os.makedirs(OUTPUT_DIR, exist_ok=True)

full_filename = os.path.join(OUTPUT_DIR, 'complete_freq_dispersion.tsv')
results_df_sorted.to_csv(full_filename, sep='\t', index=False)
print(f"\n✅ Saved full word list with dispersion info to {full_filename}")

# ----------------------------------------------------------
# STEP 6: SELECT EVENLY DISTRIBUTED WORDS
# ----------------------------------------------------------

even_words_df = results_df_sorted[
    (results_df_sorted['Total_Frequency'] >= MIN_TOTAL_FREQ) &
    (results_df_sorted['Juillands_D'] >= MAX_JUILLANDS_D_EVEN)
]

# Remove stopwords
even_words_df = even_words_df[
    ~even_words_df['Word'].str.lower().isin(GERMAN_STOPWORDS)
]

even_filename = os.path.join(OUTPUT_DIR, 'evenly_distributed_words.tsv')
even_words_df.to_csv(even_filename, sep='\t', index=False)

print(f"✅ Saved evenly distributed words (stopwords removed) to {even_filename}")
print(f"Number of evenly distributed words: {len(even_words_df)}")

# ----------------------------------------------------------
# STEP 7: SELECT UNEVENLY DISTRIBUTED WORDS
# ----------------------------------------------------------

uneven_words_df = results_df_sorted[
    (results_df_sorted['Total_Frequency'] >= MIN_TOTAL_FREQ) &
    (results_df_sorted['KL_Divergence'] >= MIN_KL_DIVERGENCE_UNEVEN) &
    (results_df_sorted['Juillands_D'] <= (1 - MAX_JUILLANDS_D_EVEN))
]

uneven_filename = os.path.join(OUTPUT_DIR, 'unevenly_distributed_words.tsv')
uneven_words_df.to_csv(uneven_filename, sep='\t', index=False)

print(f"✅ Saved unevenly distributed words to {uneven_filename}")
print(f"Number of unevenly distributed words: {len(uneven_words_df)}")

# Save only top N
uneven_topN_df = uneven_words_df.head(TOP_N)
topN_filename = os.path.join(OUTPUT_DIR, f'unevenly_distributed_words_TOP{TOP_N}.tsv')
uneven_topN_df.to_csv(topN_filename, sep='\t', index=False)

print(f"✅ Saved TOP {TOP_N} unevenly distributed words to {topN_filename}")

print("\n🎉 All done!")