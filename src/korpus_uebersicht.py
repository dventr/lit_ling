#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd

CORPUS_FILE = "cleaned_file.tsv"
OUT_DIR = "outfiles"
os.makedirs(OUT_DIR, exist_ok=True)


def normalize_year(value: str):
    """Extract a 4-digit year (1900–2100) from a possibly messy field.
    Returns int year or pd.NA if none found.
    """
    if pd.isna(value):
        return pd.NA
    s = str(value)
    m = re.search(r"(19\d{2}|20\d{2}|2100)", s)
    if m:
        try:
            year = int(m.group(1))
            return year
        except Exception:
            return pd.NA
    return pd.NA


def count_tokens(text: str) -> int:
    """Rough token count: count word-like sequences (unicode aware)."""
    if pd.isna(text):
        return 0
    return len(re.findall(r"\b\w+\b", str(text), flags=re.UNICODE))


def main():
    print(f"Lade Korpusdatei: {CORPUS_FILE}")
    df = pd.read_csv(CORPUS_FILE, sep="\t")

    # Ensure expected columns exist
    expected = ["partei", "land", "jahr", "text"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in {CORPUS_FILE}: {missing}")

    # Normalize year and compute basic stats per Dokument
    print("Normalisiere Jahr und zähle Tokens je Dokument…")
    df["jahr_raw"] = df["jahr"]
    df["jahr"] = df["jahr"].apply(normalize_year)
    df["tokens"] = df["text"].apply(count_tokens)
    df["chars"] = df["text"].astype(str).str.len()

    # Per-Dokument-Übersicht
    cols_doc = ["partei", "land", "jahr_raw", "jahr", "tokens", "chars"]
    doc_overview = df[cols_doc].copy()
    doc_path = os.path.join(OUT_DIR, "corpus_overview.tsv")
    doc_overview.to_csv(doc_path, sep="\t", index=False)
    print(f"✅ Per-Dokument-Übersicht gespeichert: {doc_path}")

    # Aggregation: Partei x Jahr x Land
    print("Aggregiere nach Partei × Land × Jahr…")
    by_party_year = (
        df.groupby(["partei", "land", "jahr"], dropna=False)
          .agg(
              docs=("text", "size"),
              tokens_sum=("tokens", "sum"),
              tokens_avg=("tokens", "mean"),
          )
          .reset_index()
          .sort_values(["partei", "land", "jahr"], kind="mergesort")
    )
    ppy_path = os.path.join(OUT_DIR, "corpus_by_party_year.tsv")
    by_party_year.to_csv(ppy_path, sep="\t", index=False)
    print(f"✅ Aggregation Partei×Jahr gespeichert: {ppy_path}")

    # Aggregation: Partei x Land (gesamt) mit Jahrspanne und Jahr-Liste
    def years_list(g):
        yrs = sorted([int(y) for y in g.dropna().unique().tolist()])
        return ",".join(map(str, yrs)) if yrs else ""

    agg_party = (
        df.groupby(["partei", "land"], dropna=False)
          .agg(
              docs=("text", "size"),
              tokens_sum=("tokens", "sum"),
              tokens_avg=("tokens", "mean"),
              jahr_min=("jahr", "min"),
              jahr_max=("jahr", "max"),
          )
          .reset_index()
    )
    # Jahre-Liste separat berechnen (wegen dropna)
    years_per_group = (
        df.groupby(["partei", "land"], dropna=False)["jahr"].apply(years_list).reset_index(name="jahre")
    )
    agg_party = agg_party.merge(years_per_group, on=["partei", "land"], how="left")
    agg_party = agg_party.sort_values(["partei", "land"], kind="mergesort")

    p_path = os.path.join(OUT_DIR, "corpus_by_party.tsv")
    agg_party.to_csv(p_path, sep="\t", index=False)
    print(f"✅ Aggregation Partei gesamt gespeichert: {p_path}")

    # Kurze Übersicht in der Konsole
    print("\nKurze Übersicht (Top 10 Parteien nach Tokens):")
    preview = (
        agg_party.sort_values("tokens_sum", ascending=False)
                 .head(20)
                 [["partei", "land", "docs", "tokens_sum", "jahr_min", "jahr_max", "jahre"]]
    )
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()

