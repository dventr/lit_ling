from pathlib import Path
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import Counter, defaultdict
import os
import re

# ---------------- Config ----------------
DATA_DIR = Path("/Users/dventr/litling/outfiles_kwic_ling_LL")
FILE_PATTERN = "collocations_Migration_ideology_*.tsv"
OUT_DIR = Path("/Users/dventr/litling/viz_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fonts and sizes (Arial/Helvetica für Publikation)
FONT_FAMILY = "Arial, Helvetica, sans-serif"
FONT_SIZE = 25
TITLE_FONT_SIZE = 30
SUBTITLE_FONT_SIZE = 25
AXIS_TITLE_FONT_SIZE = 25
TICK_FONT_SIZE = 25
LEGEND_FONT_SIZE = 25
WORD_LABEL_FONT_SIZE = 25

# Inclusive, high-contrast colorblind-friendly palette
block_color_map = {
    "sozial": "#AF5F0B",
    "grün": "#E3DA48",
    "mittelinks": "#830861",
    "Mitte": "#8B0000",
    "mitterechts": "#441076",
    "rechts": "#013343",
    "other": "#666666",
}

symbol_map = {
    "sozial": "circle",
    "grün": "square",
    "mittelinks": "diamond",
    "Mitte": "triangle-up",
    "mitterechts": "star",
    "rechts": "x",
    "_default": "circle"
}

# ---------------- Stopwords (base + optional custom file) ----------------
STOPWORDS_FILE = Path("/Users/dventr/litling/stopwords_custom.txt")  # optional, one word per line
MIN_WORD_LEN = 2
DROP_TOKENS_WITH_DIGITS = True
DROP_PUNCT_ONLY = True

def load_stopwords() -> set:
    base = {
        # Articles/determiners
        "der","die","das","ein","eine","einer","einem","einen","eines","den","dem","des",
        "dies","dieser","diese","dieses","jene","jener","jenes","solch","solche","solcher",
        # Pronouns
        "ich","du","er","sie","es","wir","ihr","sie","mich","dich","ihn","uns","euch","ihnen",
        "mein","meine","meiner","meinem","meinen","meines","dein","deine","sein","seine","ihr","ihre",
        "man","jemand","niemand","wer","was","etwas","nichts","alle","einige","viele","wenige","mehrere",
        # Conjunctions/particles
        "und","oder","aber","doch","sondern","denn","sowie","sowohl","als","auch","entweder","weder","noch",
        "weil","da","dass","daß","ob","wenn","falls","damit","sodass","so","so dass","bevor","nachdem","während",
        "seit","seitdem","bis","trotzdem","jedoch","zwar","schon","nur","noch","auch","sehr","mehr","weniger",
        # Prepositions
        "in","im","ins","auf","an","am","ans","aus","bei","beim","mit","ohne","für","gegen","durch","um","über","unter",
        "vor","nach","zwischen","per","pro","je","vom","zum","zur","übers","übers","aufs","aufs","beider","beiden",
        # Auxiliaries/modals
        "sein","bin","bist","ist","sind","seid","war","waren","wird","werden","wurde","wurden","gewesen",
        "haben","habe","hast","hat","haben","hattet","hatten","gehabt",
        "können","kann","könnt","konnte","konnten","dürfen","darf","durfte","durften","müssen","muss","musste","mussten",
        "sollen","soll","sollte","sollten","wollen","will","wollte","wollten","mögen","mag","mochte","mochten",
        # Adverbs/misc
        "hier","da","dort","daher","darin","darum","wo","wohin","woher","wie","so","bereits","heute","gestern","morgen",
        "etwa","circa","ca","etc","bzw","bspw","z.b.","z.b","d.h.","d.h","u.a.","u.a","usw","u.v.m","uvm",
        # Common English spillovers
        "the","and","of","to","in","for","on","with","by",
    }
    # Load custom stopwords if present
    custom = set()
    if STOPWORDS_FILE.exists():
        with open(STOPWORDS_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                w = line.strip()
                if not w or w.startswith("#"):
                    continue
                custom.add(w.casefold())
    return {w.casefold() for w in base} | custom

STOPWORDS = load_stopwords()

def save_html_with_font(fig, filepath, title=None):
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    font_inject = '''
    <link href="https://fonts.googleapis.com/css2?family=Arial&display=swap" rel="stylesheet">
    <style>
        body { font-family: Arial, Helvetica, sans-serif !important; background-color: #fafbfc; margin: 0; padding: 20px; }
        .js-plotly-plot, .plotly, text { font-family: Arial, Helvetica, sans-serif !important; }
        .main-svg { border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .plot-container { background: white; border: 1px solid #e1e5e9; }
    </style>
    '''
    html = html.replace("<head>", f"<head>\n{font_inject}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

def save_multi_format(fig, base_name, fig_num):
    """Export figure in multiple formats for publication"""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from export_figures import save_figure_multi_format
        save_figure_multi_format(fig, f"Abb{fig_num}_{base_name}", output_dir="abbildungen", dpi=600)
    except ImportError:
        print(f"⚠️  Multi-format export not available, saving HTML only")

def list_name(p: Path) -> str:
    n = p.stem
    if n.startswith("collocations_"):
        n = n[len("collocations_"):]
    return n

def infer_block(name: str):
    s = name.lower()
    if "grün" in s or "gruen" in s or "grüne" in s or "gruene" in s:
        return "grün"
    if "sozial" in s or "spd" in s:
        return "sozial"
    if "mittelinks" in s or "mitte-links" in s or "mitte_links" in s or "center-left" in s:
        return "mittelinks"
    if s == "mitte" or ("mitte" in s and "links" not in s and "rechts" not in s):
        return "Mitte"
    if "mitterechts" in s or "mitte-rechts" in s or "center-right" in s:
        return "mitterechts"
    if "rechts" in s:
        return "rechts"
    return None

# ---------------- Load data ----------------
files = sorted(DATA_DIR.glob(FILE_PATTERN))
if not files:
    print(f"No files found at {DATA_DIR}/{FILE_PATTERN}.")
    sys.exit(1)
if len(files) < 2:
    print("Only one table found. Add more 'collocations_*.tsv' files for intersections to be meaningful.")

frames = []
filtered_out = []  # collect removed words for transparency
for f in files:
    try:
        df = pd.read_csv(f, sep="\t", dtype=str, on_bad_lines="skip")
    except TypeError:
        # pandas < 1.3
        df = pd.read_csv(f, sep="\t", dtype=str)
    except Exception as e:
        print(f"Failed to read {f}: {e}")
        continue

    df.columns = [str(c).strip() for c in df.columns]
    if "Word" not in df.columns:
        print(f"Skipping {f} (no 'Word' column).")
        continue

    df = df[df["Word"].notna()].copy()
    df["Word"] = df["Word"].astype(str).str.strip()
    df = df[df["Word"] != ""].drop_duplicates(subset=["Word"]).copy()
    # Stopword filtering (case-insensitive)
    wl = df["Word"].str.casefold()
    to_drop = pd.Series(False, index=df.index)
    to_drop |= wl.isin(STOPWORDS)
    if MIN_WORD_LEN > 0:
        to_drop |= df["Word"].str.len() < MIN_WORD_LEN
    if DROP_TOKENS_WITH_DIGITS:
        to_drop |= df["Word"].str.contains(r"\d", regex=True)
    if DROP_PUNCT_ONLY:
        to_drop |= df["Word"].str.fullmatch(r"[\W_]+", na=False)
    # Log removed
    if to_drop.any():
        filtered_out.extend([(list_name(f), w) for w in df.loc[to_drop, "Word"].tolist()])
    df = df.loc[~to_drop].copy()

    df["List"] = list_name(f)
    # keep score columns if present for later ranking
    keep_cols = [c for c in ["Word", "Log-Likelihood", "Observed collocate frequency"] if c in df.columns]
    frames.append(df[keep_cols + ["List"]])

if not frames:
    print("No valid tables loaded.")
    sys.exit(1)

all_df = pd.concat(frames, ignore_index=True)
all_df["Word"] = all_df["Word"].astype(str).str.strip()
all_df["List"] = all_df["List"].astype(str)
# parse numeric scores if available
for col in ["Log-Likelihood", "Observed collocate frequency"]:
    if col in all_df.columns:
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

lists = sorted(all_df["List"].unique().tolist())
words = sorted(all_df["Word"].unique().tolist())

# Presence matrix
presence = pd.DataFrame(0, index=words, columns=lists, dtype=int)
for ln, sub in all_df.groupby("List"):
    for w in sub["Word"].unique():
        presence.at[w, ln] = 1

# Name index so melt(id_vars="Word") works after reset_index()
presence.index.name = "Word"

presence_path = OUT_DIR / "presence_matrix.csv"
presence.to_csv(presence_path, index_label="Word")

# Save filtered words (if any) for reproducibility
if filtered_out:
    pd.DataFrame(filtered_out, columns=["List", "Word"])\
      .drop_duplicates()\
      .sort_values(["List","Word"])\
      .to_csv(OUT_DIR / "filtered_stopwords.csv", index=False)

# ---------------- 1) Unique vs Shared per list (stacked bars) ----------------
unique_mask = (presence.sum(axis=1) == 1)
per_list_unique = {ln: int((presence[ln] == 1).where(unique_mask, False).sum()) for ln in lists}
per_list_shared = {ln: int((presence[ln] == 1).where(~unique_mask, False).sum()) for ln in lists}

fig1 = go.Figure()
fig1.add_bar(
    x=lists, y=[per_list_shared[l] for l in lists],
    name="Shared", marker_color="#999999", hovertemplate="List: %{x}<br>Shared words: %{y}<extra></extra>"
)
fig1.add_bar(
    x=lists, y=[per_list_unique[l] for l in lists],
    name="Unique", marker_color="#111111", hovertemplate="List: %{x}<br>Unique words: %{y}<extra></extra>"
)
fig1.update_layout(
    barmode="stack",
    title=dict(
        text="<b>Unique vs. Shared Collocations per List</b>",
        font=dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE),
        x=0.5
    ),
    xaxis=dict(
        title="<b>List</b>",
        tickangle=45,
        titlefont=dict(size=AXIS_TITLE_FONT_SIZE, family=FONT_FAMILY),
        tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY)
    ),
    yaxis=dict(
        title="<b>Words</b>",
        titlefont=dict(size=AXIS_TITLE_FONT_SIZE, family=FONT_FAMILY),
        tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY)
    ),
    legend=dict(
        font=dict(size=LEGEND_FONT_SIZE, family=FONT_FAMILY),
        bgcolor="rgba(255,255,255,0.9)"
    ),
    font=dict(family=FONT_FAMILY, size=FONT_SIZE),
    plot_bgcolor="rgba(250,251,252,0.9)",
    paper_bgcolor="white",
    margin=dict(l=100, r=60, t=100, b=150),
    autosize=False,
    height=750,
    width=min(2400, max(1300, 100*len(lists)))
)
save_html_with_font(fig1, str(OUT_DIR / "list_unique_shared.html"))
save_multi_format(fig1, "list_unique_shared", 1)

# ---------------- 2) Dot-matrix of top shared words ----------------
n_lists_per_word = presence.sum(axis=1)
top_shared_words = (
    n_lists_per_word[n_lists_per_word >= 3]  # require common to ≥ 3 lists
    .sort_values(ascending=False)
    .head(50)
    .index.tolist()
)

if top_shared_words:
    # Long df for presence
    long_df = (
        presence.loc[top_shared_words]
        .reset_index()  # now has a 'Word' column because we named the index
        .melt(id_vars="Word", var_name="List", value_name="Present")
    )
    long_df = long_df[long_df["Present"] == 1]
    # Order words by number of lists then alphabetically
    order_counts = n_lists_per_word.loc[top_shared_words].sort_values(ascending=False)
    # Stable sort: count desc then word asc
    ordered_words = (order_counts.reset_index()
                     .rename(columns={0: "n", "Word": "Word"})
                     .sort_values(["Word"])
                     .set_index("Word")
                     .loc[order_counts.index].index.tolist())

    # Map list to block color/symbol
    long_df["Block"] = long_df["List"].apply(infer_block)
    long_df["Color"] = long_df["Block"].map(block_color_map).fillna("#444444")
    long_df["Symbol"] = long_df["Block"].map(symbol_map).fillna(symbol_map["_default"])

    # Build one trace per list (keeps legend concise)
    fig2 = go.Figure()
    for ln in lists:
        sub = long_df[long_df["List"] == ln]
        if sub.empty:
            continue
        block = infer_block(ln)
        color = block_color_map.get(block, "#444444")
        symbol = symbol_map.get(block, symbol_map["_default"])
        fig2.add_trace(go.Scatter(
            x=sub["List"],  # categorical single value repeated; we want column per list
            y=pd.Categorical(sub["Word"], categories=ordered_words, ordered=True),
            mode="markers",
            name=ln,
            marker=dict(color=color, size=16, symbol=symbol, line=dict(color="white", width=1)),
            text=sub["Word"],
            hovertemplate="<b>List:</b> %{x}<br><b>Word:</b> %{text}<extra></extra>",
            showlegend=True
        ))

    fig2.update_layout(
        title=dict(
            text="<b>Shared Words Across Lists (≥ 3 lists; Top 50)</b>",
            font=dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE),
            x=0.5
        ),
        xaxis=dict(
            title="<b>Kollokationslisten</b>",
            tickangle=45,
            titlefont=dict(size=AXIS_TITLE_FONT_SIZE, family=FONT_FAMILY),
            tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY)
        ),
        yaxis=dict(
            title="<b>Gemeinsamkeiten zwischen den Kollokationslisten ≥ 3 lists)</b>",
            categoryorder="array",
            categoryarray=ordered_words,
            titlefont=dict(size=AXIS_TITLE_FONT_SIZE, family=FONT_FAMILY),
            tickfont=dict(size=WORD_LABEL_FONT_SIZE, family=FONT_FAMILY)
        ),
        legend=dict(
            font=dict(size=LEGEND_FONT_SIZE, family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.9)"
        ),
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        plot_bgcolor="rgba(250,251,252,0.9)",
        paper_bgcolor="white",
        margin=dict(l=220, r=60, t=100, b=180),
        autosize=False,
        height=min(2000, max(1200, 24*len(ordered_words))),
        width=min(2600, max(1400, 110*len(lists)))
    )
    save_html_with_font(fig2, str(OUT_DIR / "shared_dotmatrix.html"))
    save_multi_format(fig2, "shared_dotmatrix", 2)

# ---------------- 3) Top membership intersections (bar chart + CSV) ----------
# Represent each word's membership as a sorted tuple of lists
memberships = []
for w, row in presence.iterrows():
    combo = tuple([c for c, v in row.items() if v == 1])
    if combo:
        memberships.append(tuple(sorted(combo)))

combo_counts = Counter(memberships)
combo_df = (
    pd.DataFrame(
        [( " • ".join(k), len(k), v) for k, v in combo_counts.items()],
        columns=["Combination", "Size", "Count"]
    )
    .sort_values(["Count", "Size"], ascending=[False, False])
)
combo_df.to_csv(OUT_DIR / "membership_combinations.csv", index=False)

top_k = min(30, len(combo_df))
if top_k > 0:
    top_df = combo_df.head(top_k).copy()
    # Plotly HTML instead of matplotlib PNG
    fig3 = go.Figure(go.Bar(
        x=top_df["Count"],
        y=top_df["Combination"],
        orientation="h",
        marker_color="#111111",
        hovertemplate="<b>%{y}</b><br>Words: %{x}<extra></extra>"
    ))
    fig3.update_layout(
        title=dict(
            text="<b>Top List Combinations (by number of words)</b>",
            font=dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE),
            x=0.5
        ),
        xaxis=dict(
            title="<b>Words</b>",
            titlefont=dict(size=AXIS_TITLE_FONT_SIZE, family=FONT_FAMILY),
            tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY)
        ),
        yaxis=dict(
            title="<b>List combination</b>",
            titlefont=dict(size=AXIS_TITLE_FONT_SIZE, family=FONT_FAMILY),
            tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY)
        ),
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        plot_bgcolor="rgba(250,251,252,0.9)",
        paper_bgcolor="white",
        margin=dict(l=320, r=70, t=110, b=100),
        autosize=False,
        height=min(2000, max(900, 32*top_k)),
        width=1800
    )
    save_html_with_font(fig3, str(OUT_DIR / "top_intersections.html"))
    save_multi_format(fig3, "top_intersections", 3)

# ---------------- 4) Singular (unique) words per list — Treemap (Top 10 per list) ----------------
# Build long-form dataframe of unique words
unique_mask = (presence.sum(axis=1) == 1)
uni_long = (
    presence.reset_index()
    .melt(id_vars="Word", var_name="List", value_name="Present")
)
uni_long = uni_long[(uni_long["Present"] == 1) & (uni_long["Word"].isin(presence.index[unique_mask]))].copy()

if not uni_long.empty:
    # score unique words per list: prefer Log-Likelihood, else Observed collocate frequency
    unique_pairs = uni_long[["Word", "List"]].drop_duplicates()
    score_cols = ["Log-Likelihood", "Observed collocate frequency"]
    avail_score_cols = [c for c in score_cols if c in all_df.columns]
    if avail_score_cols:
        uni_scored = unique_pairs.merge(all_df[["Word", "List"] + avail_score_cols], on=["Word", "List"], how="left")
        # compute unified score
        if "Log-Likelihood" in uni_scored.columns:
            score = uni_scored["Log-Likelihood"]
        else:
            score = pd.Series([pd.NA] * len(uni_scored))
        if "Observed collocate frequency" in uni_scored.columns:
            score = score.fillna(uni_scored["Observed collocate frequency"])
        uni_scored["score"] = pd.to_numeric(score, errors="coerce").fillna(-1e12)
    else:
        # no scoring columns available; fall back to alphabetical
        uni_scored = unique_pairs.copy()
        uni_scored["score"] = 0.0

    # take top 10 words per list (by score desc, then alphabetical)
    uni_scored = uni_scored.sort_values(["List", "score", "Word"], ascending=[True, False, True])
    uni_top10 = uni_scored.groupby("List", group_keys=False).head(10).copy()
    uni_top10["Block"] = uni_top10["List"].apply(infer_block).fillna("other")
    uni_top10["Value"] = 1

    # Order lists by number of unique words included (desc)
    uniq_counts = uni_top10.groupby("List")["Word"].nunique().sort_values(ascending=False)
    ordered_lists = uniq_counts.index.tolist()
    uni_top10["List"] = pd.Categorical(uni_top10["List"], categories=ordered_lists, ordered=True)

    # Create treemap: List (parent) -> Word (leaf), only top10 per list
    fig4 = px.treemap(
        uni_top10,
        path=["List", "Word"],
        values="Value",
        color="Block",
        color_discrete_map=block_color_map,
    )
    fig4.update_traces(
        textinfo="label",
        sort=False,
        tiling=dict(pad=2)
    )
    # Show full path in hover (List > Word)
    fig4.data[0].hovertemplate = "<b>%{currentPath}</b><extra></extra>"
    fig4.update_layout(
        title=dict(
            text="<b>Top 10 Unique Words per List</b>",
            font=dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE),
            x=0.5
        ),
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        legend=dict(font=dict(size=LEGEND_FONT_SIZE, family=FONT_FAMILY)),
        margin=dict(l=40, r=40, t=90, b=40),
        autosize=False,
        height=min(2400, max(900, 300 + 60*len(ordered_lists))),
        width=min(2600, max(1400, 100*len(ordered_lists))),
        paper_bgcolor="white",
        plot_bgcolor="rgba(250,251,252,0.9)",
    )
    save_html_with_font(fig4, str(OUT_DIR / "unique_words_treemap.html"))
    save_multi_format(fig4, "unique_words_treemap", 4)
else:
    print("No unique words found; unique_words_treemap.html not created.")

print(f"Done. Outputs saved in: {OUT_DIR}")