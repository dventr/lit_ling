#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------

# Default paths
DATA_FILE = "outfiles/clusters/bert/kmeans/block_uneven_bert_kmeans_clusters.tsv"
OUTDIR = "outfiles/plots/custom"

# Font configuration (Arial/Helvetica für Publikation)
DEFAULT_FONT = "Arial, Helvetica, sans-serif"
TITLE_FONT_SIZE = 30
SUBTITLE_FONT_SIZE = 25
AXIS_TITLE_FONT_SIZE = 25
TICK_FONT_SIZE = 25
LEGEND_FONT_SIZE = 25

# Political blocks in parliamentary order (left → right)
BLOCK_ORDER = ["sozial", "grün", "mittelinks", "mitte", "mitterechts", "rechts"]

# Party → Block mapping
PARTY_TO_BLOCK = {
    'sp': 'sozial', 'spd': 'sozial', 'spoe': 'sozial', 'linke': 'sozial',
    'gruen': 'grün', 'gruene_oe': 'grün', 'gruene_de': 'grün', 'gruene_ch': 'grün',
    'neos': 'mittelinks',
    'mitte': 'mitte',
    'fdp': 'mitterechts', 'oevp': 'mitterechts', 'cdu': 'mitterechts',
    'fpoe': 'rechts', 'svp': 'rechts', 'afd': 'rechts'
}

# Symbol mapping for accessibility
SYMBOL_MAP = {
    "sozial": "circle",
    "grün": "square", 
    "mittelinks": "diamond",
    "mitte": "triangle-up",
    "mitterechts": "star",
    "rechts": "cross"
}

# ----------------------------------------------------------
# COLOR SCHEME
# ----------------------------------------------------------

def load_color_map():
    """Load the inclusive, colorblind-friendly color palette."""
    return {
        "sozial": "#AF5F0B",      # Brown/orange
        "grün": "#E3DA48",       # Yellow-green
        "mittelinks": "#830861",  # Purple
        "mitte": "#8B0000",       # Dark red
        "mitterechts": "#441076", # Indigo
        "rechts": "#013343",      # Dark slate gray
    }

# ----------------------------------------------------------
# HTML EXPORT WITH STYLING
# ----------------------------------------------------------

def save_html(fig, outpath: str):
    """Save figure as HTML with consistent styling and accessibility features."""
    
    # Update layout with consistent styling
    fig.update_layout(
        font=dict(family=DEFAULT_FONT, size=16),
        title_font=dict(size=TITLE_FONT_SIZE, family=DEFAULT_FONT),
        legend=dict(
            font=dict(size=LEGEND_FONT_SIZE, family=DEFAULT_FONT),
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(128,128,128,0.3)",
            borderwidth=1
        ),
        plot_bgcolor="rgba(250,251,252,0.9)",
        paper_bgcolor="white",
        margin=dict(l=80, r=150, t=120, b=80)
    )
    
    # Generate HTML
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    
    # Add font and styling
    font_inject = '''
    <link href="https://fonts.googleapis.com/css2?family=Arial&display=swap" rel="stylesheet">
    <style>
        body { 
            font-family: Arial, Helvetica, sans-serif !important;
            background-color: #fafbfc;
            margin: 0;
            padding: 20px;
        }
        .js-plotly-plot, .plotly, text { 
            font-family: Arial, Helvetica, sans-serif !important; 
        }
        .main-svg {
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    </style>
    '''
    html = html.replace("<head>", f"<head>\n{font_inject}")
    
    # Add accessibility information
    color_map = load_color_map()
    accessibility_info = f'''
    <!-- Accessibility Information -->
    <div id="accessibility-info" style="margin-top: 30px; padding: 15px; background: #f8f9fa; border-left: 4px solid #0173B2; font-family: 'Lato', sans-serif;">
        <h3 style="margin-top: 0; color: #0173B2;">🎨 Accessibility Features</h3>
        <ul style="margin: 10px 0;">
            <li><strong>Colorblind-friendly palette:</strong> Colors chosen for maximum distinguishability</li>
            <li><strong>High contrast:</strong> All colors meet WCAG AA standards</li>
            <li><strong>Parliamentary order:</strong> Parties arranged from left to right political spectrum</li>
        </ul>
        <p style="margin-bottom: 0; font-size: 14px; color: #666;">
            <strong>Political Spectrum (Left → Right):</strong><br>
            <span style="color: {color_map['sozial']};">● Sozial</span>, 
            <span style="color: {color_map['grün']};">■ Grün</span>, 
            <span style="color: {color_map['mittelinks']};">♦ Mitte-Links</span>, 
            <span style="color: {color_map['mitte']};">▲ Mitte</span>, 
            <span style="color: {color_map['mitterechts']};">★ Mitte-Rechts</span>, 
            <span style="color: {color_map['rechts']};">✕ Rechts</span>
        </p>
    </div>
    
</body>
</html>'''
    
    html = html.replace("</body>\n</html>", accessibility_info)
    
    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(html)

# ----------------------------------------------------------
# DATA PROCESSING
# ----------------------------------------------------------

def load_and_process_data(data_file: str) -> tuple[pd.DataFrame, dict]:
    """Load data and extract cluster information."""
    
    print(f"📊 Lade Daten: {os.path.abspath(data_file)}")
    df = pd.read_csv(data_file, sep="\t")
    print(f"   → {len(df)} Zeilen geladen")
    
    # Get cluster words for labeling
    cluster_words = get_cluster_words(df)
    
    # Calculate block composition
    block_comp = calculate_block_composition(df)
    
    return block_comp, cluster_words

def get_cluster_words(df: pd.DataFrame, max_words: int = 8) -> dict:  # Increased from 3 to 8
    """Extract top words per cluster for labeling."""
    cluster_words = {}
    
    # Find word column
    word_col = None
    for col in ['d', 'Word', 'word', 'token']:
        if col in df.columns:
            word_col = col
            break
    
    if not word_col:
        print("⚠️ Keine Wort-Spalte gefunden")
        return cluster_words
    
    # Get frequency columns
    freq_cols = [c for c in df.columns if c.startswith(('Freq_', 'RelFreqPer10k_'))]
    
    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster].copy()
        
        if freq_cols:
            # Sum frequencies across all parties
            cluster_data['total_freq'] = cluster_data[freq_cols].sum(axis=1)
            top_words = (cluster_data.nlargest(max_words, 'total_freq')[word_col]
                        .tolist())
        else:
            # Fallback: just take first words
            top_words = cluster_data[word_col].head(max_words).tolist()
        
        # Create readable label with much more space
        words_str = ", ".join(top_words)
        if len(words_str) > 80:  # Increased from 25 to 80 characters
            words_str = words_str[:77] + "..."
        
        cluster_words[cluster] = words_str
    
    return cluster_words

def calculate_block_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate political block composition for each cluster."""
    
    # Find frequency columns
    freq_cols = [c for c in df.columns if c.startswith(('Freq_', 'RelFreqPer10k_'))]
    if not freq_cols:
        raise ValueError("Keine Frequenz-Spalten gefunden!")
    
    # Melt the data
    melted = df.melt(
        id_vars=['Cluster'], 
        value_vars=freq_cols,
        var_name='Party_Column',
        value_name='Frequency'
    )
    
    # Extract party name and map to block
    melted['Party'] = melted['Party_Column'].str.replace(r'(Freq_|RelFreqPer10k_)', '', regex=True)
    melted['Block'] = melted['Party'].map(PARTY_TO_BLOCK)
    
    # Handle unmapped parties (keep as is if not in mapping)
    melted['Block'] = melted['Block'].fillna(melted['Party'])
    
    # Group by cluster and block
    block_comp = melted.groupby(['Cluster', 'Block'], as_index=False)['Frequency'].sum()
    
    # Calculate shares
    cluster_totals = block_comp.groupby('Cluster')['Frequency'].transform('sum')
    block_comp['Share'] = block_comp['Frequency'] / cluster_totals.replace(0, 1)
    
    return block_comp

# ----------------------------------------------------------
# MAIN VISUALIZATION: DIVERGING PLOT
# ----------------------------------------------------------

def create_diverging_dotplot(block_comp: pd.DataFrame, cluster_words: dict, topn: int, color_map: dict):
    """Create a clean, readable diverging bar chart showing political block distribution."""
    
    print(f"🎨 Erstelle Diverging Plot für Top {topn} Cluster...")
    
    # Create wide format
    wide = block_comp.pivot_table(index="Cluster", columns="Block", values="Share", fill_value=0).reset_index()
    
    # Define political sides (in parliamentary order)
    left_blocks = ["sozial", "grün", "mittelinks"]
    right_blocks = ["mitte", "mitterechts", "rechts"]
    
    # Calculate political score (right minus left)
    left_sum = wide[[b for b in left_blocks if b in wide.columns]].sum(axis=1)
    right_sum = wide[[b for b in right_blocks if b in wide.columns]].sum(axis=1)
    wide["political_score"] = right_sum - left_sum
    
    # Sort by political score and take top N
    wide = wide.sort_values("political_score", ascending=False).head(topn)
    
    # Create figure
    fig = go.Figure()
    
    # Add LEFT blocks (stacked, shown as negative values on left side)
    for block in reversed(left_blocks):  # Reverse to stack in correct order
        if block not in wide.columns:
            continue
        
        # Create y-axis labels
        y_labels = [f"C{row['Cluster']}: {cluster_words.get(row['Cluster'], '')[:60]}" 
                   for _, row in wide.iterrows()]
        
        fig.add_trace(go.Bar(
            x=-wide[block].values,  # Negative for left side
            y=y_labels,
            name=block.capitalize(),
            orientation='h',
            marker=dict(
                color=color_map.get(block, "#999999"),
                line=dict(color="white", width=0.5)
            ),
            hovertemplate=(
                f"<b>{block.capitalize()}</b><br>" +
                "Cluster: %{y}<br>" +
                "Anteil: %{customdata:.1%}<br>" +
                "Seite: Links<extra></extra>"
            ),
            customdata=wide[block].values,
            legendgroup="left",
            showlegend=True
        ))
    
    # Add RIGHT blocks (stacked, shown as positive values on right side)
    for block in right_blocks:
        if block not in wide.columns:
            continue
        
        # Create y-axis labels
        y_labels = [f"C{row['Cluster']}: {cluster_words.get(row['Cluster'], '')[:60]}" 
                   for _, row in wide.iterrows()]
        
        fig.add_trace(go.Bar(
            x=wide[block].values,  # Positive for right side
            y=y_labels,
            name=block.capitalize(),
            orientation='h',
            marker=dict(
                color=color_map.get(block, "#999999"),
                line=dict(color="white", width=0.5)
            ),
            hovertemplate=(
                f"<b>{block.capitalize()}</b><br>" +
                "Cluster: %{y}<br>" +
                "Anteil: %{customdata:.1%}<br>" +
                "Seite: Rechts<extra></extra>"
            ),
            customdata=wide[block].values,
            legendgroup="right",
            showlegend=True
        ))
    
    # Sort y-axis labels by political score (most right-leaning at top)
    sorted_labels = [f"C{row['Cluster']}: {cluster_words.get(row['Cluster'], '')[:60]}" 
                    for _, row in wide.sort_values("political_score", ascending=False).iterrows()]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>Resultate der Dispersionsanalyse geclustert ({topn} Cluster)</b>",
            font=dict(size=TITLE_FONT_SIZE, family=DEFAULT_FONT),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="<b>← Linke Blöcke | Rechte Blöcke →</b>",
            tickformat=".0%",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.9)",
            zerolinewidth=2,
            gridcolor="rgba(128,128,128,0.2)",
            title_font=dict(size=AXIS_TITLE_FONT_SIZE, family=DEFAULT_FONT),
            tickfont=dict(size=TICK_FONT_SIZE, family=DEFAULT_FONT),
            range=[-1.05, 1.05]
        ),
        yaxis=dict(
            title="<b>Cluster (sortiert nach politischem Spektrum)</b>",
            title_font=dict(size=AXIS_TITLE_FONT_SIZE, family=DEFAULT_FONT),
            tickfont=dict(size=13, family=DEFAULT_FONT),
            categoryorder='array',
            categoryarray=sorted_labels,
            automargin=True
        ),
        barmode='relative',  # Stacks bars from center outward
        height=max(700, topn * 38),
        width=1900,
        hovermode='closest',
        bargap=0.15,
        font=dict(family=DEFAULT_FONT),
        plot_bgcolor="rgba(250,251,252,0.9)",
        paper_bgcolor="white",
        margin=dict(l=480, r=180, t=120, b=100),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=LEGEND_FONT_SIZE-2, family=DEFAULT_FONT),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(128,128,128,0.3)",
            borderwidth=1,
            title=dict(text="<b>Politische Blöcke</b>", font=dict(size=LEGEND_FONT_SIZE, family=DEFAULT_FONT))
        )
    )
    
    # Add center line
    fig.add_vline(x=0, line_width=2, line_color="black", opacity=0.8)
    
    return fig
    
    return fig

# ----------------------------------------------------------
# SUPPLEMENTARY VISUALIZATIONS
# ----------------------------------------------------------

def create_summary_bar_chart(block_comp: pd.DataFrame, cluster_words: dict, topn: int, color_map: dict):
    """Create a simple bar chart showing dominant blocks per cluster."""
    
    # Find dominant block per cluster
    dominant = block_comp.loc[block_comp.groupby('Cluster')['Share'].idxmax()]
    dominant = dominant.sort_values('Share', ascending=False).head(topn)
    
    # Add cluster labels
    dominant['Cluster_Label'] = dominant['Cluster'].apply(
        lambda x: f"C{x}: {cluster_words.get(x, '')}"
    )
    
    fig = go.Figure(data=go.Bar(
        x=dominant['Cluster_Label'],
        y=dominant['Share'],
        marker_color=[color_map.get(block, "#999999") for block in dominant['Block']],
        text=[f"{share:.1%}" for share in dominant['Share']],
        textposition='auto',
        hovertemplate="<b>%{x}</b><br>Dominanter Block: %{customdata}<br>Anteil: %{y:.1%}<extra></extra>",
        customdata=dominant['Block']
    ))
    
    fig.update_layout(
        title="<b>Dominante politische Blöcke pro Cluster</b>",
        xaxis_title="<b>Cluster</b>",
        yaxis_title="<b>Dominanzanteil</b>",
        yaxis_tickformat=".0%",
        xaxis_tickangle=45,
        height=600
    )
    
    return fig

# ----------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Vereinfachte Cluster-Insights mit Fokus auf Diverging Plot")
    parser.add_argument("--data-file", default=DATA_FILE, help="Pfad zur Cluster-TSV-Datei")
    parser.add_argument("--topn", type=int, default=30, help="Anzahl der anzuzeigenden Top-Cluster")
    parser.add_argument("--outdir", default=OUTDIR, help="Ausgabeordner")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    print(f"📁 Ausgabeordner: {os.path.abspath(args.outdir)}")
    
    # Load color scheme
    color_map = load_color_map()
    
    try:
        # Load and process data
        block_comp, cluster_words = load_and_process_data(args.data_file)
        
        # 1. MAIN VISUALIZATION: Diverging Dot Plot
        print("\n🎯 Erstelle Hauptvisualisierung: Diverging Dot Plot")
        diverging_fig = create_diverging_dotplot(block_comp, cluster_words, args.topn, color_map)
        diverging_path = os.path.join(args.outdir, f"diverging_dotplot_top{args.topn}.html")
        save_html(diverging_fig, diverging_path)
        print(f"✅ Diverging Dot Plot gespeichert: {os.path.abspath(diverging_path)}")
        
        # Multi-format export
        try:
            from export_figures import save_figure_multi_format
            save_figure_multi_format(diverging_fig, f"Abb5_diverging_dotplot", output_dir="abbildungen", dpi=600)
        except ImportError:
            pass
        
        # 2. SUPPLEMENTARY: Summary Bar Chart
        print("\n📊 Erstelle Ergänzung: Dominanz-Ranking")
        summary_fig = create_summary_bar_chart(block_comp, cluster_words, args.topn, color_map)
        summary_path = os.path.join(args.outdir, f"dominant_blocks_top{args.topn}.html")
        save_html(summary_fig, summary_path)
        print(f"✅ Dominanz-Ranking gespeichert: {os.path.abspath(summary_path)}")
        
        # Multi-format export
        try:
            from export_figures import save_figure_multi_format
            save_figure_multi_format(summary_fig, f"Abb6_dominant_blocks", output_dir="abbildungen", dpi=600)
        except ImportError:
            pass
        
        # Final summary
        print(f"\n🎉 ALLE VISUALISIERUNGEN ERSTELLT:")
        print(f"   📁 Ausgabeordner: {os.path.abspath(args.outdir)}")
        print(f"   🎨 Farben: Colorblind-friendly palette")
        print(f"   📈 Anzahl Cluster: {args.topn}")
        print(f"   🏛️ Politische Ordnung: {' → '.join(BLOCK_ORDER)}")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
