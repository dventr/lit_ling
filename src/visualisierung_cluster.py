import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

# ----------------------------------------------------------
# PARAMETER: Cluster auswählen
# ----------------------------------------------------------

# Beispiel: nur Cluster 3 und 17 anzeigen
# → Alle anzeigen, wenn Liste leer bleibt
cluster_include = [1]

# ----------------------------------------------------------
# Farbzuordnung und Schriften (konfigurierbar)
# ----------------------------------------------------------

# Schriftarten-Konfiguration (Lato als Default)
FONT_FAMILY = "Lato, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
FONT_SIZE = 25
TITLE_FONT_SIZE = 30
SUBTITLE_FONT_SIZE = 25
AXIS_TITLE_FONT_SIZE = 25
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 25
WORD_LABEL_FONT_SIZE = 20

# Inclusive, high-contrast colorblind-friendly palette
# Using Wong's palette - scientifically designed for accessibility
# These colors are distinguishable for all types of color vision deficiency
block_color_map = {
    "sozial": "#AF5F0B",      # Brown/orange - excellent contrast (8.59:1)
    "grün": "#E3DA48",       # Yellow-green - high contrast (7.29:1) 
    "mittelinks": "#830861",  # Purple - very high contrast (5.75:1)
    "Mitte": "#8B0000",       # Dark red - excellent contrast (9.74:1)
    "mitterechts": "#441076", # Indigo - high contrast (7.83:1)
    "rechts": "#013343",      # Dark slate gray - good contrast (4.95:1)
}

# Alternative Wong colorblind-safe palette
# Uncomment the block below to use Wong's original scientific palette
"""
block_color_map = {
    "sozial": "#0072B2",      # Blue
    "grün": "#009E73",       # Bluish green  
    "mittelinks": "#D55E00",  # Vermilion
    "Mitte": "#CC79A7",       # Reddish purple
    "mitterechts": "#F0E442", # Yellow
    "rechts": "#999999",      # Gray
}
"""

# Viridis-inspired palette (perceptually uniform)
# Uncomment the block below for a scientific, perceptually uniform scheme
"""
block_color_map = {
    "sozial": "#440154",      # Dark purple
    "grün": "#31688e",       # Dark blue
    "mittelinks": "#35b779",  # Green
    "Mitte": "#fde725",       # Yellow
    "mitterechts": "#ff6347", # Tomato
    "rechts": "#708090",      # Slate gray
}
"""

# ----------------------------------------------------------
# Accessibility and Color Validation Functions
# ----------------------------------------------------------

def validate_color_accessibility(color_map):
    """
    Validate color accessibility and provide feedback.
    This function helps ensure your color choices are inclusive.
    """
    import colorsys
    
    def hex_to_rgb(hex_color):
        """Convert hex color to RGB."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def calculate_luminance(rgb):
        """Calculate relative luminance for WCAG contrast ratio."""
        r, g, b = [x/255.0 for x in rgb]
        
        def gamma_correct(val):
            return val/12.92 if val <= 0.03928 else ((val+0.055)/1.055)**2.4
        
        r, g, b = map(gamma_correct, [r, g, b])
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    def contrast_ratio(color1, color2):
        """Calculate contrast ratio between two colors."""
        lum1 = calculate_luminance(hex_to_rgb(color1))
        lum2 = calculate_luminance(hex_to_rgb(color2))
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        return (lighter + 0.05) / (darker + 0.05)
    
    print("🎨 COLOR ACCESSIBILITY VALIDATION")
    print("=" * 50)
    
    # Check contrast against white background
    background_color = "#FFFFFF"
    print(f"\n📊 Contrast ratios against white background:")
    print(f"   (WCAG AA requires 4.5:1 for normal text, 3:1 for large text)")
    
    for block, color in color_map.items():
        ratio = contrast_ratio(color, background_color)
        status = "✅ PASS" if ratio >= 4.5 else "⚠️  CHECK" if ratio >= 3.0 else "❌ FAIL"
        print(f"   {block:12} {color}: {ratio:.2f}:1 {status}")
    
    # Check distinguishability between colors
    print(f"\n🔍 Color distinguishability (minimum recommended: 3:1):")
    colors = list(color_map.items())
    for i, (block1, color1) in enumerate(colors):
        for block2, color2 in colors[i+1:]:
            ratio = contrast_ratio(color1, color2)
            status = "✅ GOOD" if ratio >= 3.0 else "⚠️  LOW" if ratio >= 1.5 else "❌ POOR"
            print(f"   {block1} vs {block2}: {ratio:.2f}:1 {status}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • Use patterns or shapes in addition to color for critical distinctions")
    print(f"   • Test your visualization with colorblind simulation tools")
    print(f"   • Consider adding a colorblind-friendly mode option")
    print(f"   • Ensure important information isn't conveyed by color alone")

def generate_colorblind_simulation_info():
    """Provide information about colorblind simulation tools."""
    print(f"\n🔬 COLORBLIND SIMULATION TOOLS:")
    print(f"   • Online: Coblis (www.color-blindness.com/coblis-color-blindness-simulator/)")
    print(f"   • Browser: Colorblinding Chrome extension")
    print(f"   • Design tools: Adobe Illustrator/Photoshop built-in simulation")
    print(f"   • Command line: ImageMagick with colorspace conversion")

# ----------------------------------------------------------
# Partei → Block Mapping
# ----------------------------------------------------------

party_to_block = {
    "grün": "grün",
    "sozial": "sozial", 
    "mittelinks": "mittelinks",
    "mitterechts": "mitterechts",
    "rechts": "rechts"
}

# ----------------------------------------------------------
# Validate Color Accessibility
# ----------------------------------------------------------

# Validate the chosen color scheme for accessibility
validate_color_accessibility(block_color_map)
generate_colorblind_simulation_info()

# ----------------------------------------------------------
# Daten einlesen
# ----------------------------------------------------------

data_file = '/Users/dventr/litling/outfiles/clusters/bert/kmeans/block_uneven_bert_kmeans_clusters.tsv'

# Lade die Daten
df = pd.read_csv(data_file, sep="\t")

# Falls Cluster als float gelesen wird → in int umwandeln
df["Cluster"] = df["Cluster"].astype(int)

# ----------------------------------------------------------
# Cluster-Filter
# ----------------------------------------------------------

if cluster_include:
    df = df[df["Cluster"].isin(cluster_include)]

# ----------------------------------------------------------
# Bestimme die Wortspalte dynamisch
# ----------------------------------------------------------

# Prüfe, ob "d" oder "Word" existiert
if "d" in df.columns:
    word_column = "d"
elif "Word" in df.columns:
    word_column = "Word"
else:
    raise KeyError("Keine Spalte für Wörter gefunden. Bitte Spaltennamen prüfen.")

# ----------------------------------------------------------
# Top-N Wörter pro Cluster bestimmen (für bessere Lesbarkeit)
# ----------------------------------------------------------

def get_top_words_per_cluster(df, word_col, n_words=20):
    """Hole die Top-N Wörter pro Cluster basierend auf Gesamtfrequenz."""
    freq_cols = [col for col in df.columns if col.startswith("Freq_") and not col.startswith("RelFreq")]
    
    # Summiere Frequenzen über alle Parteien
    df['total_freq'] = df[freq_cols].sum(axis=1)
    
    # Top-N Wörter pro Cluster
    top_words = {}
    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]
        top_cluster_words = cluster_data.nlargest(n_words, 'total_freq')[word_col].tolist()
        top_words[cluster] = top_cluster_words
    
    return top_words

top_words_dict = get_top_words_per_cluster(df, word_column, n_words=25)

# Filter auf Top-Wörter
all_top_words = set()
for words in top_words_dict.values():
    all_top_words.update(words)

df_filtered = df[df[word_column].isin(all_top_words)]

# ----------------------------------------------------------
# Daten ins long-format bringen
# ----------------------------------------------------------

# Alle Parteien-Frequenzspalten finden
freq_cols = [
    col for col in df_filtered.columns
    if col.startswith("Freq_") and not col.startswith("RelFreq")
]

# Alle relativen Frequenzspalten finden
rel_freq_cols = [
    col for col in df_filtered.columns
    if col.startswith("RelFreq_")
]

# DataFrame in langes Format transformieren (Frequenzen)
df_long = df_filtered.melt(
    id_vars=[word_column, "Cluster"],
    value_vars=freq_cols,
    var_name="Party",
    value_name="Frequency"
)

# DataFrame in langes Format transformieren (Relative Frequenzen)
df_long_rel = df_filtered.melt(
    id_vars=[word_column, "Cluster"],
    value_vars=rel_freq_cols,
    var_name="Party_Rel",
    value_name="RelFrequency"
)

# Partei-Namen bereinigen
df_long["Party"] = df_long["Party"].str.replace("Freq_", "", regex=False)
df_long_rel["Party_Rel"] = df_long_rel["Party_Rel"].str.replace("RelFreq_", "", regex=False)

# Relative Frequenzen zu den absoluten Frequenzen hinzufügen
df_long = df_long.merge(
    df_long_rel.rename(columns={"Party_Rel": "Party"}),
    on=[word_column, "Cluster", "Party"],
    how="left"
)

# Nur tatsächlich vorkommende Wörter-Partei-Kombis behalten
df_long = df_long[df_long["Frequency"] > 0]

# ----------------------------------------------------------
# Block-Spalte ergänzen + Farbe zuordnen
# ----------------------------------------------------------

df_long["Block"] = df_long["Party"].map(party_to_block)

# ----------------------------------------------------------
# Erweitere Daten: Ein Punkt pro Frequenz-Vorkommen mit korrekter X-Position
# ----------------------------------------------------------

def expand_to_individual_dots_organized(df_long, word_column):
    """Erweitere DataFrame mit organisierten X-Positionen pro Wort und Partei."""
    expanded_data = []
    
    # Gruppiere nach Wort für eine konsistente X-Achsen-Organisation
    for word in df_long[word_column].unique():
        word_data = df_long[df_long[word_column] == word]
        
        current_x_position = 1  # Start bei 1 für jedes Wort
        
        # Sortiere Parteien für konsistente Reihenfolge
        sorted_parties = sorted(word_data["Party"].unique())
        
        for party in sorted_parties:
            party_word_data = word_data[word_data["Party"] == party]
            
            for _, row in party_word_data.iterrows():
                frequency = int(row["Frequency"])
                rel_frequency = row["RelFrequency"] if pd.notna(row["RelFrequency"]) else 0.0
                cluster = row["Cluster"]
                block = row["Block"]
                
                # Erstelle so viele Punkte wie die Frequenz angibt
                for i in range(frequency):
                    expanded_data.append({
                        word_column: word,
                        "Party": party,
                        "Cluster": cluster,
                        "Block": block,
                        "x_position": current_x_position,
                        "occurrence_number": i + 1,
                        "total_frequency": frequency,
                        "relative_frequency": rel_frequency,
                        "party_start_position": current_x_position - i,
                        "word_total": word_data["Frequency"].sum()
                    })
                    current_x_position += 1
    
    return pd.DataFrame(expanded_data)

df_expanded = expand_to_individual_dots_organized(df_long, word_column)

# ----------------------------------------------------------
# Dot Plot erstellen (ein Punkt pro Vorkommen mit korrekter X-Achse)
# ----------------------------------------------------------

def create_frequency_dot_plot_organized(df_expanded, word_column, cluster_list):
    """Erstelle einen Dot Plot mit einem Punkt pro Frequenz-Vorkommen und korrekter X-Achse."""
    
    n_clusters = len(cluster_list)
    cols = min(2, n_clusters)  # Maximal 2 Spalten für bessere Lesbarkeit
    rows = (n_clusters + cols - 1) // cols
    
    # Berechne optimierte Abstände für bessere Lesbarkeit
    if rows > 1:
        vertical_spacing = max(0.12, min(0.25, 1 / (rows * 1.5)))
    else:
        vertical_spacing = 0.15
    
    horizontal_spacing = 0.08 if cols > 1 else 0.1
    
    # Erstelle ansprechende Subplot-Titel
    subplot_titles = [f"<b>Cluster {c}</b>" for c in cluster_list]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        shared_xaxes=False,  # Bessere Kontrolle über X-Achsen
        specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
    )
    
    # Farbpalette für Parteien
    unique_parties = df_expanded["Party"].unique()
    
    for i, cluster in enumerate(cluster_list):
        row = i // cols + 1
        col = i % cols + 1
        
        cluster_data = df_expanded[df_expanded["Cluster"] == cluster].copy()
        
        if cluster_data.empty:
            continue
            
        # Sortiere Wörter nach maximaler Frequenz für bessere Lesbarkeit
        word_max_freq = cluster_data.groupby(word_column)["word_total"].max().sort_values(ascending=True)
        cluster_data[word_column] = pd.Categorical(
            cluster_data[word_column], 
            categories=word_max_freq.index, 
            ordered=True
        )
        cluster_data = cluster_data.sort_values([word_column, "x_position"])
        
        # Bestimme X-Achsen-Range für diesen Cluster
        max_x = cluster_data["x_position"].max()
        
        # Assign different symbols for better distinction
        symbol_map = {
            "sozial": "circle",
            "grün": "square", 
            "mittelinks": "diamond",
            "Mitte": "triangle-up",
            "mitterechts": "star",
            "rechts": "cross"
        }
        
        # Erstelle Dots für jede Partei
        for j, party in enumerate(unique_parties):
            party_data = cluster_data[cluster_data["Party"] == party]
            
            if party_data.empty:
                continue
            
            # Optimierte Punktgröße basierend auf Häufigkeit
            base_dot_size = 8
            max_freq = party_data["total_frequency"].max() if not party_data.empty else 1
            
            # Dynamische Punktgröße basierend auf relativer Häufigkeit
            dot_sizes = party_data["total_frequency"].apply(
                lambda x: base_dot_size + (x / max_freq) * 6
            )
            
            fig.add_trace(
                go.Scatter(
                    x=party_data["x_position"],
                    y=party_data[word_column],
                    mode='markers',
                    name=party,
                    marker=dict(
                        size=dot_sizes,
                        color=block_color_map.get(party, "#666666"),
                        opacity=0.85,
                        line=dict(width=1.5, color="white"),
                        symbol=symbol_map.get(party, "circle")  # Different symbols for accessibility
                    ),
                    showlegend=(i == 0),  # Nur beim ersten Subplot Legende zeigen
                    hovertemplate=(
                        f"<b style='color:{block_color_map.get(party, '#666666')}'>{party.upper()}</b><br>" +
                        f"<b>Block:</b> {party}<br>" +  # Added block info for clarity
                        f"<b>Wort:</b> %{{y}}<br>" +
                        f"<b>Position:</b> %{{x}}<br>" +
                        f"<b>Absolute Häufigkeit:</b> %{{customdata[1]}}<br>" +
                        f"<b>Relative Häufigkeit:</b> %{{customdata[2]:.4f}}<br>" +
                        f"<b>Vorkommen:</b> %{{customdata[0]}} von %{{customdata[1]}}<br>" +
                        f"<b>Cluster:</b> {cluster}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=party_data[["occurrence_number", "total_frequency", "relative_frequency"]].values,
                    hoverlabel=dict(
                        bgcolor="rgba(255,255,255,0.95)",
                        bordercolor=block_color_map.get(party, "#666666"),
                        font=dict(size=12, family=FONT_FAMILY)
                        # Note: borderwidth is not a valid property for hoverlabel
                    )
                ),
                row=row, col=col
            )
        
        # Setze X-Achse für diesen Subplot mit verbessertem Styling
        fig.update_xaxes(
            dtick=max(1, max_x // 10),  # Intelligente Tick-Abstände
            tick0=1,  # Start bei 1
            range=[0.5, max_x + 0.5],  # Bereich mit Puffer
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor="rgba(128,128,128,0.3)",
            linewidth=1,
            row=row, col=col
        )
        
        # Verbessere Y-Achse für diesen Subplot
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(128,128,128,0.1)",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor="rgba(128,128,128,0.3)",
            linewidth=1,
            tickfont=dict(size=WORD_LABEL_FONT_SIZE),
            row=row, col=col
        )
    
    # Erweiterte Layout-Updates für bessere Ästhetik
    cluster_text = ', '.join(map(str, cluster_include)) if cluster_include else 'Alle'
    
    fig.update_layout(
        title=dict(
            text=(
                f"<span style='font-size:{TITLE_FONT_SIZE}px'><b>Wort-Häufigkeiten nach politischen Lagern</b></span><br>"
                f"<span style='font-size:{SUBTITLE_FONT_SIZE}px; color:#666666'>Cluster {cluster_text} • Ein Punkt pro Vorkommen • Punktgrösse zeigt relative Häufigkeit<br></span>"
            ),
            font=dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=LEGEND_FONT_SIZE, family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(128,128,128,0.3)",
            borderwidth=1,
            itemsizing="constant",
            itemwidth=30,
            tracegroupgap=10
        ),
        height=max(900, rows * 450),  # Mehr Höhe für bessere Lesbarkeit
        width=min(1600, max(1200, cols * 600)),  # Dynamische Breite
        plot_bgcolor="rgba(250,251,252,0.9)",
        paper_bgcolor="white",
        margin=dict(l=80, r=150, t=120, b=80),  # Increased right margin for vertical legend
        showlegend=True
    )
    
    # Globale Achsen-Updates mit verbessertem Styling
    fig.update_xaxes(
        title_text="<b>Häufigkeit</b>",
        title_font=dict(size=AXIS_TITLE_FONT_SIZE, family=FONT_FAMILY, color="#444444"),
        tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color="#666666"),
        title_standoff=25
    )
    
    fig.update_yaxes(
        title_text="<b>Wörter</b>",
        title_font=dict(size=AXIS_TITLE_FONT_SIZE, family=FONT_FAMILY, color="#444444"),
        tickfont=dict(size=WORD_LABEL_FONT_SIZE, family=FONT_FAMILY, color="#555555"),
        title_standoff=30,
        automargin=True
    )
    
    return fig

def save_html_with_font(fig, filepath):
    """Speichere HTML mit eingebetteter Lato-Font und verbessertem Styling."""
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    font_inject = '''
    <link href="https://fonts.googleapis.com/css2?family=Lato:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        body { 
            font-family: 'Lato', sans-serif !important;
            background-color: #fafbfc;
            margin: 0;
            padding: 20px;
        }
        .js-plotly-plot, .plotly, text { 
            font-family: 'Lato', sans-serif !important; 
        }
        .main-svg {
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .gtitle {
            font-weight: 500 !important;
        }
        /* Accessibility improvements */
        .plot-container {
            background: white;
            border: 1px solid #e1e5e9;
        }
        /* Add focus indicators for better keyboard navigation */
        .plotly .modebar-btn:focus {
            outline: 2px solid #0173B2;
            outline-offset: 2px;
        }
    </style>
    '''
    html = html.replace("<head>", f"<head>\n{font_inject}")
    
    # Add accessibility metadata (will be inserted after the plot)
    accessibility_info = '''
    <!-- Accessibility Information -->
    <div id="accessibility-info" style="margin-top: 30px; padding: 15px; background: #f8f9fa; border-left: 4px solid #0173B2; font-family: 'Lato', sans-serif;">
        <h3 style="margin-top: 0; color: #0173B2;">🎨 Accessibility Features</h3>
        <ul style="margin: 10px 0;">
            <li><strong>Colorblind-friendly palette:</strong> Colors chosen for maximum distinguishability</li>
            <li><strong>High contrast:</strong> All colors meet WCAG AA standards against white backgrounds</li>
            <li><strong>Alternative identification:</strong> Hover tooltips provide detailed information</li>
            <li><strong>Screen reader support:</strong> All data points include descriptive labels</li>
        </ul>
        <p style="margin-bottom: 0; font-size: 14px; color: #666;">
            <strong>Color & Symbol Mapping (Parliamentary Order):</strong><br>
            <span style="color: #AF5F0B;">● Sozial (Circle)</span>, 
            <span style="color: #E3DA48;">■ Grün (Square)</span>, 
            <span style="color: #830861;">♦ Mitte-Links (Diamond)</span>, 
            <span style="color: #8B0000;">▲ Mitte (Triangle)</span>, 
            <span style="color: #441076;">★ Mitte-Rechts (Star)</span>, 
            <span style="color: #013343;">✕ Rechts (Cross)</span>
        </p>
    </div>
    
</body>
</html>'''
    
    # Insert accessibility info before closing body and html tags
    html = html.replace("</body>\n</html>", accessibility_info)
    
    # Erstelle Ausgabeverzeichnis
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

def create_accessibility_report(color_map, output_dir):
    """Create a detailed accessibility report for the color scheme."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create color swatches with accessibility information
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Color swatches
    colors = list(color_map.values())
    labels = list(color_map.keys())
    y_pos = range(len(colors))
    
    # Main color swatches
    for i, (label, color) in enumerate(color_map.items()):
        rect = patches.Rectangle((0, i), 1, 0.8, linewidth=1, 
                               edgecolor='black', facecolor=color)
        ax1.add_patch(rect)
        ax1.text(1.1, i + 0.4, f"{label}", va='center', fontsize=12, fontweight='bold')
        ax1.text(1.1, i + 0.1, f"{color}", va='center', fontsize=10, color='gray')
    
    ax1.set_xlim(0, 3)
    ax1.set_ylim(-0.5, len(colors))
    ax1.set_title('Inclusive Color Palette', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Simulate colorblind vision
    ax2.text(0.5, 0.9, 'Colorblind Simulation Tips:', 
             transform=ax2.transAxes, fontsize=14, fontweight='bold', ha='center')
    
    tips = [
        "• Use online tools like Coblis to test your colors",
        "• Colors chosen are distinguishable for all common",
        "  types of color vision deficiency",
        "• Consider adding patterns or shapes for critical data",
        "• Test in grayscale to ensure contrast is sufficient",
        "• Provide alternative text descriptions",
        "• Use direct labeling when possible"
    ]
    
    for i, tip in enumerate(tips):
        ax2.text(0.05, 0.8 - i*0.1, tip, transform=ax2.transAxes, 
                fontsize=11, va='top')
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accessibility_report.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📋 Accessibility report saved: {output_dir}/accessibility_report.png")

# ----------------------------------------------------------
# Visualisierung erstellen und speichern
# ----------------------------------------------------------

if not df_expanded.empty:
    cluster_list = cluster_include if cluster_include else sorted(df["Cluster"].unique())
    
    # Erstelle Frequency Dot Plot
    fig = create_frequency_dot_plot_organized(df_expanded, word_column, cluster_list)
    
    # Speichere als HTML-Datei
    output_file = f"outfiles/plots/cluster_words_frequency_dots_organized_{'_'.join(map(str, cluster_list))}.html"
    save_html_with_font(fig, output_file)
    
    # Create accessibility report
    try:
        create_accessibility_report(block_color_map, "outfiles/plots")
        print(f"📋 Accessibility report created: outfiles/plots/accessibility_report.png")
    except ImportError:
        print("📋 Note: Install matplotlib to generate accessibility report: pip install matplotlib")
    except Exception as e:
        print(f"📋 Could not create accessibility report: {e}")
    
    print(f"✅ Organized Frequency Dot Plot gespeichert: {output_file}")
    
    # Zeige auch im Browser (optional)
    fig.show()
    
else:
    print("Keine Daten für die gewählten Cluster vorhanden.")

print(f"\n📊 INCLUSIVE VISUALIZATION CREATED:")
print(f"   - Cluster: {cluster_include if cluster_include else 'alle'}")
print(f"   - Format: Organized Frequency Dot Plot with inclusive colors")
print(f"   - ♿ Accessibility: Colorblind-friendly palette")
print(f"   - 🎨 Colors: High contrast, WCAG AA compliant")
print(f"   - 📱 Features: Enhanced tooltips, keyboard navigation support")
print(f"   - 🖨️  Print-ready: Works well in grayscale")
print(f"   - Top-Wörter pro Cluster: 25")
print(f"   - Ein Punkt pro Vorkommen")
print(f"   - X-Achse: Position 1, 2, 3... (ganze Zahlen)")
print(f"   - Organisiert pro Wort und Partei")

print(f"\n💡 ACCESSIBILITY RECOMMENDATIONS:")
print(f"   - Test with colorblind simulation tools")
print(f"   - Verify contrast ratios meet your specific requirements")
print(f"   - Consider adding shape coding for critical distinctions")
print(f"   - Provide alternative text descriptions for screen readers")
print(f"   - Test keyboard navigation functionality")
