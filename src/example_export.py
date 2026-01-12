#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beispiel: Springer-konformer Export für eine Plotly-Figur
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import plotly.graph_objects as go
from export_figures import export_figure

# Beispiel-Figur erstellen
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[1, 4, 2, 3, 5],
    mode='markers+lines',
    name='Beispiel'
))

fig.update_layout(
    title="Beispiel-Abbildung",
    xaxis_title="X-Achse",
    yaxis_title="Y-Achse"
)

# Export in Springer-Formate
output_dir = Path.cwd() / "abbildungen"
files = export_figure(
    fig,
    basename="beispiel_abbildung",
    output_dir=str(output_dir),
    figure_number=99,  # Beispiel-Nummer
    width_mm=122,      # Zwei-Spalten-Breite (kleinformatig)
    height_mm=90,      # Angepasste Höhe
    save_html=True,
    save_eps=True,
    save_png=True
)

print("\n✅ Beispiel-Export abgeschlossen!")
print(f"Erstellte Dateien:")
for fmt, path in files.items():
    print(f"  • {fmt.upper()}: {path}")
