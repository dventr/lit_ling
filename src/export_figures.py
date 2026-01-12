#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Springer-konforme Abbildungsexporte
====================================

Exportiert Plotly-Figuren in die von Springer geforderten Formate:
- EPS (Vektorgrafiken) für Strichzeichnungen und einfache Diagramme
- PNG mit 600 dpi für Kombinationsgrafiken (mit vielen Beschriftungen/Farben)
- HTML für interaktive Online-Versionen

Springer-Vorgaben:
- Schriftart: Helvetica oder Arial (8-12 pt in Endgröße = 2-3 mm)
- Breite: 80 mm oder 122 mm (kleinformatig) | 39, 84, 129, 174 mm (großformatig)
- Maximale Höhe: 198 mm (kleinformatig) | 234 mm (großformatig)
- Strichstärke: mindestens 0,1 mm (0,3 pt)
- Auflösung: 1200 dpi (Strich), 300 dpi (Halbton), 600 dpi (Kombination)
"""

import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

# Springer-konforme Größen (in mm)
SPRINGER_WIDTHS = {
    "single": 80,      # Eine Spalte (kleinformatig)
    "double": 122,     # Zwei Spalten / Seitenbreite (kleinformatig)
    "large_s": 84,     # Eine Spalte (großformatig)
    "large_d": 174,    # Zwei Spalten (großformatig)
}

# Konvertierung mm → px (bei 600 dpi für Kombination, 300 dpi für Halbton)
MM_TO_PX_600DPI = 23.622  # 600 dpi / 25.4 mm/inch
MM_TO_PX_300DPI = 11.811  # 300 dpi / 25.4 mm/inch

# Springer-konforme Schriftarten
SPRINGER_FONTS = ["Helvetica", "Arial", "sans-serif"]
SPRINGER_FONT = ", ".join(SPRINGER_FONTS)


def configure_springer_layout(fig, width_mm=122, height_mm=None, font_size=10):
    """
    Konfiguriert Plotly-Figur für Springer-Vorgaben.
    
    Args:
        fig: Plotly Figure object
        width_mm: Breite in mm (80 oder 122 für kleinformatig)
        height_mm: Höhe in mm (max 198 für kleinformatig, None = auto)
        font_size: Schriftgröße in pt (8-12 empfohlen)
    """
    # Konvertierung mm → px (bei 96 dpi Standard-Display)
    width_px = int(width_mm * 3.7795)  # 96 dpi standard
    height_px = int(height_mm * 3.7795) if height_mm else None
    
    fig.update_layout(
        font=dict(
            family=SPRINGER_FONT,
            size=font_size,
            color="black"
        ),
        width=width_px,
        height=height_px,
        plot_bgcolor="white",
        paper_bgcolor="white",
        # Minimale Ränder für optimale Raumnutzung
        margin=dict(l=50, r=30, t=40, b=40)
    )
    
    # Strichstärke mindestens 0,3 pt
    fig.update_xaxes(
        linewidth=0.5,
        gridcolor="lightgray",
        gridwidth=0.3,
        showline=True,
        linecolor="black"
    )
    fig.update_yaxes(
        linewidth=0.5,
        gridcolor="lightgray",
        gridwidth=0.3,
        showline=True,
        linecolor="black"
    )
    
    return fig


def export_figure(fig, basename, output_dir="abbildungen", 
                  figure_number=None, width_mm=122, height_mm=None,
                  save_html=True, save_eps=False, save_pdf=True, save_png=True):
    """
    Exportiert Plotly-Figur in Springer-konforme Formate.
    
    Args:
        fig: Plotly Figure object
        basename: Basis-Dateiname (ohne Nummer und Extension)
        output_dir: Ausgabeverzeichnis
        figure_number: Abbildungsnummer (z.B. 1 → Abb1.eps)
        width_mm: Breite in mm (Standard: 122 = zwei Spalten)
        height_mm: Höhe in mm (None = automatisch, max 198 kleinformatig)
        save_html: HTML-Version speichern (für Online)
        save_eps: EPS-Version speichern (erfordert poppler/pdftops)
        save_pdf: PDF-Version speichern (Vektorgrafik)
        save_png: PNG-Version speichern (600 dpi, für Kombination)
    
    Returns:
        dict: Pfade zu den erstellten Dateien
    """
    # Ausgabeverzeichnis erstellen
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Springer-Layout anwenden
    fig = configure_springer_layout(fig, width_mm, height_mm, font_size=10)
    
    # Dateinamen generieren
    if figure_number is not None:
        prefix = f"Abb{figure_number}"
    else:
        prefix = basename
    
    files = {}
    
    # 1. HTML für Online-Version (interaktiv)
    if save_html:
        html_path = output_path / f"{prefix}.html"
        fig.write_html(str(html_path))
        files['html'] = html_path
        print(f"✅ HTML: {html_path}")
    
    # 2. PDF für Vektorgrafik (Alternative zu EPS, weit verbreitet)
    if save_pdf:
        try:
            pdf_path = output_path / f"{prefix}.pdf"
            # Konvertierung mm → inch für PDF
            width_inch = width_mm / 25.4
            height_inch = height_mm / 25.4 if height_mm else width_inch * 0.75
            
            fig.write_image(
                str(pdf_path),
                format="pdf",
                width=width_inch * 300,  # 300 dpi für PDF
                height=height_inch * 300,
                scale=1
            )
            files['pdf'] = pdf_path
            print(f"✅ PDF: {pdf_path}")
        except Exception as e:
            print(f"⚠️  PDF-Export fehlgeschlagen: {e}")
    
    # 3. EPS für Vektorgrafik (Druck) - optional, erfordert poppler
    if save_eps:
        try:
            eps_path = output_path / f"{prefix}.eps"
            # Konvertierung mm → inch für EPS
            width_inch = width_mm / 25.4
            height_inch = height_mm / 25.4 if height_mm else width_inch * 0.75
            
            fig.write_image(
                str(eps_path),
                format="eps",
                width=width_inch * 300,  # 300 dpi für EPS
                height=height_inch * 300,
                scale=1
            )
            files['eps'] = eps_path
            print(f"✅ EPS: {eps_path}")
        except Exception as e:
            print(f"⚠️  EPS-Export fehlgeschlagen: {e}")
            print("   Installieren Sie poppler: brew install poppler")
    
    # 4. PNG mit 600 dpi (Kombinationsgrafik)
    if save_png:
        try:
            png_path = output_path / f"{prefix}.png"
            # 600 dpi für Kombinationsgrafiken
            width_px = int(width_mm * MM_TO_PX_600DPI)
            height_px = int(height_mm * MM_TO_PX_600DPI) if height_mm else int(width_px * 0.75)
            
            fig.write_image(
                str(png_path),
                format="png",
                width=width_px,
                height=height_px,
                scale=2  # Höhere Qualität
            )
            files['png'] = png_path
            print(f"✅ PNG (600 dpi): {png_path}")
        except Exception as e:
            print(f"⚠️  PNG-Export fehlgeschlagen: {e}")
    
    # Metadaten-Datei erstellen
    meta_path = output_path / f"{prefix}_meta.txt"
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f"Abbildung: {prefix}\n")
        f.write(f"Erstellt mit: Plotly (Python)\n")
        f.write(f"Größe: {width_mm} mm × {height_mm or 'auto'} mm\n")
        f.write(f"Schriftart: {SPRINGER_FONT}\n")
        f.write(f"Schriftgröße: 10 pt\n")
        f.write(f"\nDateien:\n")
        for fmt, path in files.items():
            f.write(f"  - {fmt.upper()}: {path.name}\n")
    
    files['meta'] = meta_path
    print(f"📄 Metadaten: {meta_path}\n")
    
    return files


def copy_and_export_html_figures(input_dir, output_dir="abbildungen", 
                                  pattern="*.html", start_number=1):
    """
    Kopiert HTML-Figuren und erstellt zusätzlich statische Exporte.
    
    HINWEIS: Für vollständigen Export müssen die Visualisierungsskripte
    direkt mit export_figure() aufgerufen werden.
    
    Args:
        input_dir: Verzeichnis mit HTML-Figuren
        output_dir: Ausgabeverzeichnis
        pattern: Dateiname-Pattern (Standard: *.html)
        start_number: Start-Nummer für Abbildungen
    """
    import shutil
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    html_files = sorted(input_path.glob(pattern))
    
    if not html_files:
        print(f"⚠️  Keine Dateien gefunden in {input_dir} mit Pattern {pattern}")
        return
    
    print(f"Kopiere {len(html_files)} HTML-Abbildungen...\n")
    
    for i, html_file in enumerate(html_files, start=start_number):
        # Kopiere HTML
        dest_name = f"Abb{i}.html"
        dest_path = output_path / dest_name
        shutil.copy2(html_file, dest_path)
        print(f"✅ Abb. {i}: {html_file.name} → {dest_name}")
    
    print(f"\n✅ {len(html_files)} HTML-Dateien kopiert nach: {output_dir}")
    print("\n⚠️  WICHTIG:")
    print("   Für EPS/PNG-Export müssen die Visualisierungsskripte")
    print("   mit export_figure() aufgerufen werden.")
    print("   Siehe: update_visualization_scripts.py")


# Beispiel-Verwendung
if __name__ == "__main__":
    print("Springer-konformer Abbildungsexport")
    print("=" * 50)
    print("\nDieses Modul muss aus anderen Skripten importiert werden.")
    print("\nBeispiel:")
    print("  from export_figures import export_figure")
    print("  export_figure(fig, 'kollokation_migration', figure_number=1)")
    print("\nOder HTML-Figuren kopieren:")
    print("  from export_figures import copy_and_export_html_figures")
    print("  copy_and_export_html_figures('viz_out', 'abbildungen')")
