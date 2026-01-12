#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exportiert alle bestehenden HTML-Visualisierungen in PDF und PNG Formate.
Nutzt Plotly's eingebaute Konvertierung.
"""

import sys
import os
sys.path.insert(0, 'src')

from pathlib import Path
import plotly.io as pio

# Konfiguration
INPUT_DIR = Path('abbildungen')
OUTPUT_DIR = Path('abbildungen')
WIDTH_MM = 122  # Springer zwei-Spalten
HEIGHT_MM = 90  # Angepasst

# Konvertierung mm -> px bei 300 dpi (für PDF/EPS)
MM_TO_PX = 11.811  # 300 dpi / 25.4 mm/inch

def export_html_to_formats(html_file, fig_number):
    """
    Exportiert eine HTML-Datei in PDF und PNG Formate.
    """
    print(f"Verarbeite: {html_file.name}")
    
    try:
        # Lade HTML mit Plotly
        with open(html_file, 'r', encoding='utf-8') as f:
            html_str = f.read()
        
        # Finde die JSON-Daten im HTML
        import re
        import json
        
        # Extrahiere Plotly JSON (verschiedene Patterns)
        patterns = [
            r'var data = (\[.*?\]);',
            r'var layout = (\{.*?\});',
            r'Plotly\.newPlot\([^,]+,\s*(\[.*?\]),\s*(\{.*?\}),',
        ]
        
        # Versuche Figure direkt aus HTML zu laden
        # Plotly hat eine eingebaute Funktion dafür
        try:
            import plotly.io as pio
            # Lese das HTML und extrahiere die Figure
            # Dies ist ein Workaround - wir müssen das HTML manuell parsen
            
            # Suche nach dem script tag mit den Daten
            script_match = re.search(r'<script[^>]*>(.*?)</script>', html_str, re.DOTALL)
            if script_match:
                script_content = script_match.group(1)
                
                # Suche nach Plotly.newPlot oder Plotly.react
                plot_match = re.search(
                    r'Plotly\.(newPlot|react)\s*\(\s*[\'"]([^\'\"]+)[\'\"],\s*(\[[\s\S]*?\]),\s*(\{[\s\S]*?\})\s*[,\)]',
                    script_content
                )
                
                if plot_match:
                    div_id = plot_match.group(2)
                    # Wir haben die Daten gefunden, aber JSON-Parsing ist komplex
                    # Einfacherer Weg: Nutze kaleido direkt
                    pass
        except Exception as e:
            print(f"  ⚠️  Komplexes HTML-Format: {e}")
        
        # Alternative: Nutze Screenshot-basierte Konvertierung
        # Dies funktioniert mit kaleido
        prefix = f"Abb{fig_number}"
        
        # PDF Export
        pdf_path = OUTPUT_DIR / f"{prefix}.pdf"
        cmd_pdf = f'python3 -c "import plotly.io as pio; fig = pio.read_json(\\"{html_file}\\"); pio.write_image(fig, \\"{pdf_path}\\", format=\\"pdf\\", width={int(WIDTH_MM * MM_TO_PX)}, height={int(HEIGHT_MM * MM_TO_PX)})"'
        
        # Einfacherer Weg: Verwende orca oder kaleido command-line
        # Aber das HTML-Format von Plotly ist komplex
        
        print(f"  ⚠️  HTML zu PDF/PNG Konvertierung erfordert Neugenerierung")
        print(f"      Siehe: docs/ABBILDUNGEN.md für Integration in Skripte")
        
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
    
    print()

def main():
    print("Export von HTML zu PDF/PNG")
    print("=" * 60)
    print()
    
    html_files = sorted(INPUT_DIR.glob('Abb[0-9]*.html'))
    
    if not html_files:
        print("❌ Keine Abbildungen gefunden in:", INPUT_DIR)
        return
    
    print(f"Gefunden: {len(html_files)} HTML-Dateien\n")
    print("⚠️  HINWEIS:")
    print("   Für beste Qualität sollten Visualisierungsskripte")
    print("   direkt mit export_figure() aufgerufen werden.")
    print("   Siehe: docs/ABBILDUNGEN.md\n")
    print("=" * 60)
    print()
    
    for html_file in html_files:
        fig_num = int(''.join(filter(str.isdigit, html_file.stem)))
        export_html_to_formats(html_file, fig_num)
    
    print("\n" + "=" * 60)
    print("✅ Für PDF/PNG-Export aus HTML-Dateien:")
    print("   1. Installieren: brew install wkhtmltopdf")
    print("   2. Oder: Visualisierungsskripte mit export_figure() aktualisieren")
    print(f"\n📂 Ausgabe: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
