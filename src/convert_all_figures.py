#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kopiert alle Visualisierungen und nummeriert sie nach Springer-Vorgaben
"""

import sys
sys.path.insert(0, '/Users/dventr/lit_ling/src')

from export_figures import copy_and_export_html_figures
from pathlib import Path

# Verzeichnisse mit Visualisierungen
VIZ_DIRS = [
    "/Users/dventr/litling/viz_out",
    "/Users/dventr/litling/outfiles/plots",
]

OUTPUT_DIR = "/Users/dventr/lit_ling/abbildungen"

def main():
    print("Springer-konforme Nummerierung aller Visualisierungen")
    print("=" * 60)
    print()
    
    figure_number = 1
    
    for viz_dir in VIZ_DIRS:
        viz_path = Path(viz_dir)
        if not viz_path.exists():
            print(f"⚠️  Verzeichnis nicht gefunden: {viz_dir}")
            continue
        
        print(f"\n📁 Verarbeite: {viz_dir}")
        print("-" * 60)
        
        copy_and_export_html_figures(
            viz_dir,
            OUTPUT_DIR,
            pattern="*.html",
            start_number=figure_number
        )
        
        # Update figure counter
        html_files = list(viz_path.glob("*.html"))
        figure_number += len(html_files)
    
    print("\n" + "=" * 60)
    print(f"✅ Kopieren abgeschlossen!")
    print(f"📂 Ausgabe: {OUTPUT_DIR}")
    print(f"📊 {figure_number - 1} Abbildungen nummeriert")
    print("\n📝 Nächste Schritte:")
    print("   1. Für EPS/PNG-Export: Visualisierungsskripte mit")
    print("      export_figure() aktualisieren")
    print("   2. Abbildungslegenden erstellen (siehe Springer-Vorgaben)")
    print("   3. Bei Bedarf Breite anpassen (80mm oder 122mm)")

if __name__ == "__main__":
    main()
