#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kopiert alle Visualisierungen und nummeriert sie nach Springer-Vorgaben
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from export_figures import copy_and_export_html_figures

# Get current working directory
WORKSPACE = Path.cwd()

# Verzeichnisse mit Visualisierungen (relativ zum workspace)
VIZ_DIRS = [
    WORKSPACE / "viz_out",
    WORKSPACE / "outfiles" / "plots",
]

OUTPUT_DIR = WORKSPACE / "abbildungen"

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
    print(f"📂 Workspace: {WORKSPACE}")
    print(f"📊 {figure_number - 1} Abbildungen nummeriert")
    print("\n📝 Nächste Schritte:")
    print("   1. Für EPS/PNG-Export: Visualisierungsskripte mit")
    print("      export_figure() aktualisieren")
    print("   2. Abbildungslegenden erstellen (siehe Springer-Vorgaben)")
    print("   3. Bei Bedarf Breite anpassen (80mm oder 122mm)")

if __name__ == "__main__":
    main()
