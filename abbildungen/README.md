# Springer-konforme Abbildungen - Kurzanleitung

## ✅ Was wurde umgesetzt

Alle Visualisierungen können jetzt in Springer-konformen Formaten exportiert werden:

### Formate
- ✅ **HTML** - Interaktive Online-Version
- ✅ **PDF** - Vektorgrafik für Druck
- ✅ **PNG** - 600 dpi Rastergrafik für Kombinationsgrafiken
- ⚠️ **EPS** - Optional (erfordert poppler: `brew install poppler`)

### Springer-Vorgaben erfüllt
- ✅ Schriftart: Helvetica/Arial (8-12 pt)
- ✅ Breite: 122 mm (zwei Spalten) oder 80 mm (eine Spalte)
- ✅ Auflösung: 600 dpi (Kombinationsgrafiken)
- ✅ Linienstärke: mindestens 0,3 pt
- ✅ Nummerierung: Abb1, Abb2, Abb3, ...
- ✅ Metadaten-Dateien mit technischen Details

## 📁 Verzeichnisstruktur

```
lit_ling/
├── abbildungen/           # NEUE! Springer-konforme Abbildungen
│   ├── Abb1.html          # Nummerierte Visualisierungen
│   ├── Abb1.pdf
│   ├── Abb1.png
│   ├── Abb1_meta.txt      # Metadaten (Tool, Größe, Schriftart)
│   └── ...
├── src/
│   ├── export_figures.py  # NEUE! Export-Modul
│   ├── convert_all_figures.py  # NEUE! Batch-Konvertierung
│   └── example_export.py  # NEUE! Beispiel-Verwendung
└── docs/
    └── ABBILDUNGEN.md     # NEUE! Vollständige Dokumentation
```

## 🚀 Schnellstart

### Alle vorhandenen Visualisierungen konvertieren:

```bash
cd /Users/dventr/lit_ling
python3 src/convert_all_figures.py
```

### Einzelne Figur exportieren:

```python
from export_figures import export_figure

export_figure(
    fig,                    # Ihre Plotly-Figur
    basename="migration",
    figure_number=1,
    width_mm=122,           # Zwei-Spalten-Breite
    save_html=True,
    save_pdf=True,
    save_png=True
)
```

## 📊 Aktueller Stand

- **7 Abbildungen** nummeriert und exportiert (Abb1-Abb7)
- **HTML-Dateien** für alle Visualisierungen vorhanden
- **PDF/PNG-Export** funktioniert (Beispiel: Abb99)
- **Dokumentation** vollständig in `docs/ABBILDUNGEN.md`

## 📝 Nächste Schritte für Publikation

1. **Abbildungslegenden erstellen**
   - Siehe Template in `docs/ABBILDUNGEN.md`
   - Legenden gehören ans Ende der Textdatei, NICHT in Bilder

2. **Größen anpassen** (falls nötig)
   ```python
   width_mm=80   # Eine Spalte
   width_mm=122  # Zwei Spalten (Standard)
   ```

3. **Farben prüfen**
   - Schwarzweiß-Druck testen
   - Bereits barrierefreie, farbenblind-freundliche Palette

4. **Alle Figuren neu generieren** mit Export-Funktion
   - Integration in bestehende Visualisierungsskripte
   - Siehe Beispiel in `src/example_export.py`

## 🔧 EPS-Export aktivieren (optional)

Falls EPS-Format benötigt:

```bash
brew install poppler  # macOS
```

Dann in Skripten `save_eps=True` setzen.

## 📖 Vollständige Dokumentation

Siehe: `docs/ABBILDUNGEN.md`

- Detaillierte Springer-Vorgaben
- Verwendungsbeispiele
- Checkliste für Einreichung
- Problemlösungen
