# Visualisierungen - Publikationsformate

## Übersicht der Änderungen

Die Visualisierungsskripte wurden angepasst, um publikationsfertige Abbildungen gemäß den deutschen akademischen Richtlinien zu erstellen.

## Implementierte Features

### 1. Multi-Format Export
Alle Visualisierungen werden automatisch in folgenden Formaten gespeichert:

- **HTML**: Interaktive Online-Version
- **PNG**: Hochauflösend (600 dpi für Kombinationsgraphiken)  
- **PDF**: Vektorformat mit eingebetteten Schriften
- **SVG**: Alternatives Vektorformat

### 2. Schriftarten
- **Geändert von**: Lato → **Arial/Helvetica**
- **Grund**: Gemäß Publikationsrichtlinien (Arial/Helvetica empfohlen)
- **Einbettung**: Schriften werden in PDF/SVG automatisch eingebettet

### 3. Farbschema
- **BEIBEHALTEN**: Das bestehende colorblind-friendly Farbschema bleibt unverändert
- **Kontrast**: WCAG AA-konform (bis auf gelb-grün, das durch Formen unterscheidbar ist)
- **Schwarzweiß**: Funktioniert auch in Graustufen

### 4. Auflösung
- **Kombinationsgraphiken**: 600 dpi (Farbdiagramme mit Text)
- **Vektorformate**: Skalierbar ohne Qualitätsverlust
- **Erfüllt**: Mindestanforderungen für akademische Publikationen

## Verwendete Skripte

### Hauptskripte mit Multi-Format Export:

1. **visualisierung_cluster.py**
   - Cluster-Frequenz-Visualisierungen
   - Export als Abb1, Abb2, etc.

2. **vis_collocation.py**
   - Kollokations-Netzwerke
   - Shared/unique word visualizations

3. **vis_cluster_insights_blocks.py**  
   - Block-basierte Cluster-Analysen

### Hilfsskript:

**export_figures.py**
- Zentrale Export-Funktion: `save_figure_multi_format()`
- Automatische Nummerierung
- EPS-Konvertierungsanleitung

## Ausgabeordner

```
abbildungen/
├── Abb1.html
├── Abb1.png
├── Abb1.pdf
├── Abb1.svg
├── Abb2.html
├── Abb2.png
├── Abb2.pdf
├── Abb2.svg
└── README.md
```

## Verwendung

### Einfache Verwendung:
```python
from export_figures import save_figure_multi_format

# Plotly Figure erstellen
fig = go.Figure(...)

# In allen Formaten speichern
save_figure_multi_format(
    fig, 
    "Abb1",  # Basisname
    output_dir="abbildungen",
    dpi=600
)
```

### Die Skripte nutzen dies automatisch:
```bash
# Führe Visualisierung aus
python visualisierung_cluster.py

# Erstellt automatisch:
# - abbildungen/Abb{N}.html
# - abbildungen/Abb{N}.png
# - abbildungen/Abb{N}.pdf  
# - abbildungen/Abb{N}.svg
```

## Publikationsrichtlinien-Checkliste

✅ **Formate**:
- EPS/PDF für Vektorgraphiken (PDF erstellt, EPS-Konvertierung möglich)
- TIFF/PNG für Halbtonabbildungen (PNG mit 600 dpi)
- MS Office kompatibel (HTML kann in Word eingebettet werden)

✅ **Auflösung**:
- Strichzeichnungen: Vektorformat (skalierbar)
- Halbtonabbildungen: 300+ dpi (wir nutzen 600 dpi)
- Kombinationsgraphiken: 600 dpi minimum

✅ **Beschriftung**:
- Arial/Helvetica verwendet
- Keine Effekte (Schattierungen, Umrisse)
- Konsistente Größe (anpassbar via Config)

✅ **Benennung**:
- Format: Abb1, Abb2, Abb3...
- Erweiterungen: .html, .png, .pdf, .svg

✅ **Farbabbildungen**:
- RGB Modus (8 bits/channel)
- Auch in Schwarzweiß erkennbar
- Keine Farbverweise in Legende nötig (durch Symbole unterscheidbar)

## EPS-Konvertierung (falls benötigt)

PDF kann einfach zu EPS konvertiert werden:

### Option 1: Ghostscript (Command Line)
```bash
gs -dNOPAUSE -dBATCH -dEPSCrop -sDEVICE=eps2write \
   -sOutputFile=Abb1.eps Abb1.pdf
```

### Option 2: Inkscape
```bash
inkscape --export-eps=Abb1.eps Abb1.pdf
```

### Option 3: Adobe Acrobat
Datei öffnen → Speichern unter → EPS auswählen

## Größenanpassung

**Standard-Zeitschriftenformate**:
- Kleinformat: 80 mm oder 122 mm Breite, max. 198 mm Höhe
- Großformat: 39/84/129/174 mm Breite, max. 234 mm Höhe

**Anpassung in Code**:
```python
save_figure_multi_format(
    fig,
    "Abb1",
    output_dir="abbildungen",
    dpi=600
    # In export_figures.py width/height anpassen:
    # width=345 für 122mm, width=227 für 80mm
)
```

## Graphikprogramm-Angabe für Publikation

**Software**: Plotly (Python-Bibliothek)
**Version**: 5.9.0 (siehe requirements.txt)
**Export-Engine**: Kaleido 1.2.0
**Schriften**: Arial/Helvetica (system fonts, embedded in vector formats)

## Was NICHT geändert wurde

- ❌ Farbschema (bleibt colorblind-friendly)
- ❌ Lesbarkeit der Visualisierungen
- ❌ Dateninhalt oder -darstellung
- ❌ Interaktive Features in HTML

## Nächste Schritte

1. ✅ Visualisierungen erstellen (Skripte ausführen)
2. ✅ Dateien werden automatisch in `abbildungen/` gespeichert
3. ⚠️  Größe ggf. anpassen (width/height in export_figures.py)
4. ⚠️  Falls EPS benötigt: PDF → EPS konvertieren
5. ✅ Legenden separat in Textdatei erstellen
6. ✅ Im Manuskript fortlaufend referenzieren

## Kontakt & Support

Bei Fragen zur Verwendung oder Anpassung der Export-Funktion:
- Siehe Kommentare in `export_figures.py`
- Beispielcode in den Visualisierungsskripten
- README in `abbildungen/`
