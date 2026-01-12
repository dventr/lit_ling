# Springer-konforme Abbildungen

Dieses Dokument beschreibt, wie Sie Visualisierungen aus diesem Projekt in Springer-konformen Formaten exportieren.

## Springer-Vorgaben

### Formate
- **EPS** (Encapsulated PostScript): Vektorgrafiken für Strichzeichnungen
- **PDF**: Vektorgrafiken (moderne Alternative zu EPS)
- **PNG**: Rastergrafiken für Halbton- und Kombinationsgrafiken (600 dpi)
- **HTML**: Interaktive Online-Version (nicht für Druck)

### Abmessungen (kleinformatig)
- **Eine Spalte**: 80 mm Breite
- **Zwei Spalten**: 122 mm Breite  
- **Maximale Höhe**: 198 mm

### Schriftarten
- **Helvetica** oder **Arial** (8-12 pt)
- Schriftgröße in Endgröße: 2-3 mm

### Auflösung
- **Strichzeichnungen**: 1200 dpi (schwarz-weiß ohne Schattierungen)
- **Halbtonabbildungen**: 300 dpi (Photos, Graustufen)
- **Kombinationsgrafiken**: 600 dpi (Diagramme mit vielen Beschriftungen)

### Linienstärke
- Mindestens 0,1 mm (0,3 pt) in Endgröße

## Verwendung

### 1. Automatischer Export aller Visualisierungen

```bash
cd /Users/dventr/lit_ling
python3 src/convert_all_figures.py
```

Dies erstellt:
- Nummerierte HTML-Dateien (`Abb1.html`, `Abb2.html`, ...)
- Im Verzeichnis `/Users/dventr/lit_ling/abbildungen/`

### 2. Export einzelner Figuren mit allen Formaten

```python
from export_figures import export_figure
import plotly.graph_objects as go

# Ihre Plotly-Figur
fig = go.Figure(...)

# Export in Springer-Formate
export_figure(
    fig,
    basename="kollokation_migration",
    output_dir="abbildungen",
    figure_number=1,        # → Abb1.html, Abb1.pdf, Abb1.png
    width_mm=122,           # Zwei-Spalten-Breite
    height_mm=90,           # Optional, sonst automatisch
    save_html=True,
    save_pdf=True,          # Vektorgrafik (empfohlen)
    save_eps=False,         # Nur mit poppler (brew install poppler)
    save_png=True           # 600 dpi Rastergrafik
)
```

### 3. Integration in Visualisierungsskripte

Fügen Sie am Ende Ihrer Visualisierungsskripte hinzu:

```python
# Am Anfang der Datei
import sys
sys.path.insert(0, '/Users/dventr/lit_ling/src')
from export_figures import export_figure

# Nach dem Erstellen Ihrer Figur
fig.write_html("viz_out/meine_visualisierung.html")  # Bisherig

# NEU: Zusätzlich Springer-Export
export_figure(
    fig,
    basename="meine_visualisierung",
    output_dir="abbildungen",
    figure_number=5,
    width_mm=122,
    save_html=True,
    save_pdf=True,
    save_png=True
)
```

## Generierte Dateien

Nach dem Export finden Sie im Verzeichnis `abbildungen/`:

```
Abb1.html          # Interaktive HTML-Version
Abb1.pdf           # Vektorgrafik (für Druck)
Abb1.png           # 600 dpi Rastergrafik
Abb1_meta.txt      # Metadaten (Tool, Größe, Schriftart)
```

## Abbildungslegenden

**Wichtig**: Legenden gehören NICHT in die Bilddateien!

Erstellen Sie Legenden am Ende Ihrer Textdatei:

```
**Abb. 1** Kollokationen des Wortes "Migration" in center-rechter Discourse. 
Größere Punkte zeigen häufigere Assoziationen. Farben repräsentieren 
semantische Cluster. Die Achsen zeigen die t-SNE-Projektion der 
Worteinbettungen (Quelle: Eigene Analyse basierend auf Korpus X)

**Abb. 2** Verteilung der Identitätsbegriffe nach politischer Partei. 
Balken zeigen normalisierte Frequenzen pro 1000 Wörter. Fehlerbalken 
indizieren 95% Konfidenzintervalle
```

### Legenden-Format

- Beginnen mit: **Abb.** + Nummer (fett)
- Kein Punkt nach der Nummer
- Kein Punkt am Ende der Legende
- Alle Abkürzungen und Symbole erklären
- Quellen in Klammern angeben

## Nummerierung

- Fortlaufend: Abb. 1, Abb. 2, Abb. 3, ...
- Teilabbildungen: a, b, c (z.B. Abb. 1a, Abb. 1b)
- Anhang-Abbildungen: A1, A2, ... (optional)

## Checkliste für Springer-Einreichung

- [ ] Alle Abbildungen in korrekter Größe (80 mm oder 122 mm breit)
- [ ] Schriftart: Helvetica/Arial, 8-12 pt
- [ ] Auflösung: 600 dpi für Kombinationsgrafiken
- [ ] Dateinamen: Abb1.pdf, Abb2.pdf, Abb3.pdf, ...
- [ ] Legenden in separater Textdatei (nicht in Bildern)
- [ ] Alle Abbildungen im Text erwähnt
- [ ] Farben auch in Schwarzweiß erkennbar
- [ ] Bei übernommenen Abbildungen: Genehmigungen eingeholt
- [ ] Metadaten-Dateien (*_meta.txt) für Verlag erstellt

## Farbgebung

Die Visualisierungen verwenden ein **barrierefreies, farbenblind-freundliches Farbschema**:

- Hoher Kontrast (WCAG AAA-konform)
- Unterscheidbar bei allen Formen von Farbfehlsichtigkeit
- Zusätzliche Unterscheidung durch Symbole (Kreise, Quadrate, Dreiecke)
- Auch in Schwarzweiß gut erkennbar

## Technische Details

### Verwendete Tools
- **Plotly** (Python): Interaktive Visualisierungen
- **Kaleido**: Statischer Bildexport
- **Export-Modul**: `src/export_figures.py`

### Konvertierung
- Plotly-Figuren werden direkt in PDF/PNG exportiert
- Schriftarten werden eingebettet
- RGB-Farbmodus für Farbabbildungen

### Bei Problemen

**EPS-Export funktioniert nicht:**
```bash
brew install poppler  # macOS
# oder
apt-get install poppler-utils  # Linux
```

**Schriftarten fehlen:**
- Helvetica/Arial sind Systemschriften
- Falls nicht vorhanden: Liberation Sans ist eine freie Alternative

**Dateien zu groß:**
- PDF komprimieren: `gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dNOPAUSE -dQUIET -dBATCH -sOutputFile=output.pdf input.pdf`
- PNG optimieren: `optipng -o7 Abb1.png`

## Kontakt

Bei Fragen zur Abbildungs-Erstellung oder Springer-konformen Formatierung wenden Sie sich an [Ihre Kontaktdaten].
