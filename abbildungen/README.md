# Abbildungen (Figures)

Publikationsfertige Abbildungen für die Einreichung.

## Format-Spezifikationen

Alle Abbildungen werden automatisch in mehreren Formaten exportiert:

### Verfügbare Formate:
- **HTML**: Interaktive Version für Online-Publikation
- **PNG**: Hochauflösende Rastergrafik (600 dpi für Kombinationsgraphiken)
- **PDF**: Vektorformat für Druck (Schriften eingebettet)
- **SVG**: Alternatives Vektorformat (Schriften eingebettet)

### Publikationsrichtlinien-Konformität:

**Kombinationsgraphiken** (Farbdiagramme mit Beschriftungen):
- ✅ Mindestauflösung: 600 dpi (PNG)
- ✅ Vektorformat: PDF, SVG verfügbar
- ✅ Schriftart: Arial/Helvetica (eingebettet)
- ⚠️  Größe: Manuell anpassen auf 80mm oder 122mm Breite

**Farbabbildungen**:
- ✅ RGB Modus (8 bits per channel)
- ✅ Auch in Schwarzweiß lesbar (colorblind-friendly Palette)
- ✅ Farbverweise in Legende vermieden

**Beschriftung**:
- ✅ Arial/Helvetica verwendet
- ✅ Keine Effekte (Schattierungen, Umrisse)
- ✅ Konsistente Größe: ~8-12 pt

**Nummerierung**:
- ✅ Fortlaufend: Abb1, Abb2, Abb3...
- ✅ Teilabbildungen: Abb1a, Abb1b, etc.

## Generierung

Abbildungen werden durch folgende Skripte erstellt:

- `visualisierung_cluster.py` - Cluster-Visualisierungen
- `vis_collocation.py` - Kollokations-Netzwerke
- `vis_cluster_insights_blocks.py` - Block-Analysen

## Graphikprogramm

**Software**: Plotly (Python) mit kaleido für statische Exports
**Version**: Siehe requirements.txt
**Schriften**: Arial/Helvetica (systembasiert)

## EPS-Konvertierung (falls benötigt)

PDF → EPS Konvertierung:

```bash
# Option 1: Ghostscript
gs -dNOPAUSE -dBATCH -dEPSCrop -sDEVICE=eps2write \
   -sOutputFile=Abb1.eps Abb1.pdf

# Option 2: Inkscape
inkscape --export-eps=Abb1.eps Abb1.pdf

# Option 3: Adobe Acrobat
# Öffnen → Speichern unter → EPS
```

## Dateibenennung

Folgt dem Schema: `Abb{Nummer}[_Beschreibung].{format}`

Beispiele:
- `Abb1.html`, `Abb1.png`, `Abb1.pdf`, `Abb1.svg`
- `Abb2_cluster_analysis.html`, etc.

## Verwendung in Manuskript

1. Abbildungen fortlaufend im Text erwähnen
2. Legenden separat am Ende der Textdatei anfügen
3. Legenden-Format: **Abb. X** Beschreibungstext (kein Punkt am Ende)
4. Alle Abkürzungen und Symbole in Legende erklären
5. Bei übernommenen Abbildungen: Quelle angeben

## Größenanpassung

Standard-Breiten für Zeitschriften:
- Kleinformat: 80 mm oder 122 mm
- Großformat: 39 mm, 84 mm, 129 mm, oder 174 mm

Maximale Höhe:
- Kleinformat: 198 mm
- Großformat: 234 mm

**Anpassung in Code**: Editiere `width` und `height` Parameter in `save_figure_multi_format()`
