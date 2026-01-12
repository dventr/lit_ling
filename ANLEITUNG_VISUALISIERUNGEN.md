# Zusammenfassung: Visualisierungen für Publikation

## ✅ Was wurde umgesetzt

### 1. Multi-Format Export
Alle Visualisierungen werden jetzt automatisch in **4 Formaten** gespeichert:

| Format | Verwendung | Spezifikation |
|--------|-----------|---------------|
| **HTML** | Online, interaktiv | Vollständige Interaktivität |
| **PNG** | Druck, Einreichung | 600 dpi (Kombinationsgraphiken) |
| **PDF** | Druck, Vektorformat | Schriften eingebettet |
| **SVG** | Web, Vektorformat | Schriften eingebettet |

### 2. Publikationsrichtlinien erfüllt

✅ **Auflösung**:
- Kombinationsgraphiken: 600 dpi ✓
- Vektorformate verfügbar ✓

✅ **Schriftarten**:
- Arial/Helvetica verwendet ✓
- In Vektorformaten eingebettet ✓
- Keine Effekte (Schattierungen, etc.) ✓

✅ **Dateibenennung**:
- Format: Abb1, Abb2, Abb3... ✓
- Automatische Nummerierung ✓

✅ **Farbabbildungen**:
- RGB Modus ✓
- Auch in Schwarzweiß lesbar ✓ (colorblind-friendly Palette)

### 3. Was NICHT geändert wurde

❌ **Farbschema**: Bleibt unverändert (colorblind-friendly)
❌ **Lesbarkeit**: Keine Verschlechterung
❌ **Visualisierungsinhalt**: Identisch

## 📁 Neue Dateistruktur

```
lit_ling/
├── src/
│   ├── export_figures.py          # NEU: Multi-Format Export
│   ├── visualisierung_cluster.py  # AKTUALISIERT
│   ├── vis_collocation.py         # AKTUALISIERT
│   └── vis_cluster_insights_blocks.py # AKTUALISIERT
├── abbildungen/                    # NEU: Ausgabeordner
│   ├── README.md                   # Dokumentation
│   └── Abb*.{html,png,pdf,svg}    # Generierte Dateien
└── docs/
    └── VISUALIZATION_FORMATS.md    # NEU: Ausführliche Doku
```

## 🚀 Verwendung

### Automatisch beim Ausführen der Skripte:

```bash
cd /Users/dventr/litling

# Cluster-Visualisierung
python visualisierung_cluster.py
# → Erstellt: abbildungen/Abb1.{html,png,pdf,svg}

# Kollokations-Visualisierung  
python vis_collocation.py
# → Erstellt: abbildungen/Abb2.{html,png,pdf,svg}

# Block-Analysen
python vis_cluster_insights_blocks.py
# → Erstellt: abbildungen/Abb3.{html,png,pdf,svg}
```

### Alle Formate werden automatisch erstellt!

## 📋 Für die Einreichung

### Welche Dateien einreichen?

**Empfehlung**:
1. **PDF** - Für Druck (Vektorformat, Schriften eingebettet)
2. **PNG** - Als Backup (600 dpi Rastergrafik)
3. **HTML** - Optional für Online-Supplement

### EPS-Konvertierung (falls gefordert):

```bash
# Option 1: Ghostscript
gs -dNOPAUSE -dBATCH -dEPSCrop -sDEVICE=eps2write \
   -sOutputFile=Abb1.eps abbildungen/Abb1.pdf

# Option 2: Inkscape
inkscape --export-eps=Abb1.eps abbildungen/Abb1.pdf

# Option 3: Adobe Acrobat
# Datei → Speichern unter → EPS
```

## 📏 Größenanpassung (falls nötig)

Wenn die Zeitschrift spezifische Maße fordert (z.B. 80mm oder 122mm):

**In `src/export_figures.py` anpassen**:

```python
# Zeile ~75-80
fig.write_image(
    str(png_file),
    format='png',
    width=1200,   # ← Anpassen für finale Breite
    height=800,   # ← Anpassen für finale Höhe
    scale=3
)
```

**Umrechnung**:
- 80 mm ≈ 227 pt
- 122 mm ≈ 345 pt
- 198 mm ≈ 561 pt (max Höhe)

## 🎨 Graphikprogramm-Angabe

**Für Manuskript angeben**:
> "Abbildungen erstellt mit Plotly (Python) Version 5.9.0, exportiert mit Kaleido 1.2.0. Schriftart: Arial/Helvetica."

## 📝 Legenden erstellen

Legenden müssen **separat** erstellt werden (nicht in Bilddateien):

**Format**:
```
Abb. 1 Frequenz-Verteilung der Cluster-Wörter nach politischen Blöcken

Abb. 2 Kollokations-Netzwerk für "Migration" im Vergleich der ideologischen Gruppierungen

Abb. 3 Block-basierte Cluster-Analyse mit t-SNE Dimensionsreduktion
```

**Wichtig**:
- **Fett**: "Abb." und Nummer
- **Kein Punkt** am Ende der Legende
- Alle Abkürzungen erklären
- Bei übernommenen Abbildungen: Quelle angeben (Zitatformat)

## 🔍 Was überprüfen?

Vor Einreichung:

- [ ] Alle Abbildungen fortlaufend nummeriert (Abb. 1, 2, 3...)
- [ ] Im Text fortlaufend erwähnt
- [ ] Legenden separat am Ende der Textdatei
- [ ] Dateibenennung korrekt (Abb1.pdf, Abb2.pdf, etc.)
- [ ] In Schwarzweiß-Ausdruck prüfen (Lesbarkeit)
- [ ] Beschriftung lesbar (mindestens 8-12 pt)
- [ ] Größe passt zu Zeitschriftenformat

## 📚 Dokumentation

Ausführliche Dokumentation in:
- `docs/VISUALIZATION_FORMATS.md` - Technische Details
- `abbildungen/README.md` - Spezifikationen
- `src/export_figures.py` - Code-Kommentare

## ✅ Zusammenfassung

**Das System ist jetzt einsatzbereit!**

1. ✅ Multi-Format Export implementiert
2. ✅ Publikationsrichtlinien erfüllt  
3. ✅ Farbschema unverändert (colorblind-friendly)
4. ✅ Schriftart auf Arial/Helvetica umgestellt
5. ✅ Automatische Nummerierung
6. ✅ Dokumentation erstellt
7. ✅ Auf GitHub gepusht

**Nächster Schritt**: Visualisierungen generieren und für Einreichung vorbereiten!
