# Abbildungen für Publikation erstellen

Diese Anleitung erklärt, wie Sie publikationsfertige Abbildungen gemäß Verlagsvorgaben erstellen.

## Voraussetzungen

Installieren Sie zunächst kaleido für den Export statischer Formate:

```bash
pip install kaleido
```

## Visualisierungen erstellen

Das Skript `vis_collocation.py` erstellt automatisch alle Abbildungen in mehreren Formaten:

```bash
python src/vis_collocation.py
```

## Generierte Formate

Für jede Abbildung werden automatisch erstellt:

### HTML (Interaktiv)
- Für Präsentationen und Online-Publikationen
- Mit Zoom- und Hover-Funktionen
- Dateiname: `Abb1.html`, `Abb2.html`, etc.

### PNG (600 dpi)
- Hochauflösend für Kombinationsgraphiken
- Entspricht Verlagsvorgabe: 600 dpi für Farbdiagramme
- Auflösung: 4110 × 3000 Pixel (174mm × ca. 180mm bei 600 dpi)
- Dateiname: `Abb1.png`, `Abb2.png`, etc.

### PDF (Vektor)
- Skalierbar ohne Qualitätsverlust
- Bevorzugtes Format für viele Verlage
- Abmessungen: 174mm × 234mm (großformatige Zeitschriften)
- Dateiname: `Abb1.pdf`, `Abb2.pdf`, etc.

## Verlagsvorgaben

Die Abbildungen erfüllen folgende Spezifikationen:

### Schriftarten
- **Verwendet**: Helvetica/Arial (wie empfohlen)
- **Größe**: 8-12 pt in Endgröße
- **Keine Effekte**: Keine Schattierungen oder Umrisse

### Auflösungen
- **Kombinationsgraphiken**: 600 dpi (PNG)
- **Vektorgraphiken**: Skalierbar (PDF)
- **Linienstärke**: Mindestens 0,3 pt

### Farbmodus
- **RGB**: 8 bits per channel
- **Schwarzweiß-kompatibel**: Zusätzliche Symbole für Unterscheidbarkeit

### Abmessungen
- **Breite**: 174 mm (großformatiges Layout)
- **Maximale Höhe**: 234 mm
- Bilder werden automatisch in der richtigen Größe erstellt

## Farbschema

Das verwendete Farbschema ist:
- ✅ Farbenblindfreundlich
- ✅ Hoher Kontrast (auch in S/W erkennbar)
- ✅ Konsistent über alle Abbildungen

Farben nach ideologischer Zuordnung:
- **Sozial**: #AF5F0B (Orange-Braun) - Kreis ●
- **Grün**: #E3DA48 (Gelb-Grün) - Quadrat ■
- **Mittelinks**: #830861 (Violett) - Diamant ◆
- **Mitte**: #8B0000 (Dunkelrot) - Dreieck ▲
- **Mitterechts**: #441076 (Dunkelviolett) - Stern ★
- **Rechts**: #013343 (Dunkelblau) - X ✕

## Dateinamen

Gemäß Verlagsvorgaben werden die Dateien benannt als:
- `Abb1.html`, `Abb1.png`, `Abb1.pdf`
- `Abb2.html`, `Abb2.png`, `Abb2.pdf`
- `Abb3.html`, `Abb3.png`, `Abb3.pdf`
- `Abb4.html`, `Abb4.png`, `Abb4.pdf`

## EPS-Format (optional)

Falls der Verlag explizit EPS-Format verlangt, können Sie PDF-Dateien konvertieren:

### Mit Adobe Acrobat
1. Öffnen Sie die PDF-Datei
2. Speichern unter → Format: EPS

### Mit Inkscape (Open Source)
```bash
inkscape Abb1.pdf --export-filename=Abb1.eps
```

### Mit pdf2ps (Command Line)
```bash
pdf2ps Abb1.pdf Abb1.eps
```

## Abbildungslegenden

**Wichtig**: Legenden werden NICHT in die Bilddateien eingefügt, sondern separat im Manuskript geführt.

Format der Legenden:
- Beginnen mit "**Abb. X**" (fett)
- Keine Punkte nach Nummer und am Ende
- Alle Abkürzungen und Symbole erklären

Beispiel:
```
Abb. 1 Verteilung von einzigartigen und geteilten Wörtern über ideologische 
Listen. Die Farben repräsentieren verschiedene ideologische Positionen: 
Sozial (orange-braun), Grün (gelb-grün), Mittelinks (violett), Mitte (dunkelrot), 
Mitterechts (dunkelviolett), Rechts (dunkelblau)
```

## Qualitätskontrolle

Vor der Einreichung prüfen Sie:

- [ ] Alle Beschriftungen sind lesbar (mindestens 8 pt)
- [ ] Linien sind mindestens 0,3 pt dick
- [ ] Farben sind auch in Schwarzweiß unterscheidbar
- [ ] Keine Legenden in den Bilddateien
- [ ] Abmessungen entsprechen Vorgaben (174mm × max. 234mm)
- [ ] PDF-Dateien sind Vektorgraphiken (skalierbar)
- [ ] PNG-Dateien haben 600 dpi Auflösung

## Software-Information

**Erstellt mit**: Plotly (Python library, Version 5.9.0) + kaleido export engine

Diese Information sollte in Ihrem Manuskript unter "Methoden" oder in einer Fußnote angegeben werden.

## Weitere Visualisierungen

Falls Sie weitere Visualisierungen aus anderen Skripten erstellen möchten, können Sie die `save_html_with_font()` Funktion aus `vis_collocation.py` als Vorlage verwenden. Sie exportiert automatisch alle erforderlichen Formate.
