# Example Data Format

This directory should contain your corpus data files.

## Required Format

Your main corpus file should be a tab-separated values (TSV) file named `cleaned_file.tsv` with the following structure:

### Minimum Required Columns:
- **text**: The text content to analyze
- **partei**: Party affiliation (optional, for party-based analysis)
- **land**: Country code (CH, DE, OE) (optional, for country-based analysis)

### Example:

```tsv
text	partei	land
"Dies ist ein Beispieltext für die Analyse."	spd	DE
"Ein weiterer Text aus der Schweiz."	svp	CH
"Österreichischer Text für die Korpusanalyse."	oevp	OE
```

## Adding Your Data

1. Place your `cleaned_file.tsv` file in the project root directory (not in this folder)
2. Ensure proper encoding (UTF-8 recommended)
3. Use tabs as separators (not spaces or commas)

## Note

Corpus data files are excluded from git by default (see `.gitignore`). This protects potentially sensitive or copyrighted text data.
