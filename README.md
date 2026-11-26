# lit_ling - Linguistic Analysis Toolkit

A comprehensive Python toolkit for corpus-based linguistic analysis, focusing on lexical dispersion, collocation analysis, and semantic clustering. Designed for computational linguistics research on German political texts.

## Features

### 📊 Core Analysis Tools

- **Corpus Overview** (`korpus_uebersicht.py`): Generate statistical summaries of your text corpus
- **Dispersion Analysis** (`dispersion.py`, `dispersion_land.py`, `dispersion_ideologie_w2v.py`): Calculate lexical dispersion across different dimensions (party, ideology, country)
- **Collocation Analysis** (`kollo_ll.py`, `kollo_logratio.py`): Extract and rank collocations using log-likelihood and log-ratio statistics
- **Semantic Clustering** (`verteilung_cluster.py`, `visualisierung_cluster.py`): Word2Vec-based semantic clustering with k-means

### 📈 Visualization Tools

- **Interactive Collocation Visualizer** (`vis_collocation.py`): Create interactive HTML visualizations of collocation networks
- **Cluster Visualization** (`vis_cluster_insights_blocks.py`): Generate visual representations of word clusters

### 🛠️ Utilities

- **Data Cleaning** (`clean_tabs.py`): Preprocess and clean TSV corpus files
- **Threshold Selection** (`threshold_selection.py`): Determine optimal thresholds for dispersion metrics

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/lit_ling.git
cd lit_ling
```

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the German spaCy model (required for lemmatization):**
```bash
python -m spacy download de_core_news_sm
```

## Quick Start

### 1. Prepare Your Corpus

Your corpus should be a tab-separated file (`cleaned_file.tsv`) with at least these columns:
- `text`: The text content
- `partei`: Party affiliation (optional)
- `land`: Country code (CH, DE, OE) (optional)

Place your corpus file in the project root directory.

### 2. Generate Corpus Overview

```bash
python src/korpus_uebersicht.py
```

This creates statistical summaries in `outfiles/corpus_overview.tsv`.

### 3. Run Dispersion Analysis

```bash
python src/dispersion.py
```

This analyzes lexical dispersion and creates:
- `outfiles/evenly_distributed_words.tsv`
- `outfiles/unevenly_distributed_words.tsv`

### 4. Collocation Analysis

```bash
python src/kollo_logratio.py
```

Configure keywords in the script, then run to extract collocations with log-ratio statistics.

### 5. Semantic Clustering

```bash
python src/verteilung_cluster.py
```

Performs Word2Vec embeddings and k-means clustering on dispersed words.

## Configuration

Each script contains a `CONFIG` section at the top. Key parameters:

### Collocation Analysis (`kollo_logratio.py`)
```python
CORPUS_FILE = "cleaned_file.tsv"
KEYWORDS = ["weiblich", "weibliche", "weiblichen"]
WINDOW_SIZE = 5
USE_LEMMATIZATION = False  # Toggle lemmatization
```

### Dispersion Analysis (`dispersion.py`)
```python
INPUT_FILE = "cleaned_file.tsv"
OUTPUT_DIR = "outfiles"
MIN_FREQ = 10  # Minimum word frequency
```

### Clustering (`verteilung_cluster.py`)
```python
EMBEDDING_MODEL_NAME = "distilbert-base-german-cased"
N_CLUSTERS = 50
TSNE_PERPLEXITY = 5
```

## Project Structure

```
lit_ling/
├── src/                          # Source code
│   ├── clean_tabs.py            # Data preprocessing
│   ├── korpus_uebersicht.py     # Corpus statistics
│   ├── dispersion.py            # Dispersion analysis
│   ├── dispersion_land.py       # Country-based dispersion
│   ├── dispersion_ideologie_w2v.py  # Ideology-based analysis
│   ├── kollo_ll.py              # Log-likelihood collocations
│   ├── kollo_logratio.py        # Log-ratio collocations
│   ├── verteilung_cluster.py    # Semantic clustering
│   ├── visualisierung_cluster.py # Cluster visualization
│   ├── vis_collocation.py       # Collocation network viz
│   ├── vis_cluster_insights_blocks.py
│   └── threshold_selection.py   # Threshold optimization
├── data/                         # Input data (add your corpus here)
├── outfiles/                     # Analysis outputs
├── docs/                         # Documentation
├── examples/                     # Example usage
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Output Files

Analysis results are saved in `outfiles/` with subdirectories:

- `outfiles/corpus_overview.tsv` - Corpus statistics
- `outfiles/evenly_distributed_words.tsv` - Words with even distribution
- `outfiles/unevenly_distributed_words.tsv` - Words with uneven distribution
- `outfiles_koll_kwic_ling/` - KWIC concordances and collocation data
- `outfiles/clusters/` - Clustering results and visualizations

## Methodology

### Dispersion Metrics
The toolkit uses several dispersion measures:
- **Gries' DP (Deviation of Proportions)**: Measures how evenly a word is distributed across corpus parts
- **Log-Ratio**: Compares observed vs. expected frequencies
- **Log-Likelihood**: Statistical significance of collocation associations

### Lemmatization
Optional lemmatization using spaCy preserves German capitalization conventions (important for nouns and proper names).

### Semantic Similarity
Uses transformer-based embeddings (DistilBERT) for German, capturing contextual semantic relationships.

## Citation

If you use this toolkit in your research, please cite:

```
[Your citation information here]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Contact

davide.ventre@uzh.ch

## Acknowledgments

This toolkit was developed for computational linguistic research on German political discourse.

### Key Dependencies
- **spaCy**: NLP preprocessing and lemmatization
- **gensim**: Word2Vec embeddings
- **sentence-transformers**: Transformer-based embeddings
- **scikit-learn**: Clustering and dimensionality reduction
- **plotly**: Interactive visualizations
