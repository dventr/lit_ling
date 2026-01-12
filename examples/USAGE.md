# Example Usage

## Basic Workflow Example

This example demonstrates a complete analysis workflow from corpus overview to semantic clustering.

### Step 1: Corpus Overview

```python
# Run corpus statistics
python src/korpus_uebersicht.py
```

**Output**: `outfiles/corpus_overview.tsv`

### Step 2: Dispersion Analysis

```python
# Analyze lexical dispersion across parties
python src/dispersion.py
```

**Output**: 
- `outfiles/evenly_distributed_words.tsv`
- `outfiles/unevenly_distributed_words.tsv`

### Step 3: Collocation Analysis for Keywords

Edit `src/kollo_logratio.py` to set your keywords:

```python
KEYWORDS = [
    "Arbeitsmarkt",
    "Ausbildung",
    "Berufsausbildung"
]
WINDOW_SIZE = 5
USE_LEMMATIZATION = False
```

Then run:

```python
python src/kollo_logratio.py
```

**Output**: `outfiles_koll_kwic_ling/collocations_*.tsv` and KWIC files

### Step 4: Semantic Clustering

```python
# Cluster words by semantic similarity
python src/verteilung_cluster.py
```

**Output**: 
- `outfiles/clusters/*_embedding_clusters.tsv`
- t-SNE visualizations

### Step 5: Interactive Visualization

```python
# Create interactive collocation network
python src/vis_collocation.py
```

**Output**: Interactive HTML visualizations in `viz_out/`

## Advanced Configuration Examples

### Example 1: Country-Specific Analysis

```python
# Edit dispersion_land.py
GROUP_FIELD = "land"  # Analyze by country

python src/dispersion_land.py
```

### Example 2: Ideology-Based Clustering

```python
# Edit dispersion_ideologie_w2v.py
# Uses Word2Vec embeddings with ideological groupings

python src/dispersion_ideologie_w2v.py
```

### Example 3: Custom Collocation Window

```python
# In kollo_logratio.py
WINDOW_SIZE = 10  # Larger context window
USE_LEMMATIZATION = True  # Enable lemmatization

python src/kollo_logratio.py
```

## Interpreting Results

### Dispersion Metrics

- **DP (Deviation of Proportions)**: 0 = perfectly even distribution, 1 = maximally uneven
- **Low DP**: Words used consistently across all groups
- **High DP**: Words specific to certain groups/parties

### Log-Ratio Scores

- **Positive log-ratio**: Word appears more than expected with keyword
- **Negative log-ratio**: Word appears less than expected
- **Higher absolute values**: Stronger association

### Cluster Interpretation

Examine cluster summaries to identify:
- Semantic fields (e.g., economic terminology, social policy terms)
- Thematic groupings
- Party-specific vocabulary clusters

## Tips

1. **Start small**: Test with a subset of your corpus first
2. **Adjust thresholds**: Use `threshold_selection.py` to optimize MIN_FREQ
3. **Compare methods**: Run both log-likelihood and log-ratio for different insights
4. **Visualize**: Always check visualizations to validate quantitative results
