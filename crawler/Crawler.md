# Search through directories or the internet for Insights

## Directory

Configure the crawler by running it and pointing it to the directory. Provide an output directory for the insights, knowledge graph, and vector storage.

```python
python crawl.py -i mydirectory/ -o outputdir/
```

In the output directory will be the following structure:

```python
vector.npz # the vector storage of embeddings
graph.sql # the knowledge graph for connections and relationships
metadata.sql # metadata gathered by the crawler
details.json # Summaries and other interpretations along with the initial link of the data
```
