from google_labs_html_chunker.html_chunker import HtmlChunker
from urllib.request import urlopen

with urlopen(url) as f:
    html = f.read().decode("utf-8")

chunker = HtmlChunker(
    max_words_per_aggregate_passage=200,
    greedily_aggregate_sibling_nodes=True,
    html_tags_to_exclude={"noscript", "script", "style"},
)
passages = chunker.chunk(html)