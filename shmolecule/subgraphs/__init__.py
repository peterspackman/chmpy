from pathlib import Path
import logging

__all__ = ["load_data"]
DIR = Path(__file__).parent
LOG = logging.getLogger(__name__)


def load_data():
    import graph_tool as gt

    graphs = {}
    LOG.info("Loading graph data from %s", DIR)
    for fname in DIR.glob("*.gt"):
        graphs[fname.stem] = gt.load_graph(str(fname))
    return graphs
