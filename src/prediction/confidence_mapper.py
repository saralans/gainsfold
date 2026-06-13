"""
Maps Boltz-2 pLDDT confidence scores onto the assembly graph.

folding_rate is replaced entirely by pLDDT / 100 for any node that has a
prediction result.  Nodes with no prediction retain their original value.
"""

from __future__ import annotations

import networkx as nx

from src.prediction.boltz_runner import BoltzRunner, PredictionResult


def update_graph_from_predictions(
    G: nx.DiGraph,
    predictions: dict[str, PredictionResult],
) -> nx.DiGraph:
    """Replace folding_rate on each graph node with Boltz-2 pLDDT / 100.

    Nodes not present in predictions are left unchanged.  Each updated node
    gains two additional attributes:
      - boltz_plddt   : raw mean pLDDT from Boltz-2 (0–100)
      - boltz_is_mock : True if the score came from the mock cache

    Parameters
    ----------
    G : nx.DiGraph
        Assembly graph returned by build_assembly_graph().
    predictions : dict[str, PredictionResult]
        Mapping of node_id → PredictionResult, as returned by BoltzRunner.predict_all().

    Returns
    -------
    The same graph G with node attributes updated in place.
    """
    for node in G.nodes:
        result = predictions.get(node)
        if result is None:
            continue
        G.nodes[node]["folding_rate"] = result.folding_rate
        G.nodes[node]["boltz_plddt"] = result.plddt_mean
        G.nodes[node]["boltz_is_mock"] = result.is_mock
    return G


def prediction_summary(predictions: dict[str, PredictionResult]) -> str:
    """Return a human-readable table of pLDDT scores for all predicted domains."""
    lines = [
        "=== Boltz-2 Confidence Scores ===",
        f"{'Domain':<45}  {'pLDDT':>7}  {'Rate':>6}  {'Mock?':>5}",
        "-" * 70,
    ]
    for domain_key, result in sorted(predictions.items()):
        lines.append(
            f"{domain_key:<45}  {result.plddt_mean:>7.1f}  {result.folding_rate:>6.4f}"
            f"  {'yes' if result.is_mock else 'no':>5}"
        )
    return "\n".join(lines)
