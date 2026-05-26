"""
Sarcomere assembly graph.

Models the ordered binding events that reconstruct a sarcomere after microtear
damage. Each node is a protein domain; each directed edge is a binding event
that must occur for assembly to proceed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import networkx as nx
import yaml


_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "proteins.yaml"


def _load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_assembly_graph(config: dict | None = None) -> nx.DiGraph:
    """Return a DiGraph encoding the sarcomere assembly sequence.

    Nodes carry:
        - protein: protein key in config
        - domain: domain name
        - residues: domain length
        - folding_rate: intrinsic rate constant [0, 1]
        - function: biological role
        - step: assembly step index (lower = earlier)

    Edges carry:
        - kd_nM: dissociation constant in nM (lower = tighter binding)
        - binding_condition: experimental condition for the Kd value
    """
    if config is None:
        config = _load_config()

    G = nx.DiGraph()

    # Build a fast lookup: domain name → folding_rate
    domain_meta: dict[str, dict] = {}
    for prot_key, prot in config["proteins"].items():
        for domain in prot.get("domains", []):
            node_id = f"{prot_key}__{domain['name']}"
            domain_meta[node_id] = {
                "protein": prot_key,
                "domain": domain["name"],
                "residues": domain["end"] - domain["start"] + 1,
                "folding_rate": domain["folding_rate"],
                "function": domain["function"],
                "domain_type": domain["type"],
            }

    # Add nodes from the ordered assembly steps
    for step in config["assembly_order"]:
        step_idx = step["step"]
        for prot_key, domain_name in zip(step["proteins"], step["domains"]):
            node_id = f"{prot_key}__{domain_name}"
            if node_id not in G:
                meta = domain_meta.get(node_id, {})
                G.add_node(node_id, step=step_idx, assembly_step=step["name"], **meta)

    # Add binding edges using known affinities
    affinities = config.get("binding_affinities", {})
    _add_edges_from_affinities(G, affinities, config)

    # Add sequential dependency edges between consecutive assembly steps
    _add_step_dependency_edges(G, config)

    return G


def _add_edges_from_affinities(
    G: nx.DiGraph, affinities: dict, config: dict
) -> None:
    """Add edges for documented protein–protein binding events."""
    for interaction_name, aff in affinities.items():
        partners = aff["partners"]
        if len(partners) < 2:
            continue
        kd = aff["kd_nM"]
        condition = aff.get("condition", "")

        # Find nodes in G that match each partner
        partner_nodes: list[list[str]] = []
        for partner_key in partners:
            matched = [n for n in G if n.startswith(f"{partner_key}__")]
            partner_nodes.append(matched)

        # Create an edge between every cross-partner node pair
        for src in partner_nodes[0]:
            for dst in partner_nodes[1]:
                if src != dst:
                    G.add_edge(
                        src,
                        dst,
                        interaction=interaction_name,
                        kd_nM=kd,
                        binding_condition=condition,
                    )


def _add_step_dependency_edges(G: nx.DiGraph, config: dict) -> None:
    """Ensure step N nodes depend on step N-1 nodes via 'precedes' edges."""
    steps: dict[int, list[str]] = {}
    for node, data in G.nodes(data=True):
        s = data.get("step", 0)
        steps.setdefault(s, []).append(node)

    sorted_steps = sorted(steps.keys())
    for i in range(len(sorted_steps) - 1):
        prev_step = sorted_steps[i]
        next_step = sorted_steps[i + 1]
        for src in steps[prev_step]:
            for dst in steps[next_step]:
                if not G.has_edge(src, dst):
                    G.add_edge(src, dst, interaction="sequential_dependency", kd_nM=None)


def node_stability_score(G: nx.DiGraph, node_id: str) -> float:
    """Composite stability score for a single domain node.

    Score ∈ [0, 1]. Combines:
      - folding_rate of the domain itself
      - binding strength (1/Kd) of all incident edges (normalized)
    """
    data = G.nodes[node_id]
    base = data.get("folding_rate", 0.5)

    incident_kds = [
        d["kd_nM"]
        for _, _, d in G.edges(node_id, data=True)
        if d.get("kd_nM") is not None
    ]
    if not incident_kds:
        return base

    # Use geometric mean of binding affinities, scaled to [0, 1]
    # Lower Kd → tighter → better score. We invert: score = 1 / (1 + log10(Kd))
    aff_scores = [1.0 / (1.0 + math.log10(max(kd, 0.1))) for kd in incident_kds]
    mean_aff = sum(aff_scores) / len(aff_scores)

    return 0.6 * base + 0.4 * mean_aff


def assembly_step_nodes(G: nx.DiGraph) -> dict[int, list[str]]:
    """Return nodes grouped by assembly step index."""
    steps: dict[int, list[str]] = {}
    for node, data in G.nodes(data=True):
        s = data.get("step", 0)
        steps.setdefault(s, []).append(node)
    return dict(sorted(steps.items()))


def bottleneck_nodes(G: nx.DiGraph, top_n: int = 3) -> list[tuple[str, float]]:
    """Return the top_n nodes with the lowest stability scores (bottlenecks)."""
    scores = [(n, node_stability_score(G, n)) for n in G.nodes]
    return sorted(scores, key=lambda x: x[1])[:top_n]
