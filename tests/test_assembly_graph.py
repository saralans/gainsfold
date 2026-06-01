"""Tests for src/pathway/assembly_graph.py and src/pathway/repair_simulator.py"""

import math
from pathlib import Path

import networkx as nx
import pytest

from src.pathway.assembly_graph import (
    assembly_step_nodes,
    bottleneck_nodes,
    build_assembly_graph,
    node_stability_score,
)
from src.pathway.repair_simulator import (
    RepairReport,
    simulate_repair,
    optimal_therapy_summary,
)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "proteins.yaml"


# ── Assembly graph ─────────────────────────────────────────────────────────────

class TestBuildAssemblyGraph:
    def test_returns_directed_graph(self):
        G = build_assembly_graph()
        assert isinstance(G, nx.DiGraph)

    def test_has_nodes(self):
        G = build_assembly_graph()
        assert G.number_of_nodes() > 0

    def test_has_edges(self):
        G = build_assembly_graph()
        assert G.number_of_edges() > 0

    def test_node_has_required_attributes(self):
        G = build_assembly_graph()
        for node, data in G.nodes(data=True):
            assert "step" in data, f"Node {node} missing 'step'"
            assert "folding_rate" in data, f"Node {node} missing 'folding_rate'"
            assert "protein" in data, f"Node {node} missing 'protein'"

    def test_folding_rates_in_range(self):
        G = build_assembly_graph()
        for node, data in G.nodes(data=True):
            rate = data["folding_rate"]
            assert 0.0 <= rate <= 1.0, f"folding_rate {rate} out of range for {node}"

    def test_known_proteins_present(self):
        G = build_assembly_graph()
        proteins_in_graph = {data["protein"] for _, data in G.nodes(data=True)}
        assert "actin" in proteins_in_graph
        assert "myosin_heavy_chain" in proteins_in_graph
        assert "titin" in proteins_in_graph

    def test_actin_myosin_edge_exists(self):
        G = build_assembly_graph()
        # There should be at least one edge involving actin and myosin
        actin_nodes = [n for n in G if n.startswith("actin__")]
        myosin_nodes = [n for n in G if n.startswith("myosin_heavy_chain__")]
        cross_edges = [
            (u, v) for u, v in G.edges()
            if (u in actin_nodes and v in myosin_nodes)
            or (u in myosin_nodes and v in actin_nodes)
        ]
        assert len(cross_edges) > 0, "No actin–myosin edges found"


class TestAssemblyStepNodes:
    def test_returns_dict_keyed_by_int(self):
        G = build_assembly_graph()
        steps = assembly_step_nodes(G)
        assert isinstance(steps, dict)
        for k in steps:
            assert isinstance(k, int)

    def test_steps_are_sorted(self):
        G = build_assembly_graph()
        steps = assembly_step_nodes(G)
        keys = list(steps.keys())
        assert keys == sorted(keys)

    def test_all_nodes_covered(self):
        G = build_assembly_graph()
        steps = assembly_step_nodes(G)
        all_step_nodes = [n for nodes in steps.values() for n in nodes]
        assert set(all_step_nodes) == set(G.nodes)


class TestNodeStabilityScore:
    def test_score_in_range(self):
        G = build_assembly_graph()
        for node in G.nodes:
            score = node_stability_score(G, node)
            assert 0.0 <= score <= 1.0, f"Stability score {score} out of range for {node}"

    def test_high_folding_rate_yields_higher_score(self):
        """A node with folding_rate=1.0 and no edges should score 1.0."""
        G = nx.DiGraph()
        G.add_node("test__domain", folding_rate=1.0, protein="test", step=1)
        assert node_stability_score(G, "test__domain") == pytest.approx(1.0)

    def test_low_folding_rate_yields_lower_score(self):
        G_high = nx.DiGraph()
        G_high.add_node("x__d", folding_rate=0.9, protein="x", step=1)
        G_low = nx.DiGraph()
        G_low.add_node("x__d", folding_rate=0.2, protein="x", step=1)
        assert node_stability_score(G_high, "x__d") > node_stability_score(G_low, "x__d")


class TestBottleneckNodes:
    def test_returns_correct_count(self):
        G = build_assembly_graph()
        bns = bottleneck_nodes(G, top_n=2)
        assert len(bns) == 2

    def test_sorted_ascending(self):
        G = build_assembly_graph()
        bns = bottleneck_nodes(G, top_n=3)
        scores = [s for _, s in bns]
        assert scores == sorted(scores)

    def test_default_top_n(self):
        G = build_assembly_graph()
        bns = bottleneck_nodes(G)
        assert len(bns) == 3


# ── Repair simulator ───────────────────────────────────────────────────────────

class TestSimulateRepair:
    def test_returns_repair_report(self):
        report = simulate_repair()
        assert isinstance(report, RepairReport)

    def test_positive_repair_time(self):
        report = simulate_repair()
        assert report.total_repair_time_h > 0

    def test_has_steps(self):
        report = simulate_repair()
        assert len(report.steps) > 0

    def test_each_step_has_positive_duration(self):
        report = simulate_repair()
        for step in report.steps:
            assert step.estimated_duration_h > 0

    def test_total_equals_sum_of_steps(self):
        report = simulate_repair()
        computed = sum(s.estimated_duration_h for s in report.steps)
        assert report.total_repair_time_h == pytest.approx(computed, rel=1e-3)

    def test_therapy_reduces_repair_time(self):
        """Optimal therapy params should yield shorter repair time than baseline."""
        baseline = simulate_repair()
        optimized = simulate_repair(therapy_params={
            "leucine_dose_g": 5.0,
            "sleep_quality": 1.0,
            "body_temperature_c": 37.0,
        })
        assert optimized.total_repair_time_h < baseline.total_repair_time_h

    def test_repair_score_in_range(self):
        report = simulate_repair()
        assert 0.0 < report.total_repair_score <= 1.0

    def test_bottlenecks_present(self):
        report = simulate_repair()
        assert len(report.bottlenecks) == 3


class TestOptimalTherapySummary:
    def test_returns_string(self):
        report = simulate_repair()
        summary = optimal_therapy_summary(report)
        assert isinstance(summary, str)

    def test_contains_repair_time(self):
        report = simulate_repair()
        summary = optimal_therapy_summary(report)
        assert "repair time" in summary.lower()

    def test_contains_bottleneck_section(self):
        report = simulate_repair()
        summary = optimal_therapy_summary(report)
        assert "bottleneck" in summary.lower()
