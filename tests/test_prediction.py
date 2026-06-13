"""Tests for src/prediction — BoltzRunner, confidence_mapper, and simulator integration."""

import pytest

from src.pathway.assembly_graph import build_assembly_graph
from src.pathway.repair_simulator import simulate_repair
from src.prediction.boltz_runner import BoltzRunner, PredictionResult
from src.prediction.confidence_mapper import (
    prediction_summary,
    update_graph_from_predictions,
)

# All domain node IDs present in the assembly graph
_ALL_DOMAINS = [
    "titin__Z_disk_Ig",
    "titin__N2B_unique",
    "titin__PEVK",
    "titin__A_band_FnIII",
    "titin__M_line_kinase",
    "myosin_heavy_chain__motor_domain",
    "myosin_heavy_chain__lever_arm",
    "myosin_heavy_chain__rod_domain",
    "actin__actin_monomer",
    "troponin_i__inhibitory_region",
    "troponin_c__n_lobe",
    "troponin_c__c_lobe",
]


# ── BoltzRunner (mock mode) ────────────────────────────────────────────────────

class TestBoltzRunnerMock:
    def setup_method(self):
        self.runner = BoltzRunner(use_mock=True)

    def test_predict_domain_returns_prediction_result(self):
        result = self.runner.predict_domain("actin__actin_monomer")
        assert isinstance(result, PredictionResult)

    def test_is_mock_flagged(self):
        result = self.runner.predict_domain("actin__actin_monomer")
        assert result.is_mock is True

    def test_plddt_mean_in_valid_range(self):
        for domain_key in _ALL_DOMAINS:
            result = self.runner.predict_domain(domain_key)
            assert 0.0 <= result.plddt_mean <= 100.0, (
                f"{domain_key}: pLDDT {result.plddt_mean} out of range"
            )

    def test_folding_rate_equals_plddt_over_100(self):
        for domain_key in _ALL_DOMAINS:
            result = self.runner.predict_domain(domain_key)
            assert result.folding_rate == pytest.approx(result.plddt_mean / 100.0, rel=1e-3)

    def test_folding_rate_in_unit_interval(self):
        for domain_key in _ALL_DOMAINS:
            result = self.runner.predict_domain(domain_key)
            assert 0.0 <= result.folding_rate <= 1.0

    def test_unknown_domain_raises_key_error(self):
        with pytest.raises(KeyError, match="not found in prediction cache"):
            self.runner.predict_domain("fake_protein__fake_domain")

    def test_predict_all_covers_every_graph_node(self):
        predictions = self.runner.predict_all()
        G = build_assembly_graph()
        for node in G.nodes:
            assert node in predictions, f"Node '{node}' missing from predict_all() results"

    def test_predict_all_returns_prediction_results(self):
        predictions = self.runner.predict_all()
        for domain_key, result in predictions.items():
            assert isinstance(result, PredictionResult)
            assert result.domain_key == domain_key

    def test_residue_count_positive(self):
        for domain_key in _ALL_DOMAINS:
            result = self.runner.predict_domain(domain_key)
            assert result.residue_count > 0

    def test_disordered_domains_have_low_plddt(self):
        """Intrinsically disordered titin regions should have low confidence."""
        pevk = self.runner.predict_domain("titin__PEVK")
        n2b = self.runner.predict_domain("titin__N2B_unique")
        assert pevk.plddt_mean < 50.0
        assert n2b.plddt_mean < 50.0

    def test_structured_domains_have_high_plddt(self):
        """Well-folded domains should exceed the high-confidence threshold."""
        actin = self.runner.predict_domain("actin__actin_monomer")
        lever = self.runner.predict_domain("myosin_heavy_chain__lever_arm")
        assert actin.plddt_mean > 70.0
        assert lever.plddt_mean > 70.0


# ── Confidence mapper ──────────────────────────────────────────────────────────

class TestConfidenceMapper:
    def setup_method(self):
        runner = BoltzRunner(use_mock=True)
        self.predictions = runner.predict_all()
        self.G = build_assembly_graph()

    def test_update_graph_returns_digraph(self):
        import networkx as nx
        G_updated = update_graph_from_predictions(self.G, self.predictions)
        assert isinstance(G_updated, nx.DiGraph)

    def test_folding_rates_replaced_with_plddt(self):
        G_updated = update_graph_from_predictions(self.G, self.predictions)
        for node, data in G_updated.nodes(data=True):
            expected = self.predictions[node].folding_rate
            assert data["folding_rate"] == pytest.approx(expected, rel=1e-3), (
                f"{node}: folding_rate {data['folding_rate']} != expected {expected}"
            )

    def test_boltz_plddt_attribute_added(self):
        G_updated = update_graph_from_predictions(self.G, self.predictions)
        for node, data in G_updated.nodes(data=True):
            assert "boltz_plddt" in data, f"Node '{node}' missing 'boltz_plddt'"

    def test_boltz_is_mock_attribute_added(self):
        G_updated = update_graph_from_predictions(self.G, self.predictions)
        for node, data in G_updated.nodes(data=True):
            assert "boltz_is_mock" in data
            assert data["boltz_is_mock"] is True

    def test_all_folding_rates_in_unit_interval(self):
        G_updated = update_graph_from_predictions(self.G, self.predictions)
        for node, data in G_updated.nodes(data=True):
            assert 0.0 <= data["folding_rate"] <= 1.0

    def test_prediction_summary_returns_string(self):
        summary = prediction_summary(self.predictions)
        assert isinstance(summary, str)

    def test_prediction_summary_contains_all_domains(self):
        summary = prediction_summary(self.predictions)
        for domain_key in _ALL_DOMAINS:
            assert domain_key in summary


# ── Simulator integration ──────────────────────────────────────────────────────

class TestSimulatorBoltzIntegration:
    def test_use_boltz_returns_repair_report(self):
        from src.pathway.repair_simulator import RepairReport
        report = simulate_repair(use_boltz=True)
        assert isinstance(report, RepairReport)

    def test_boltz_repair_time_positive(self):
        report = simulate_repair(use_boltz=True)
        assert report.total_repair_time_h > 0

    def test_boltz_changes_repair_time_vs_baseline(self):
        """pLDDT-derived rates differ from hand-tuned, so total times should differ."""
        baseline = simulate_repair(use_boltz=False)
        boltz = simulate_repair(use_boltz=True)
        assert baseline.total_repair_time_h != pytest.approx(boltz.total_repair_time_h)

    def test_boltz_therapy_still_reduces_time(self):
        """Therapy optimisation should improve repair even with Boltz-2 rates."""
        boltz_baseline = simulate_repair(use_boltz=True)
        boltz_optimised = simulate_repair(
            use_boltz=True,
            therapy_params={"leucine_dose_g": 5.0, "sleep_quality": 1.0},
        )
        assert boltz_optimised.total_repair_time_h < boltz_baseline.total_repair_time_h

    def test_boltz_repair_score_in_range(self):
        report = simulate_repair(use_boltz=True)
        assert 0.0 < report.total_repair_score <= 1.0

    def test_boltz_bottlenecks_are_titin_structural_domains(self):
        """With Boltz rates, the titin structural domains (A-band FnIII, M-line kinase)
        have the lowest stability scores. Disordered titin regions (PEVK, N2B) are
        not assembly-graph nodes, so they never appear as bottlenecks."""
        report = simulate_repair(use_boltz=True)
        bottleneck_ids = [node_id for node_id, _ in report.bottlenecks]
        assert any("titin" in n for n in bottleneck_ids)
