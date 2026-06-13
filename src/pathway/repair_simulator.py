"""
Sarcomere repair simulator.

Walks the assembly graph in step order and estimates a repair timeline based on
per-domain folding rates and binding affinities. Therapy parameters (from
configs/proteins.yaml therapy_space) can be passed to modify rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.pathway.assembly_graph import (
    assembly_step_nodes,
    bottleneck_nodes,
    build_assembly_graph,
    node_stability_score,
)


_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "proteins.yaml"


@dataclass
class StepResult:
    step_index: int
    step_name: str
    nodes: list[str]
    stability_scores: dict[str, float]
    mean_stability: float
    estimated_duration_h: float


@dataclass
class RepairReport:
    steps: list[StepResult] = field(default_factory=list)
    total_repair_time_h: float = 0.0
    bottlenecks: list[tuple[str, float]] = field(default_factory=list)
    therapy_params: dict[str, float] = field(default_factory=dict)

    @property
    def total_repair_score(self) -> float:
        """Higher is better: inverse of normalized repair time."""
        if self.total_repair_time_h <= 0:
            return 0.0
        # Normalize against 96h maximum rest window; score ∈ (0, 1]
        return min(1.0, 96.0 / self.total_repair_time_h)


def _therapy_rate_modifier(therapy_params: dict[str, float], config: dict) -> float:
    """Return a global rate multiplier from therapy parameters.

    Each parameter contributes a multiplicative modifier per the formulas
    documented in configs/proteins.yaml therapy_space.
    """
    space = config.get("therapy_space", {})
    modifier = 1.0

    leucine = therapy_params.get("leucine_dose_g", 0.0)
    if "leucine_dose_g" in space:
        modifier *= 1.0 + 0.15 * min(leucine, 5.0)

    sleep = therapy_params.get("sleep_quality", 0.5)
    if "sleep_quality" in space:
        modifier *= 0.7 + 0.6 * max(0.0, min(sleep, 1.0))

    temp = therapy_params.get("body_temperature_c", 37.0)
    if "body_temperature_c" in space:
        ref_temp = space["body_temperature_c"].get("reference_temp", 37.0)
        ea_kj = space["body_temperature_c"].get("activation_energy_kJ", 50.0)
        R = 0.008314  # kJ / (mol·K)
        T_ref = ref_temp + 273.15
        T = temp + 273.15
        arrhenius = math.exp((ea_kj / R) * (1.0 / T_ref - 1.0 / T))
        modifier *= arrhenius

    return max(0.01, modifier)


def _step_duration(mean_stability: float, rate_modifier: float) -> float:
    """Estimate repair duration in hours for one assembly step.

    Lower stability → longer duration. Base duration at stability=1.0 is 6h;
    at stability=0.0 it asymptotes to ~48h. Rate modifier compresses this.
    """
    # Stability ∈ [0, 1]: map to duration via inverse relationship
    base_duration = 6.0 + 42.0 * (1.0 - mean_stability) ** 1.5
    return base_duration / max(rate_modifier, 0.01)


def simulate_repair(
    therapy_params: dict[str, float] | None = None,
    config_path: Path = _CONFIG_PATH,
    use_boltz: bool = False,
) -> RepairReport:
    """Run the sarcomere repair simulation.

    Parameters
    ----------
    therapy_params:
        Optional dict with keys matching therapy_space in proteins.yaml.
        Defaults to baseline (no intervention).
    use_boltz:
        When True, replace hand-tuned folding rates with Boltz-2 pLDDT scores
        before running the simulation.  Uses the mock cache by default (no GPU
        required); set BoltzRunner(use_mock=False) on a GPU node for real
        predictions.  See src/prediction/boltz_runner.py.
    config_path:
        Path to proteins.yaml.

    Returns
    -------
    RepairReport with per-step results and total estimated repair time.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    therapy_params = therapy_params or {}
    G = build_assembly_graph(config)

    if use_boltz:
        from src.prediction.boltz_runner import BoltzRunner
        from src.prediction.confidence_mapper import update_graph_from_predictions
        runner = BoltzRunner(use_mock=True)
        predictions = runner.predict_all(config)
        G = update_graph_from_predictions(G, predictions)

    rate_modifier = _therapy_rate_modifier(therapy_params, config)

    step_map = assembly_step_nodes(G)
    step_names = {
        s["step"]: s["name"] for s in config.get("assembly_order", [])
    }

    report = RepairReport(therapy_params=therapy_params)

    for step_idx, nodes in step_map.items():
        scores = {n: node_stability_score(G, n) for n in nodes}
        mean_stab = sum(scores.values()) / len(scores)
        duration = _step_duration(mean_stab, rate_modifier)

        report.steps.append(
            StepResult(
                step_index=step_idx,
                step_name=step_names.get(step_idx, f"step_{step_idx}"),
                nodes=nodes,
                stability_scores=scores,
                mean_stability=round(mean_stab, 4),
                estimated_duration_h=round(duration, 2),
            )
        )

    report.total_repair_time_h = round(
        sum(s.estimated_duration_h for s in report.steps), 2
    )
    report.bottlenecks = bottleneck_nodes(G, top_n=3)
    return report


def optimal_therapy_summary(report: RepairReport) -> str:
    """Return a human-readable summary of the repair report."""
    lines = [
        "=== Sarcomere Repair Simulation ===",
        f"Therapy params: {report.therapy_params or 'baseline (no intervention)'}",
        f"Total estimated repair time: {report.total_repair_time_h:.1f} h",
        f"Repair score: {report.total_repair_score:.3f}  (higher = faster recovery)",
        "",
        "Assembly steps:",
    ]
    for step in report.steps:
        lines.append(
            f"  Step {step.step_index} [{step.step_name}]  "
            f"stability={step.mean_stability:.3f}  "
            f"duration={step.estimated_duration_h:.1f}h"
        )

    lines += [
        "",
        "Bottleneck domains (lowest stability):",
    ]
    for node_id, score in report.bottlenecks:
        lines.append(f"  {node_id}  score={score:.3f}")

    return "\n".join(lines)
