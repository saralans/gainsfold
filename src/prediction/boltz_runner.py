"""
Boltz-2 structure prediction runner.

CPU-local mode (default): returns cached pLDDT scores from
src/prediction/cache/predictions.json — no GPU or network needed.

GPU mode (use_mock=False): shells out to `boltz predict`, parses the
per-residue confidence JSON, and updates the cache.  Requires boltz>=1.0.0
(uncomment in requirements.txt) and a CUDA device with >=48 GB VRAM,
or a cloud GPU via Modal / RunPod — swap _run_boltz_predict() as needed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_CACHE_PATH = Path(__file__).parent / "cache" / "predictions.json"
_BOLTZ_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs" / "boltz"
_PROTEINS_CONFIG = Path(__file__).parent.parent.parent / "configs" / "proteins.yaml"


@dataclass
class PredictionResult:
    domain_key: str
    plddt_mean: float
    residue_count: int
    is_mock: bool
    plddt_per_residue: list[float] = field(default_factory=list)

    @property
    def folding_rate(self) -> float:
        """pLDDT [0, 100] → folding_rate [0, 1]."""
        return round(self.plddt_mean / 100.0, 4)


class BoltzRunner:
    """Predict per-domain structural confidence scores using Boltz-2.

    Parameters
    ----------
    cache_path:
        JSON file with pre-computed (or mock) pLDDT scores.
    output_dir:
        Directory where boltz writes structure files on GPU runs.
    use_mock:
        When True (default) always return cached scores without calling boltz.
        Set to False on a GPU node after `pip install boltz`.
    """

    def __init__(
        self,
        cache_path: Path = _CACHE_PATH,
        output_dir: Path | None = None,
        use_mock: bool = True,
    ) -> None:
        self._cache_path = cache_path
        self._output_dir = output_dir or Path("outputs/boltz")
        self._use_mock = use_mock
        self._cache: dict[str, Any] = self._load_cache_file()

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict_domain(self, domain_key: str) -> PredictionResult:
        """Return a PredictionResult for one domain node (e.g. 'actin__actin_monomer').

        Falls back to cache if boltz is unavailable, regardless of use_mock flag.
        """
        cached = self._cache.get(domain_key)
        if self._use_mock or not self._boltz_available():
            if cached is None:
                raise KeyError(
                    f"Domain '{domain_key}' not found in prediction cache "
                    f"({self._cache_path}). Add a mock entry or run with use_mock=False "
                    "on a GPU node."
                )
            return PredictionResult(
                domain_key=domain_key,
                plddt_mean=cached["plddt_mean"],
                residue_count=cached["residue_count"],
                is_mock=True,
            )
        return self._run_real_prediction(domain_key)

    def predict_all(self, proteins_config: dict | None = None) -> dict[str, PredictionResult]:
        """Predict all domain nodes found in proteins.yaml (or a supplied config dict).

        Returns a mapping of node_id → PredictionResult covering every node
        that would appear in the assembly graph.
        """
        if proteins_config is None:
            with open(_PROTEINS_CONFIG) as f:
                proteins_config = yaml.safe_load(f)

        results: dict[str, PredictionResult] = {}
        for prot_key, prot in proteins_config.get("proteins", {}).items():
            for domain in prot.get("domains", []):
                node_id = f"{prot_key}__{domain['name']}"
                results[node_id] = self.predict_domain(node_id)
        return results

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _load_cache_file(self) -> dict[str, Any]:
        if not self._cache_path.exists():
            return {}
        with open(self._cache_path) as f:
            data = json.load(f)
        return data.get("domains", {})

    def _save_to_cache(self, result: PredictionResult) -> None:
        """Persist a real (non-mock) prediction result into the cache file."""
        with open(self._cache_path) as f:
            full = json.load(f)
        full["domains"][result.domain_key] = {
            "plddt_mean": result.plddt_mean,
            "residue_count": result.residue_count,
            "is_mock": False,
            "plddt_per_residue": result.plddt_per_residue,
        }
        with open(self._cache_path, "w") as f:
            json.dump(full, f, indent=2)
        self._cache[result.domain_key] = full["domains"][result.domain_key]

    # ── GPU execution path ─────────────────────────────────────────────────────

    def _boltz_available(self) -> bool:
        return shutil.which("boltz") is not None

    def _run_real_prediction(self, domain_key: str) -> PredictionResult:
        """Fetch sequence, write Boltz-2 YAML, run prediction, parse confidence."""
        prot_key, domain_name = domain_key.split("__", 1)
        domain_meta = self._domain_meta(prot_key, domain_name)
        sequence = self._fetch_sequence(
            domain_meta["uniprot_id"],
            domain_meta["start"],
            domain_meta["end"],
        )
        input_yaml = self._write_boltz_input(domain_key, sequence)
        out_dir = self._run_boltz_predict(input_yaml)
        result = self._parse_confidence(out_dir, domain_key, len(sequence))
        self._save_to_cache(result)
        return result

    def _domain_meta(self, prot_key: str, domain_name: str) -> dict:
        with open(_PROTEINS_CONFIG) as f:
            config = yaml.safe_load(f)
        prot = config["proteins"][prot_key]
        for domain in prot["domains"]:
            if domain["name"] == domain_name:
                return {
                    "uniprot_id": prot["uniprot_id"],
                    "start": domain["start"],
                    "end": domain["end"],
                }
        raise KeyError(f"Domain '{domain_name}' not found for protein '{prot_key}'")

    def _fetch_sequence(self, uniprot_id: str, start: int, end: int) -> str:
        """Download the canonical sequence from UniProt and slice to domain range.

        Requires biopython (already in requirements.txt).
        """
        try:
            from Bio import ExPASy, SeqIO
        except ImportError as exc:
            raise RuntimeError(
                "biopython is required for real predictions: pip install biopython"
            ) from exc

        with ExPASy.get_sprot_raw(uniprot_id) as handle:
            record = SeqIO.read(handle, "swiss")
        # UniProt residue numbering is 1-based, inclusive
        return str(record.seq[start - 1 : end])

    def _write_boltz_input(self, domain_key: str, sequence: str) -> Path:
        """Write a Boltz-2-format YAML input file to a temp directory."""
        boltz_input = {
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": sequence,
                    }
                }
            ]
        }
        tmp = Path(tempfile.mkdtemp(prefix=f"boltz_{domain_key}_"))
        input_path = tmp / "input.yaml"
        with open(input_path, "w") as f:
            yaml.dump(boltz_input, f, default_flow_style=False)
        return input_path

    def _run_boltz_predict(self, input_yaml: Path) -> Path:
        """Shell out to the boltz CLI and return the output directory.

        Cloud GPU swap: replace this method body to POST to Modal / RunPod
        and return the downloaded output directory path.
        """
        out_dir = self._output_dir / input_yaml.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["boltz", "predict", str(input_yaml), "--out_dir", str(out_dir), "--device", "cuda"],
            check=True,
        )
        return out_dir

    def _parse_confidence(
        self, out_dir: Path, domain_key: str, residue_count: int
    ) -> PredictionResult:
        """Parse boltz confidence_model_0.json → PredictionResult."""
        confidence_file = out_dir / "predictions" / "input" / "confidence_model_0.json"
        if not confidence_file.exists():
            raise FileNotFoundError(
                f"Boltz confidence file not found at {confidence_file}. "
                "Check that boltz predict completed successfully."
            )
        with open(confidence_file) as f:
            data = json.load(f)

        # Boltz outputs pLDDT per token (residue) as a flat list
        plddt_per_residue: list[float] = data.get("plddt", [])
        plddt_mean = (
            sum(plddt_per_residue) / len(plddt_per_residue)
            if plddt_per_residue
            else 0.0
        )
        return PredictionResult(
            domain_key=domain_key,
            plddt_mean=round(plddt_mean, 2),
            residue_count=residue_count,
            is_mock=False,
            plddt_per_residue=plddt_per_residue,
        )
