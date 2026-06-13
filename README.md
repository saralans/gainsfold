---
title: GainsFold
emoji: 🤗
colorFrom: purple
colorTo: gray
sdk: gradio
pinned: false
---

# Sarcomere Repair Optimizer

Computational pipeline for modeling sarcomere reconstruction after resistance-training-induced microtears. Models the ordered folding and assembly of giant muscle proteins — **titin, myosin, actin, and troponin** — and estimates how therapeutic interventions (leucine supplementation, sleep quality, temperature) reduce recovery time and DOMS duration.

Built with the [Boltz](https://github.com/jwohlwend/boltz) architecture (MIT, 2024) as the structural prediction backbone, with a CPU-runnable repair pathway model and therapy optimizer on top.

---

## Live demo

**[huggingface.co/spaces/saralagold/gainsfold](https://huggingface.co/spaces/saralagold/gainsfold)**

An interactive Gradio app deployed on Hugging Face Spaces (free CPU tier, auto-deployed from this repo via GitHub Actions). No setup required — adjust the therapy sliders and run the simulation in the browser.

- **Repair Simulator** — set leucine dose, sleep quality, and body temperature; outputs total repair time, per-step breakdown, domain stability chart, and bottleneck risk table comparing baseline vs optimised protocol
- **Leucine Sensitivity** — sweeps 0–10g leucine to show where mTOR activation saturates and returns diminish
- **About** — formulas, protein table, and literature references

Boltz-2 pLDDT scores run from the pre-computed cache on CPU. Real structure predictions require a GPU Space (≥48 GB VRAM).

---

## Motivation

When muscle fibers tear during resistance training, sarcomere reconstruction requires the rapid folding and assembly of some of the largest and most complex proteins in the human body. Titin alone spans 34,350 residues. Modeling these repair pathways computationally was previously intractable; advances in large-complex protein prediction now make it possible to map structural repair sequences and identify the rate-limiting steps.

This project applies that capability to a practical question: **what interventions most effectively shorten the repair window?**

---

## Features (this branch: `feature/protein-repair`)

- **Assembly graph** — NetworkX DiGraph encoding the four-step sarcomere assembly sequence (Z-disk anchoring → thin filament elongation → thick filament integration → titin scaffold attachment)
- **Domain stability scoring** — composite metric combining per-domain folding rates and experimental binding affinities (Kd values from literature)
- **Repair simulator** — estimates assembly duration per step with Arrhenius-based temperature correction and mTOR/GH modifiers for leucine and sleep inputs
- **Bottleneck detection** — identifies which domains are the weakest links in reconstruction
- **Therapy sensitivity analysis** — sweep leucine dose and sleep quality to find the optimal recovery protocol

---

## Project structure

```
protein_repair/
├── configs/proteins.yaml          # Protein metadata, domain boundaries, Kd values
├── src/
│   └── pathway/
│       ├── assembly_graph.py      # Build and query the sarcomere assembly graph
│       └── repair_simulator.py   # Simulate repair timeline with therapy params
├── tests/
│   └── test_assembly_graph.py    # 27 unit tests (pytest)
├── notebooks/
│   └── 03_repair_pathway.ipynb   # End-to-end walkthrough with visualizations
└── requirements.txt
```

---

## Quick start

```bash
git clone <repo-url>
cd protein_predicition
pip install -r requirements.txt

# Run tests
python3 -m pytest tests/ -v

# Launch notebook
jupyter notebook notebooks/03_repair_pathway.ipynb
```

### Run a repair simulation in Python

```python
from src.pathway.repair_simulator import simulate_repair, optimal_therapy_summary

# Baseline (no intervention)
baseline = simulate_repair()
print(f"Baseline repair time: {baseline.total_repair_time_h:.1f}h")

# Optimised therapy
optimised = simulate_repair(therapy_params={
    "leucine_dose_g": 5.0,
    "sleep_quality": 0.9,
    "body_temperature_c": 37.2,
})
print(optimal_therapy_summary(optimised))
```

---

## Run structure predictions on GPU (Boltz)

The `data/boltz_inputs/` directory contains pre-built Boltz-2 YAML configs for each protein complex. To run predictions on cloud GPU:

```bash
# Install Boltz (requires >=48GB VRAM — use RunPod A100 or Modal)
pip install boltz

# Predict actin-myosin-troponin complex
boltz predict data/boltz_inputs/actin_myosin_troponin.yaml --out data/structures/

# Predict titin Z-disk domain (fragment 1 of N)
boltz predict data/boltz_inputs/titin_Z_disk_Ig.yaml --out data/structures/
```

Predicted structures (`.cif` files) drop into `data/structures/` and are automatically picked up by the repair simulator's pLDDT scoring.

---

## Proteins modeled

| Protein | UniProt | Role in repair |
|---------|---------|----------------|
| Titin (TTN) | Q8WZ42 | Elastic scaffold; Z-disk and M-line anchor |
| Myosin-2 heavy chain (MYH2) | Q9UKX2 | Thick filament motor |
| Beta-actin (ACTB) | P60709 | Thin filament polymerization unit |
| Troponin I fast (TNNI2) | P48788 | Calcium-regulated thin filament switch |
| Troponin C fast (TNNC2) | P02585 | Ca²⁺ sensor, troponin complex anchor |

---

## Therapy parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `leucine_dose_g` | 0–10 g | mTOR activation → faster ribosomal synthesis |
| `sleep_quality` | 0–1 | GH secretion proxy → folding rate modifier |
| `body_temperature_c` | 36–38.5 °C | Arrhenius-based folding kinetics |
| `rest_hours` | 24–96 h | Recovery window before next stimulus |

---

## Tech stack

- **Python 3.11+**
- `biopython` — sequence and structure parsing
- `networkx` — assembly graph
- `scikit-optimize` — Bayesian optimization (future branch)
- `py3Dmol` — 3D structure visualization
- `plotly` / `matplotlib` — charts
- `boltz` *(optional, GPU)* — structure prediction

---

## Roadmap

- [ ] `feature/therapy-optimizer` — Bayesian search over therapy parameter space
- [ ] `feature/structure-analysis` — Parse AlphaFold DB pLDDT scores into the graph
- [ ] `feature/data-ingestion` — Auto-fetch sequences from UniProt and structures from RCSB PDB
- [ ] GPU prediction integration once Boltz-2 configs are validated
