# Changelog

## [Unreleased] — feature/protein-repair

### Added
- `configs/proteins.yaml` — protein metadata, titin domain boundaries, binding affinities (Kd), assembly order, and therapy parameter space
- `src/pathway/assembly_graph.py` — NetworkX DiGraph encoding the 4-step sarcomere assembly sequence with stability scoring and bottleneck detection
- `src/pathway/repair_simulator.py` — repair timeline simulator with Arrhenius temperature correction and mTOR/GH therapy modifiers
- `tests/test_assembly_graph.py` — 27 unit tests covering graph construction, stability scoring, simulation correctness, and therapy effect
- `notebooks/03_repair_pathway.ipynb` — end-to-end walkthrough: assembly graph visualization, stability bar chart, baseline vs. optimised repair timeline, leucine sensitivity sweep
- `requirements.txt` — pinned dependencies; Boltz listed as optional (GPU-only)
- `README.md` — project overview, quick start, GPU prediction instructions, protein table, therapy parameter docs
