# GainsFold — Architecture

## Overview

GainsFold is a three-layer pipeline. Each layer has a single responsibility and can be exercised independently.

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Data                                          │
│  configs/proteins.yaml                                  │
│  · Protein metadata (UniProt IDs, domain boundaries)    │
│  · Literature Kd values (4 interactions, 3 sources)     │
│  · Assembly order (4 steps)                             │
│  · Therapy parameter space + biological formulas        │
└────────────────────────┬────────────────────────────────┘
                         │ yaml.safe_load()
┌────────────────────────▼────────────────────────────────┐
│  Layer 2: Graph  (src/pathway/assembly_graph.py)        │
│  · NetworkX DiGraph — nodes=domains, edges=binding      │
│  · Edge weights from Kd values (nM)                     │
│  · Node stability score = f(folding_rate, Kd edges)     │
│  · Bottleneck detection (top-N lowest stability)        │
│                                                         │
│  Optional injection:                                    │
│  src/prediction/ ──► update_graph_from_predictions()   │
│  (replaces folding_rate with Boltz-2 pLDDT / 100)      │
└────────────────────────┬────────────────────────────────┘
                         │ assembly_step_nodes()
┌────────────────────────▼────────────────────────────────┐
│  Layer 3: Simulation  (src/pathway/repair_simulator.py) │
│  · Walks graph step-by-step (step 1 → 4)               │
│  · Per-step duration from stability + rate_modifier     │
│  · rate_modifier = leucine × sleep × temp (Arrhenius)  │
│  · Returns RepairReport (steps, total time, score)      │
└─────────────────────────────────────────────────────────┘
```

---

## Module reference

### `configs/proteins.yaml`

Single source of truth for all biological data. Structured as:

```
proteins:           # 5 proteins, 12 named domains
binding_affinities: # 4 Kd values (literature-sourced)
assembly_order:     # 4 steps defining graph node inclusion
therapy_space:      # parameter bounds + formula documentation
```

The graph, simulator, and Boltz runner all read this file. Nothing is hardcoded in Python.

### `src/pathway/assembly_graph.py`

**`build_assembly_graph(config)`**
1. Reads `assembly_order` to determine which domains become graph nodes (and their step index).
2. Reads `binding_affinities` to add directed Kd-weighted edges between protein partners.
3. Adds sequential dependency edges (step N → step N+1) for domains not already connected by a binding edge.

Nodes not in `assembly_order` (e.g., titin PEVK, N2B_unique) are defined in `proteins` but excluded from the graph — they are structural elements, not assembly participants.

**`node_stability_score(G, node_id)`**

```python
score = 0.6 * folding_rate + 0.4 * mean_affinity_score
affinity_score(Kd) = 1 / (1 + log10(Kd))
```

Range: [0, 1]. Uses geometric-mean affinity over all outgoing Kd edges. Nodes with no Kd edges (only step-dependency edges) reduce to their folding_rate alone.

### `src/prediction/boltz_runner.py`

**`BoltzRunner(use_mock=True)`**

Two execution modes with identical output (`PredictionResult`):

| Mode | Trigger | Behavior |
|---|---|---|
| Mock (CPU) | `use_mock=True` | Returns `cache/predictions.json`; no network, no GPU |
| Real (GPU) | `use_mock=False` + `boltz` installed | Fetches sequence from UniProt, writes Boltz-2 YAML, runs `boltz predict`, parses `confidence_model_0.json` |

**Cloud GPU swap point:** `BoltzRunner._run_boltz_predict()` is the single method to replace for Modal or RunPod. The rest of the pipeline (sequence fetch, YAML write, confidence parse, cache write) remains identical.

**`predict_all(proteins_config)`** — iterates every domain in `proteins.yaml` (including non-graph domains like PEVK) and returns a dict keyed by node_id. The confidence_mapper then filters to graph nodes only.

### `src/prediction/confidence_mapper.py`

**`update_graph_from_predictions(G, predictions)`**

For each graph node present in `predictions`, sets:
- `folding_rate = pLDDT / 100` (replaces prior value entirely)
- `boltz_plddt = raw pLDDT mean`
- `boltz_is_mock = bool`

Nodes with no prediction (impossible in normal use, since `predict_all` covers all domains) are left unchanged.

### `src/pathway/repair_simulator.py`

**`simulate_repair(therapy_params, use_boltz)`**

```
1. Load config
2. Build assembly graph
3. [if use_boltz] Inject Boltz-2 pLDDT into graph
4. Compute rate_modifier from therapy_params
5. For each assembly step:
   a. Compute stability score for each node
   b. Average stability across the step
   c. duration = base_duration(mean_stability) / rate_modifier
6. Return RepairReport
```

**Rate modifier formulas:**

```
leucine_modifier = 1.0 + 0.15 × min(leucine_g, 5.0)
sleep_modifier   = 0.7 + 0.6 × sleep_quality
temp_modifier    = exp((50 kJ/mol / R) × (1/310.15 − 1/T_K))
rate_modifier    = product of all active modifiers
```

**Duration function:**

```
base = 6h + 42h × (1 − mean_stability)^1.5
duration = base / rate_modifier
```

At stability=1.0: 6h minimum. At stability=0.0: ~48h maximum. Rate modifier compresses this range.

---

## Data flow: end-to-end example

```
simulate_repair(therapy_params={"leucine_dose_g": 5.0}, use_boltz=True)

  → load configs/proteins.yaml
  → build_assembly_graph()          # 8 nodes, ~30 edges
  → BoltzRunner(use_mock=True)
      .predict_all()                # 12 PredictionResults from cache JSON
  → update_graph_from_predictions() # folding_rate patched on 8 graph nodes
  → _therapy_rate_modifier()        # leucine_modifier=1.75, others=1.0
  → per step:
      step 1: stability=0.77, duration=11.2h / 1.75 = 6.4h
      step 2: stability=0.80, duration=9.8h  / 1.75 = 5.6h
      step 3: stability=0.80, duration=9.8h  / 1.75 = 5.6h
      step 4: stability=0.73, duration=12.1h / 1.75 = 6.9h
  → RepairReport(total=24.5h, score=0.92, bottlenecks=[...])
```

---

## Kd data sources

| Interaction | Kd (nM) | Reference |
|---|---|---|
| Actin–Myosin rigor (no ATP) | 10 | Geeves & Holmes (2005). *Annu. Rev. Biochem.* 74, 247–306 |
| Actin–Myosin weak (ATP) | 5,000 | Geeves & Holmes (2005) |
| Troponin I–Troponin C (Ca²⁺-sat.) | 2 | Li et al. (2004). *J. Biol. Chem.* 279, 39905–39914 |
| Titin–Myosin rod (A-band) | 100 | Labeit & Kolmerer (1995). *Science* 270, 293–296 |

---

## Test coverage

```
tests/test_assembly_graph.py   — 27 tests
  · Graph construction and node/edge attributes
  · Stability scoring bounds and monotonicity
  · Bottleneck node ordering
  · Simulator end-to-end correctness
  · Therapy modifier direction (optimised < baseline)

tests/test_prediction.py       — 24 tests
  · BoltzRunner mock mode (cache hits, KeyError on unknown domain)
  · pLDDT range and folding_rate derivation
  · Disordered domain low-confidence assertion (PEVK < 50, N2B < 50)
  · Confidence mapper graph patching and attribute injection
  · Simulator integration: use_boltz changes repair time, therapy still works
```

51 tests, 0 failures.

---

## Extension points

| What to extend | Where |
|---|---|
| Add a new protein/domain | `configs/proteins.yaml` → `proteins` + `assembly_order` + `binding_affinities` |
| Add a new therapy parameter | `configs/proteins.yaml` → `therapy_space`; implement modifier in `_therapy_rate_modifier()` |
| Swap in a different structure predictor | Subclass `BoltzRunner`, override `_run_boltz_predict()` and `_parse_confidence()` |
| Run on cloud GPU | Replace `BoltzRunner._run_boltz_predict()` body with a Modal/RunPod call |
| Add REST API | Wrap `simulate_repair()` in a FastAPI route; `RepairReport` is already dataclass-serializable |
| Bayesian optimization | `src/optimization/` stub — use `scikit-optimize` to minimize `total_repair_time_h` over `therapy_space` bounds |
