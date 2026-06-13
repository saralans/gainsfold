"""
GainsFold — Hugging Face Spaces entry point.

Run locally:  python app.py
Deploy:       push to a Gradio HF Space (free tier, CPU)
"""

import matplotlib
matplotlib.use("Agg")  # server-side rendering, no display needed

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

from src.pathway.repair_simulator import simulate_repair
from src.pathway.assembly_graph import build_assembly_graph, node_stability_score


# ── Core simulation ────────────────────────────────────────────────────────────

def run_simulation(leucine_dose: float, sleep_quality: float, temperature: float, use_boltz: bool):
    therapy = {
        "leucine_dose_g": leucine_dose,
        "sleep_quality": sleep_quality,
        "body_temperature_c": temperature,
    }

    baseline = simulate_repair(use_boltz=use_boltz)
    optimised = simulate_repair(therapy_params=therapy, use_boltz=use_boltz)

    time_saved = baseline.total_repair_time_h - optimised.total_repair_time_h
    pct_saved = (time_saved / baseline.total_repair_time_h) * 100

    # ── Summary card ──
    boltz_note = " *(Boltz-2 pLDDT rates)*" if use_boltz else " *(hand-tuned rates)*"
    summary = (
        f"### Repair estimate{boltz_note}\n\n"
        f"| | Baseline | Optimised |\n"
        f"|---|---|---|\n"
        f"| **Total time** | {baseline.total_repair_time_h:.1f} h | **{optimised.total_repair_time_h:.1f} h** |\n"
        f"| **Repair score** | {baseline.total_repair_score:.3f} | **{optimised.total_repair_score:.3f}** |\n"
        f"| **Time saved** | — | **{time_saved:.1f} h ({pct_saved:.0f}%)** |"
    )

    # ── Per-step table ──
    step_rows = []
    for b_step, o_step in zip(baseline.steps, optimised.steps):
        step_rows.append({
            "Assembly step": b_step.step_name.replace("_", " ").title(),
            "Baseline (h)": round(b_step.estimated_duration_h, 1),
            "Optimised (h)": round(o_step.estimated_duration_h, 1),
            "Saved (h)": round(b_step.estimated_duration_h - o_step.estimated_duration_h, 1),
            "Mean stability": round(o_step.mean_stability, 3),
        })
    step_df = pd.DataFrame(step_rows)

    # ── Bottleneck table ──
    bottleneck_rows = [
        {
            "Domain": node.replace("__", " › ").replace("_", " "),
            "Stability score": round(score, 3),
            "Risk": "High" if score < 0.60 else "Medium" if score < 0.75 else "Low",
        }
        for node, score in optimised.bottlenecks
    ]
    bottleneck_df = pd.DataFrame(bottleneck_rows)

    # ── Timeline bar chart ──
    fig_timeline, ax = plt.subplots(figsize=(10, 4))
    step_labels = [s.step_name.replace("_", "\n") for s in baseline.steps]
    x = range(len(step_labels))
    w = 0.35
    ax.bar(
        [i - w / 2 for i in x],
        [s.estimated_duration_h for s in baseline.steps],
        w, label="Baseline", color="#C44E52", alpha=0.85,
    )
    ax.bar(
        [i + w / 2 for i in x],
        [s.estimated_duration_h for s in optimised.steps],
        w, label="Optimised", color="#55A868", alpha=0.85,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(step_labels, fontsize=9)
    ax.set_ylabel("Duration (h)")
    ax.set_title("Sarcomere Repair Timeline — Baseline vs Optimised", fontsize=12)
    ax.legend()
    plt.tight_layout()

    # ── Stability heatmap ──
    G = build_assembly_graph()
    node_ids = list(G.nodes)
    scores = [node_stability_score(G, n) for n in node_ids]
    labels = [n.replace("__", "\n").replace("_", " ") for n in node_ids]
    colours = ["#C44E52" if s < 0.60 else "#F5A623" if s < 0.75 else "#55A868" for s in scores]

    fig_stability, ax2 = plt.subplots(figsize=(10, 4))
    bars = ax2.barh(labels, scores, color=colours)
    ax2.axvline(0.60, color="#C44E52", linestyle="--", linewidth=1.2, label="High risk (<0.60)")
    ax2.axvline(0.75, color="#F5A623", linestyle="--", linewidth=1.2, label="Medium risk (<0.75)")
    ax2.set_xlabel("Stability score")
    ax2.set_title("Domain Stability Scores (Boltz-2 pLDDT)" if use_boltz else "Domain Stability Scores (hand-tuned)", fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=8)
    plt.tight_layout()

    return summary, step_df, bottleneck_df, fig_timeline, fig_stability


# ── Leucine sensitivity sweep ──────────────────────────────────────────────────

def leucine_sweep(sleep_quality: float, temperature: float, use_boltz: bool):
    doses = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    times = []
    for dose in doses:
        r = simulate_repair(
            therapy_params={"leucine_dose_g": dose, "sleep_quality": sleep_quality, "body_temperature_c": temperature},
            use_boltz=use_boltz,
        )
        times.append(r.total_repair_time_h)

    best_dose = doses[times.index(min(times))]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(doses, times, marker="o", color="#4C72B0", linewidth=2)
    ax.axvline(best_dose, color="#55A868", linestyle="--", linewidth=1.5, label=f"Optimal: {best_dose}g")
    ax.set_xlabel("Leucine dose (g)")
    ax.set_ylabel("Total repair time (h)")
    ax.set_title("mTOR Sensitivity — Leucine Dose vs Repair Time", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    note = f"Repair time saturates at {best_dose}g leucine → **{min(times):.1f}h** (mTOR activation plateau at 5g)"
    return fig, note


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="GainsFold") as demo:

    gr.Markdown(
        """
        # 💪 GainsFold
        **Sarcomere repair simulator** — models how leucine (mTOR), sleep (GH/IGF-1), and
        temperature (Arrhenius kinetics) interact to accelerate muscle protein assembly after
        resistance training microtears. Uses a NetworkX DiGraph weighted by published Kd binding
        affinities (Geeves & Holmes 2005; Li et al. 2004; Labeit & Kolmerer 1995).
        """
    )

    with gr.Tab("Repair Simulator"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Therapy parameters")
                leucine = gr.Slider(0, 10, value=0, step=0.5, label="Leucine dose (g)", info="mTOR activation — saturates at 5g")
                sleep = gr.Slider(0, 1, value=0.5, step=0.05, label="Sleep quality (0–1)", info="GH secretion proxy (0=poor, 1=optimal)")
                temp = gr.Slider(36.0, 38.5, value=37.0, step=0.1, label="Body temperature (°C)", info="Arrhenius folding kinetics")
                use_boltz = gr.Checkbox(value=True, label="Use Boltz-2 pLDDT rates", info="Replaces hand-tuned folding rates with structure prediction confidence scores")
                run_btn = gr.Button("Run simulation", variant="primary")

            with gr.Column(scale=2):
                summary_out = gr.Markdown()
                with gr.Row():
                    timeline_plot = gr.Plot(label="Repair timeline")
                    stability_plot = gr.Plot(label="Domain stability")

        gr.Markdown("### Per-step breakdown")
        step_table = gr.Dataframe(interactive=False)

        gr.Markdown("### Bottleneck domains")
        bottleneck_table = gr.Dataframe(interactive=False)

        run_btn.click(
            fn=run_simulation,
            inputs=[leucine, sleep, temp, use_boltz],
            outputs=[summary_out, step_table, bottleneck_table, timeline_plot, stability_plot],
        )

        # Run on load with defaults
        demo.load(
            fn=run_simulation,
            inputs=[leucine, sleep, temp, use_boltz],
            outputs=[summary_out, step_table, bottleneck_table, timeline_plot, stability_plot],
        )

    with gr.Tab("Leucine Sensitivity"):
        gr.Markdown(
            "Sweep leucine dose from 0–10g while holding other parameters fixed. "
            "Shows where mTOR activation saturates and returns diminish."
        )
        with gr.Row():
            with gr.Column(scale=1):
                sweep_sleep = gr.Slider(0, 1, value=0.7, step=0.05, label="Sleep quality")
                sweep_temp = gr.Slider(36.0, 38.5, value=37.0, step=0.1, label="Body temperature (°C)")
                sweep_boltz = gr.Checkbox(value=True, label="Use Boltz-2 pLDDT rates")
                sweep_btn = gr.Button("Run sweep", variant="primary")
            with gr.Column(scale=2):
                sweep_plot = gr.Plot()
                sweep_note = gr.Markdown()

        sweep_btn.click(
            fn=leucine_sweep,
            inputs=[sweep_sleep, sweep_temp, sweep_boltz],
            outputs=[sweep_plot, sweep_note],
        )

    with gr.Tab("About"):
        gr.Markdown(
            """
            ## How it works

            **Assembly graph** — the sarcomere reconstruction sequence is encoded as a directed graph
            (NetworkX DiGraph). Each node is a protein domain; each edge is a binding event weighted
            by its dissociation constant (Kd in nM) from the literature.

            **Stability score** per domain:
            ```
            stability = 0.6 × folding_rate + 0.4 × mean_affinity_score
            affinity_score(Kd) = 1 / (1 + log₁₀(Kd))
            ```

            **Repair duration** per assembly step:
            ```
            base_duration = 6h + 42h × (1 − mean_stability)^1.5
            duration = base_duration / rate_modifier
            ```

            **Rate modifier** — product of three independent factors:

            | Parameter | Mechanism | Formula |
            |---|---|---|
            | Leucine | mTOR → ribosomal synthesis | `1.0 + 0.15 × min(dose_g, 5.0)` |
            | Sleep quality | GH → IGF-1 / mTOR axis | `0.7 + 0.6 × quality` |
            | Temperature | Arrhenius folding kinetics | `exp((50 kJ/mol / R) × (1/T_ref − 1/T))` |

            **Boltz-2 integration** — when enabled, domain folding rates are replaced with
            `pLDDT / 100` from Boltz-2 structure predictions (MIT, 2024). CPU mode uses a
            pre-computed cache; GPU mode fetches sequences from UniProt and runs `boltz predict`.

            ## Proteins modeled
            | Protein | UniProt | Role |
            |---|---|---|
            | Titin (TTN) | Q8WZ42 | Elastic scaffold; Z-disk and M-line anchor |
            | Myosin-2 heavy chain (MYH2) | Q9UKX2 | Thick filament motor |
            | Beta-actin (ACTB) | P60709 | Thin filament polymerization |
            | Troponin I fast (TNNI2) | P48788 | Ca²⁺-regulated ATPase inhibitor |
            | Troponin C fast (TNNC2) | P02585 | Ca²⁺ sensor |

            ## References
            - Geeves & Holmes (2005). *Annu. Rev. Biochem.* 74, 247–306
            - Li et al. (2004). *J. Biol. Chem.* 279, 39905–39914
            - Labeit & Kolmerer (1995). *Science* 270, 293–296
            - Boltz-2: Wohlwend et al. (2024). MIT License. github.com/jwohlwend/boltz
            """
        )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
