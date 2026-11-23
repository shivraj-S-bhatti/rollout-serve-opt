"""Multi-topology Pareto frontier evaluation and Markdown/JSON reporting."""

from __future__ import annotations

from inference.schema import HardwareProfile, SimMetrics, TopologyConfig, TraceRequest
from inference.simulator import InferenceSimulator


def run_topology_sweep(
    trace: list[TraceRequest],
    hardware: HardwareProfile,
    topologies: list[TopologyConfig],
) -> list[SimMetrics]:
    """Run independent non-mutating simulations across candidate topologies."""
    results: list[SimMetrics] = []
    for topo in topologies:
        sim = InferenceSimulator(hardware=hardware, topology=topo)
        trace_copy = [req.model_copy(deep=True) for req in trace]
        results.append(sim.run(trace_copy))
    return results


def pareto_frontier(metrics_list: list[SimMetrics]) -> list[str]:
    """Return topology names on the Pareto frontier (max goodput_tokens/GPU-hr, min p99 TTFT & ITL)."""
    frontier: list[str] = []
    for cand in metrics_list:
        dominated = False
        for other in metrics_list:
            if other.topology_name == cand.topology_name:
                continue
            better_or_equal = (
                other.goodput_tokens_per_gpu_hr >= cand.goodput_tokens_per_gpu_hr
                and other.p99_ttft_ms <= cand.p99_ttft_ms
                and other.p99_itl_ms <= cand.p99_itl_ms
            )
            strictly_better = (
                other.goodput_tokens_per_gpu_hr > cand.goodput_tokens_per_gpu_hr
                or other.p99_ttft_ms < cand.p99_ttft_ms
                or other.p99_itl_ms < cand.p99_itl_ms
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(cand.topology_name)
    return frontier


def format_sweep_markdown(metrics_list: list[SimMetrics], pareto_names: list[str]) -> str:
    """Render a crisp Markdown summary table of sweep metrics and Pareto status."""
    if not metrics_list:
        return "No simulation results."

    cal_sources = sorted({m.calibration_source for m in metrics_list})
    hw_name = metrics_list[0].hardware_name
    lines = [
        f"### Topology Sweep — Hardware: `{hw_name}` | Calibration: `{', '.join(cal_sources)}`",
        "",
        "| Topology | Pareto | SLA Pass | p99 TTFT (ms) | p99 / max ITL (ms) | Goodput (tok/GPU-hr) | L1 / L2 Hit | Avg KV Xfer (ms) | Peak HBM |",
        "| :--- | :---: | :---: | ---: | :---: | ---: | :---: | ---: | ---: |",
    ]
    for m in metrics_list:
        star = "YES" if m.topology_name in pareto_names else "no"
        sla = f"{m.sla_compliant_requests}/{m.total_requests}"
        itl_pair = f"{m.p99_itl_ms:.1f} / {m.max_itl_ms:.1f}"
        hit = f"{m.l1_prefix_hit_rate*100:.1f}% / {m.l2_prefix_hit_rate*100:.1f}%"
        lines.append(
            f"| `{m.topology_name}` | {star} | {sla} | {m.p99_ttft_ms:.1f} | {itl_pair} | "
            f"{m.goodput_tokens_per_gpu_hr:,.0f} | {hit} | {m.avg_kv_transfer_ms:.2f} | {m.peak_hbm_utilization*100:.1f}% |"
        )
    if any("uncalibrated" in c for c in cal_sources):
        lines.append("")
        lines.append(
            "> **Calibration Notice:** Coefficients marked `analytical_prior_uncalibrated` are modeling priors. "
            "Run `inference-workbench calibrate` on live GPU step traces before citing absolute production numbers."
        )
    return "\n".join(lines)
