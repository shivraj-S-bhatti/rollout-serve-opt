"""CLI entrypoint for trace generation, NNLS calibration, topology sweeps, and guardrails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from inference.calibration import fit_latency_coeffs, read_observations_jsonl
from inference.guardrails import evaluate_guardrails, read_eval_samples_jsonl
from inference.pareto import format_sweep_markdown, pareto_frontier, run_topology_sweep
from inference.schema import HardwareProfile, TopologyConfig
from inference.workloads import (
    ALL_WORKLOAD_TYPES,
    generate_workload,
    read_trace_jsonl,
    write_trace_jsonl,
)


def _load_hardware(path: str | Path) -> HardwareProfile:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return HardwareProfile.model_validate(raw)


def _load_topologies(target: str | Path) -> list[TopologyConfig]:
    p = Path(target)
    files = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml")) if p.is_dir() else [p]
    topos: list[TopologyConfig] = []
    for f in files:
        raw = yaml.safe_load(f.read_text(encoding="utf-8"))
        topos.append(TopologyConfig.model_validate(raw))
    if not topos:
        raise FileNotFoundError(f"No topology YAML files found at {target}")
    return topos


def _cmd_generate_traces(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        list(ALL_WORKLOAD_TYPES)
        if args.workload == "all"
        else [args.workload]
    )
    for idx, wname in enumerate(targets):
        reqs = generate_workload(wname, count=args.count, seed=args.seed + idx * 100)
        dest = out_dir / f"{wname}.jsonl"
        write_trace_jsonl(dest, reqs)
        print(f"wrote {len(reqs)} requests -> {dest}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    obs = read_observations_jsonl(args.observations)
    coeffs, stats = fit_latency_coeffs(obs, source_label=args.source_label)
    hw_out = Path(args.hardware_out)
    if args.hardware_template:
        hw = _load_hardware(args.hardware_template)
        hw = hw.model_copy(update={"coeffs": coeffs})
        payload = hw.model_dump()
    else:
        payload = coeffs.model_dump()

    hw_out.parent.mkdir(parents=True, exist_ok=True)
    hw_out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"calibrated coeffs saved -> {hw_out}")
    print(json.dumps(stats, indent=2))
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    trace = read_trace_jsonl(args.trace)
    hw = _load_hardware(args.hardware)
    topos = _load_topologies(args.topologies)
    metrics_list = run_topology_sweep(trace=trace, hardware=hw, topologies=topos)
    frontier = pareto_frontier(metrics_list)

    print(format_sweep_markdown(metrics_list, frontier))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hardware": hw.name,
            "calibration_source": hw.coeffs.calibration_source,
            "pareto_frontier": frontier,
            "results": [m.model_dump() for m in metrics_list],
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def _cmd_guardrail(args: argparse.Namespace) -> int:
    base = read_eval_samples_jsonl(args.baseline)
    cand = read_eval_samples_jsonl(args.candidate)
    report = evaluate_guardrails(
        baseline_samples=base,
        candidate_samples=cand,
        max_pass_rate_drop=args.max_pass_rate_drop,
        max_mean_kl=args.max_mean_kl,
        min_determinism_rate=args.min_determinism_rate,
    )
    output = json.dumps(report.model_dump(), indent=2)
    print(output)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
    return 0 if report.passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference-workbench",
        description="Calibrated workload-shaped LLM inference topology workbench.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate-traces", help="Generate privacy-safe JSONL workload traces.")
    p_gen.add_argument(
        "--workload",
        choices=["all", *ALL_WORKLOAD_TYPES],
        default="all",
    )
    p_gen.add_argument("--count", type=int, default=120)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--out", required=True)
    p_gen.set_defaults(func=_cmd_generate_traces)

    p_cal = sub.add_parser("calibrate", help="Fit step-latency NNLS coefficients from observations.")
    p_cal.add_argument("--observations", required=True)
    p_cal.add_argument("--hardware-template", default=None)
    p_cal.add_argument("--hardware-out", required=True)
    p_cal.add_argument("--source-label", default="nnls_trace_calibrated")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_sweep = sub.add_parser("sweep", help="Sweep candidate topologies and compute Pareto frontier.")
    p_sweep.add_argument("--trace", required=True)
    p_sweep.add_argument("--hardware", required=True)
    p_sweep.add_argument("--topologies", required=True)
    p_sweep.add_argument("--out", default=None)
    p_sweep.set_defaults(func=_cmd_sweep)

    p_guard = sub.add_parser("guardrail", help="Run Zero-Score completeness, KL, and determinism gates.")
    p_guard.add_argument("--baseline", required=True)
    p_guard.add_argument("--candidate", required=True)
    p_guard.add_argument("--max-pass-rate-drop", type=float, default=0.01)
    p_guard.add_argument("--max-mean-kl", type=float, default=0.02)
    p_guard.add_argument("--min-determinism-rate", type=float, default=0.99)
    p_guard.add_argument("--out", default=None)
    p_guard.set_defaults(func=_cmd_guardrail)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
