"""Agent 수 증가 시 direct debate와 summarization-memory를 비교한다.

같은 문항, model, seed, round에서 majority(k=N), debate(N agents),
debate_memory(N agents + shared summary)를 순차 실행한다. 중단 후 같은 명령을
다시 실행하면 run_phase1.py의 문항 단위 저장을 이용해 이어서 진행한다.
"""
import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "phase1"
VALID_TASKS = {"gsm8k", "mmlu"}
VALID_METHODS = {"majority", "debate", "debate_memory"}


def csv_list(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def int_list(value):
    return [int(part) for part in csv_list(value)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase1.yaml")
    parser.add_argument("--tasks", default="gsm8k,mmlu")
    parser.add_argument("--model", default="qwen", choices=["qwen", "exaone"])
    parser.add_argument("--methods", default="majority,debate,debate_memory")
    parser.add_argument("--agent-counts", default="3,5,7")
    parser.add_argument("--n-rounds", type=int, default=2)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--run-seed", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--memory-max-new-tokens", type=int, default=256)
    parser.add_argument("--memory-temperature", type=float, default=0.0)
    parser.add_argument("--tag-prefix", default="agent_scaling_memory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def validate(args):
    args.tasks = csv_list(args.tasks)
    args.methods = csv_list(args.methods)
    args.agent_counts = int_list(args.agent_counts)
    unknown_tasks = sorted(set(args.tasks) - VALID_TASKS)
    unknown_methods = sorted(set(args.methods) - VALID_METHODS)
    if unknown_tasks:
        raise ValueError(f"unknown tasks: {', '.join(unknown_tasks)}")
    if unknown_methods:
        raise ValueError(f"unknown methods: {', '.join(unknown_methods)}")
    if not args.agent_counts or min(args.agent_counts) < 2:
        raise ValueError("--agent-counts must contain integers >= 2")
    if args.n < 1 or args.n_rounds < 1:
        raise ValueError("--n and --n-rounds must be positive")


def tag_for(args, n_agents):
    return (
        f"{args.tag_prefix}_a{n_agents:02d}_"
        f"rounds{args.n_rounds}_s{args.run_seed}"
    )


def result_path(args, task, n_agents):
    return RESULTS_DIR / (
        f"{task}_{args.model}_n{args.n}_{tag_for(args, n_agents)}.json"
    )


def build_command(args, task, n_agents):
    return [
        sys.executable,
        "scripts/run_phase1.py",
        "--config",
        args.config,
        "--task",
        task,
        "--model",
        args.model,
        "--methods",
        ",".join(args.methods),
        "--n",
        str(args.n),
        "--data-seed",
        str(args.data_seed),
        "--run-seed",
        str(args.run_seed),
        "--temperature",
        str(args.temperature),
        "--n-agents",
        str(n_agents),
        "--n-rounds",
        str(args.n_rounds),
        "--k",
        str(n_agents),
        "--memory-max-new-tokens",
        str(args.memory_max_new_tokens),
        "--memory-temperature",
        str(args.memory_temperature),
        "--tag",
        tag_for(args, n_agents),
    ]


def run_experiments(args):
    failures = []
    total = len(args.agent_counts) * len(args.tasks)
    current = 0
    for n_agents in args.agent_counts:
        for task in args.tasks:
            current += 1
            command = build_command(args, task, n_agents)
            print(
                f"\n[agent scaling {current}/{total}] {shlex.join(command)}",
                flush=True,
            )
            if args.dry_run:
                continue
            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            if completed.returncode:
                failures.append(
                    {
                        "task": task,
                        "n_agents": n_agents,
                        "returncode": completed.returncode,
                    }
                )
                if not args.keep_going:
                    raise SystemExit(completed.returncode)
    return failures


def mean_or_none(values):
    known = [value for value in values if value is not None]
    return statistics.fmean(known) if known else None


def trace_metrics(items):
    peak_direct_tokens = []
    peak_delivered_tokens = []
    per_agent_ratios = []
    total_input_ratios = []
    calls_per_item = []
    communication_modes = []

    for item in items.values():
        trace = item.get("debate_trace")
        if not trace:
            continue
        communication_modes.append(trace.get("communication_mode"))
        calls_per_item.append(trace.get("call_counts", {}).get("total_generation_calls"))
        for stats in trace.get("communication_stats_by_round", []):
            if stats.get("round_index") == 0:
                continue
            direct = [
                value
                for value in stats.get("direct_equivalent_tokens_per_agent", [])
                if value is not None
            ]
            delivered = [
                value
                for value in stats.get("delivered_tokens_per_agent", [])
                if value is not None
            ]
            if direct:
                peak_direct_tokens.append(max(direct))
            if delivered:
                peak_delivered_tokens.append(max(delivered))
            per_agent_ratios.append(stats.get("per_agent_context_token_ratio"))
            total_input_ratios.append(stats.get("total_input_token_ratio"))

    modes = sorted({mode for mode in communication_modes if mode})
    return {
        "communication_mode": ",".join(modes) if modes else None,
        "mean_peak_direct_equivalent_tokens": mean_or_none(peak_direct_tokens),
        "mean_peak_delivered_tokens": mean_or_none(peak_delivered_tokens),
        "mean_per_agent_context_token_ratio": mean_or_none(per_agent_ratios),
        "mean_total_input_token_ratio": mean_or_none(total_input_ratios),
        "mean_generation_calls_per_item": mean_or_none(calls_per_item),
    }


def summarize(args):
    rows = []
    details = []
    for n_agents in args.agent_counts:
        for task in args.tasks:
            path = result_path(args, task, n_agents)
            if path.exists():
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
            else:
                payload = {}
            for method in args.methods:
                items = payload.get(method, {})
                n_completed = len(items)
                n_correct = sum(bool(item.get("correct")) for item in items.values())
                accuracy = n_correct / n_completed if n_completed else None
                metrics = trace_metrics(items)
                row = {
                    "task": task,
                    "model": args.model,
                    "n_agents": n_agents,
                    "n_rounds": args.n_rounds,
                    "method": method,
                    "completed_items": n_completed,
                    "expected_items": args.n,
                    "accuracy": accuracy,
                    **metrics,
                }
                rows.append(row)
                details.append(
                    {
                        **row,
                        "result_path": str(path.relative_to(REPO_ROOT)),
                    }
                )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    task_scope = "-".join(args.tasks)
    stem = (
        f"agent_scaling_summary_{task_scope}_{args.model}_n{args.n}_"
        f"rounds{args.n_rounds}_{args.tag_prefix}"
    )
    csv_path = RESULTS_DIR / f"{stem}.csv"
    json_path = RESULTS_DIR / f"{stem}.json"
    md_path = RESULTS_DIR / f"{stem}.md"

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "settings": {
                    "tasks": args.tasks,
                    "model": args.model,
                    "methods": args.methods,
                    "agent_counts": args.agent_counts,
                    "n_rounds": args.n_rounds,
                    "n": args.n,
                    "data_seed": args.data_seed,
                    "run_seed": args.run_seed,
                    "temperature": args.temperature,
                    "memory_max_new_tokens": args.memory_max_new_tokens,
                    "memory_temperature": args.memory_temperature,
                },
                "summary": details,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    lines = [
        "# Agent scaling with summarization-memory",
        "",
        "| Task | Agents | Method | Items | Accuracy | Peak context tokens | Calls/item |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        accuracy = (
            f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "N/A"
        )
        peak = row["mean_peak_delivered_tokens"]
        peak_text = f"{peak:.1f}" if peak is not None else "N/A"
        calls = row["mean_generation_calls_per_item"]
        calls_text = f"{calls:.1f}" if calls is not None else "N/A"
        lines.append(
            f"| {row['task']} | {row['n_agents']} | {row['method']} | "
            f"{row['completed_items']}/{row['expected_items']} | {accuracy} | "
            f"{peak_text} | {calls_text} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n===== AGENT SCALING SUMMARY =====")
    for row in rows:
        accuracy = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "N/A"
        ratio = row["mean_per_agent_context_token_ratio"]
        ratio_text = f"{ratio:.3f}" if ratio is not None else "N/A"
        print(
            f"{row['task']:6s} agents={row['n_agents']} "
            f"{row['method']:14s} items={row['completed_items']}/{row['expected_items']} "
            f"acc={accuracy} context_ratio={ratio_text}"
        )
    print(f"[saved] {csv_path.relative_to(REPO_ROOT)}")
    print(f"[saved] {json_path.relative_to(REPO_ROOT)}")
    print(f"[saved] {md_path.relative_to(REPO_ROOT)}")
    return rows


def main():
    args = parse_args()
    validate(args)
    failures = []
    if not args.summarize_only:
        failures = run_experiments(args)
    if not args.dry_run:
        summarize(args)
    if failures:
        print("\n[warning] failed combinations:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
