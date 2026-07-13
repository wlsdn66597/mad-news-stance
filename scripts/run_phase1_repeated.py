"""Phase 1 반복실험 실행 및 mean ± stderr 집계.

데이터 서브셋은 data_seed로 고정하고, LLM 생성 seed만 반복마다 바꾼다.
각 반복은 run_phase1.py의 별도 결과 파일을 사용하므로 중단 후 같은 명령을
다시 실행하면 완료 문항은 건너뛰고 이어서 진행한다.

예:
  # 빠른 동작 확인
  python scripts/run_phase1_repeated.py --tasks gsm8k --models qwen --n 5 --repeats 2

  # 본실험
  python scripts/run_phase1_repeated.py \
      --tasks gsm8k,mmlu --models qwen,exaone --n 100 --repeats 5
"""
import argparse
import csv
import json
import math
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "phase1"
VALID_TASKS = {"gsm8k", "mmlu"}
VALID_MODELS = {"qwen", "exaone"}
VALID_METHODS = {"vanilla", "cot", "majority", "debate"}


def csv_list(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/phase1.yaml")
    ap.add_argument("--tasks", default="gsm8k,mmlu")
    ap.add_argument("--models", default="qwen,exaone")
    ap.add_argument("--methods", default="vanilla,cot,majority,debate")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--base-run-seed", type=int, default=1000)
    ap.add_argument("--tag-prefix", default="repeat")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--n-rounds", type=int, default=None)
    ap.add_argument("--n-agents", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--keep-going", action="store_true",
        help="한 조합이 실패해도 다음 반복을 계속 실행한다.",
    )
    return ap.parse_args()


def validate(args):
    args.tasks = csv_list(args.tasks)
    args.models = csv_list(args.models)
    args.methods = csv_list(args.methods)
    checks = [
        ("tasks", args.tasks, VALID_TASKS),
        ("models", args.models, VALID_MODELS),
        ("methods", args.methods, VALID_METHODS),
    ]
    for name, values, valid in checks:
        if not values:
            raise ValueError(f"--{name} must contain at least one value")
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError(f"unknown {name}: {', '.join(unknown)}")
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.repeats < 2:
        raise ValueError("stderr 계산을 위해 --repeats는 2 이상이어야 합니다")


def tag_for(args, repeat_index, run_seed):
    return f"{args.tag_prefix}_r{repeat_index + 1:02d}_s{run_seed}"


def result_path(args, task, model, repeat_index, run_seed):
    tag = tag_for(args, repeat_index, run_seed)
    return RESULTS_DIR / f"{task}_{model}_n{args.n}_{tag}.json"


def build_command(args, task, model, repeat_index):
    run_seed = args.base_run_seed + repeat_index
    command = [
        sys.executable,
        "scripts/run_phase1.py",
        "--config", args.config,
        "--task", task,
        "--model", model,
        "--methods", ",".join(args.methods),
        "--n", str(args.n),
        "--data-seed", str(args.data_seed),
        "--run-seed", str(run_seed),
        "--tag", tag_for(args, repeat_index, run_seed),
    ]
    optional = [
        ("--split", args.split),
        ("--temperature", args.temperature),
        ("--max-new-tokens", args.max_new_tokens),
        ("--n-rounds", args.n_rounds),
        ("--n-agents", args.n_agents),
        ("--k", args.k),
    ]
    for flag, value in optional:
        if value is not None:
            command.extend([flag, str(value)])
    return command


def run_experiments(args):
    failures = []
    total = len(args.tasks) * len(args.models) * args.repeats
    current = 0
    for model in args.models:
        for task in args.tasks:
            for repeat_index in range(args.repeats):
                current += 1
                command = build_command(args, task, model, repeat_index)
                print(f"\n[repeat {current}/{total}] {shlex.join(command)}", flush=True)
                if args.dry_run:
                    continue
                completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
                if completed.returncode != 0:
                    failures.append((task, model, repeat_index + 1, completed.returncode))
                    if not args.keep_going:
                        raise SystemExit(completed.returncode)
    return failures


def summarize(args):
    rows = []
    details = []
    for task in args.tasks:
        for model in args.models:
            for method in args.methods:
                accuracies = []
                run_details = []
                for repeat_index in range(args.repeats):
                    run_seed = args.base_run_seed + repeat_index
                    path = result_path(args, task, model, repeat_index, run_seed)
                    if not path.exists():
                        run_details.append({
                            "repeat": repeat_index + 1,
                            "run_seed": run_seed,
                            "status": "missing",
                            "path": str(path.relative_to(REPO_ROOT)),
                        })
                        continue

                    result = json.load(open(path, encoding="utf-8"))
                    method_result = result.get(method, {})
                    count = len(method_result)
                    if count != args.n:
                        run_details.append({
                            "repeat": repeat_index + 1,
                            "run_seed": run_seed,
                            "status": "incomplete",
                            "count": count,
                            "expected_count": args.n,
                            "path": str(path.relative_to(REPO_ROOT)),
                        })
                        continue

                    accuracy = sum(bool(item["correct"]) for item in method_result.values()) / count
                    accuracies.append(accuracy)
                    run_details.append({
                        "repeat": repeat_index + 1,
                        "run_seed": run_seed,
                        "status": "complete",
                        "count": count,
                        "accuracy": accuracy,
                        "path": str(path.relative_to(REPO_ROOT)),
                    })

                completed_runs = len(accuracies)
                mean = statistics.fmean(accuracies) if accuracies else None
                std = statistics.stdev(accuracies) if completed_runs >= 2 else None
                stderr = std / math.sqrt(completed_runs) if std is not None else None
                row = {
                    "task": task,
                    "model": model,
                    "method": method,
                    "completed_runs": completed_runs,
                    "expected_runs": args.repeats,
                    "n_per_run": args.n,
                    "mean_accuracy": mean,
                    "sample_std": std,
                    "stderr": stderr,
                    "mean_pm_stderr": (
                        f"{mean:.4f} ± {stderr:.4f}"
                        if mean is not None and stderr is not None else "N/A"
                    ),
                }
                rows.append(row)
                details.append({**row, "runs": run_details})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    task_scope = "-".join(args.tasks)
    model_scope = "-".join(args.models)
    stem = (
        f"repeat_summary_{task_scope}_{model_scope}_"
        f"n{args.n}_r{args.repeats}_{args.tag_prefix}"
    )
    csv_path = RESULTS_DIR / f"{stem}.csv"
    json_path = RESULTS_DIR / f"{stem}.json"
    md_path = RESULTS_DIR / f"{stem}.md"

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "settings": {
            "tasks": args.tasks,
            "models": args.models,
            "methods": args.methods,
            "n": args.n,
            "repeats": args.repeats,
            "data_seed": args.data_seed,
            "base_run_seed": args.base_run_seed,
            "tag_prefix": args.tag_prefix,
        },
        "summary": details,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = [
        "# Phase 1 repeated experiment summary",
        "",
        f"- n per run: {args.n}",
        f"- expected repeats: {args.repeats}",
        f"- fixed data seed: {args.data_seed}",
        f"- run seeds: {args.base_run_seed}..{args.base_run_seed + args.repeats - 1}",
        "- stderr: sample standard deviation / sqrt(completed runs)",
        "",
        "| Task | Model | Method | Runs | Accuracy (mean ± stderr) |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['model']} | {row['method']} | "
            f"{row['completed_runs']}/{row['expected_runs']} | {row['mean_pm_stderr']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n===== REPEATED EXPERIMENT SUMMARY =====")
    for row in rows:
        print(
            f"{row['task']:6s} {row['model']:7s} {row['method']:10s} "
            f"runs={row['completed_runs']}/{row['expected_runs']}  {row['mean_pm_stderr']}"
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
        print("\n[warning] failed runs:")
        for task, model, repeat, returncode in failures:
            print(f"  task={task} model={model} repeat={repeat} returncode={returncode}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
