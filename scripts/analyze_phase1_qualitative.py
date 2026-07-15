"""14B Phase 1 결과에서 paired outcome과 debate trace 분석 표본을 만든다."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_TAG = "qwen14b4_faithful_repeat_trace_r01_s1000"
DEFAULT_DIR = Path("results/phase1")
METHODS = ("vanilla", "cot", "majority", "debate")
TRANSITION_PRIORITY = {
    "corrected": 0,
    "degraded": 0,
    "persistent_wrong": 1,
    "stable_correct": 2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gsm8k-result",
        type=Path,
        default=DEFAULT_DIR / f"gsm8k_qwen_n50_{DEFAULT_TAG}.json",
    )
    parser.add_argument(
        "--mmlu-result",
        type=Path,
        default=DEFAULT_DIR / f"mmlu_qwen_n50_{DEFAULT_TAG}.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DIR / "qualitative_14b",
    )
    parser.add_argument("--gsm8k-sample", type=int, default=20)
    parser.add_argument("--mmlu-sample", type=int, default=30)
    parser.add_argument(
        "--case",
        help="한 문항의 method/trace를 JSON으로 출력한다. 예: mmlu:1234",
    )
    return parser.parse_args()


def parse_answer(task, text):
    if not text:
        return None
    if task == "mmlu":
        found = re.findall(r"\(([ABCD])\)", text)
        if found:
            return found[-1]
        found = re.findall(
            r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?\b", text, re.IGNORECASE
        )
        if found:
            return found[-1]
        found = re.findall(r"\b([ABCD])\b", text)
        return found[-1] if found else None

    found = re.findall(
        r"[Aa]nswer\s*[:：]?\s*\$?\s*(-?\d[\d,]*\.?\d*)", text
    )
    if found:
        return found[-1].replace(",", "").rstrip(".")
    found = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return found[-1].replace(",", "").rstrip(".") if found else None


def normalized_gold(task, gold):
    return "ABCD"[int(gold)] if task == "mmlu" else str(gold).replace(",", "")


def prediction_is_correct(task, prediction, gold):
    if prediction is None:
        return False
    expected = normalized_gold(task, gold)
    if task == "gsm8k":
        try:
            return abs(float(prediction) - float(expected)) < 1e-4
        except ValueError:
            pass
    return str(prediction).strip() == expected.strip()


def transition_name(majority_correct, debate_correct):
    if majority_correct and debate_correct:
        return "stable_correct"
    if majority_correct:
        return "degraded"
    if debate_correct:
        return "corrected"
    return "persistent_wrong"


def load_result(path):
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [method for method in METHODS if method not in data]
    if missing:
        raise ValueError(f"{path}: missing methods {missing}")
    expected = set(data["debate"])
    for method in METHODS:
        if set(data[method]) != expected:
            raise ValueError(f"{path}: {method} does not contain the same item IDs")
    return data


def analyze_task(task, path):
    data = load_result(path)
    rows = []
    for item_id in sorted(data["debate"], key=int):
        records = {method: data[method][item_id] for method in METHODS}
        debate = records["debate"]
        trace = debate.get("debate_trace", {})
        rounds = trace.get("answers_by_round", [])
        initial_answers = rounds[0] if rounds else []
        final_answers = rounds[-1] if rounds else []
        initial_preds = [parse_answer(task, answer) for answer in initial_answers]
        final_preds = [parse_answer(task, answer) for answer in final_answers]
        gold = debate["gold"]

        row = {
            "task": task,
            "item_id": item_id,
            "transition": transition_name(
                bool(records["majority"]["correct"]), bool(debate["correct"])
            ),
            "gold": normalized_gold(task, gold),
        }
        for method in METHODS:
            row[f"{method}_pred"] = records[method].get("pred")
            row[f"{method}_correct"] = bool(records[method]["correct"])
        row.update(
            {
                "round0_preds": "|".join(str(x or "") for x in initial_preds),
                "final_preds": "|".join(str(x or "") for x in final_preds),
                "initial_unique_answers": len(
                    {x for x in initial_preds if x is not None}
                ),
                "final_unique_answers": len({x for x in final_preds if x is not None}),
                "initial_correct_agents": sum(
                    prediction_is_correct(task, prediction, gold)
                    for prediction in initial_preds
                ),
                "final_correct_agents": sum(
                    prediction_is_correct(task, prediction, gold)
                    for prediction in final_preds
                ),
                "changed_agents": sum(
                    before != after
                    for before, after in zip(initial_preds, final_preds)
                ),
                "prompt": debate.get("prompt", ""),
            }
        )
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def select_cases(rows, limit):
    ordered = sorted(
        rows,
        key=lambda row: (
            TRANSITION_PRIORITY[row["transition"]],
            -row["initial_unique_answers"],
            -row["changed_agents"],
            int(row["item_id"]),
        ),
    )
    return ordered[: min(limit, len(ordered))]


def interpretation(task, counts):
    corrected = counts["corrected"]
    degraded = counts["degraded"]
    if task == "gsm8k" and corrected == degraded == 0:
        return "Majority와 Debate의 정오답 집합이 같아 높은 초기 정확도/천장 효과와 양립한다."
    if task == "gsm8k" and corrected == degraded:
        return "순정확도 동률은 교정과 퇴행의 상쇄다. 단순 천장 효과로만 해석하지 않는다."
    if task == "mmlu":
        return f"Debate의 순교정은 {corrected - degraded}문항이다. 교정/퇴행 trace를 함께 확인한다."
    return "교정과 퇴행의 방향을 trace에서 확인한다."


def write_summary(path, rows_by_task, selected):
    lines = ["# Qwen3-14B paired qualitative index", ""]
    for task, rows in rows_by_task.items():
        counts = Counter(row["transition"] for row in rows)
        lines.extend(
            [
                f"## {task.upper()}",
                "",
                f"- stable correct: {counts['stable_correct']}",
                f"- corrected: {counts['corrected']}",
                f"- degraded: {counts['degraded']}",
                f"- persistent wrong: {counts['persistent_wrong']}",
                f"- 해석: {interpretation(task, counts)}",
                "",
            ]
        )
    lines.extend(
        [
            "## 수동 코딩", "",
            "- 결과 전환: 유지 / 교정 / 퇴행 / 지속 실패",
            "- 초기 상태: 전원 정답 / 정답 다수 / 정답 소수 / 전원 오답",
            "- 변경 근거: 계산 검증 / 지식 근거 / 선택지 제거 / 모순 발견 / 단순 동조",
            "- 최종 수렴: 정답 수렴 / 오답 수렴 / 의견 분산",
            "- GSM8K 오류: 식 설정 / 계산 / 조건·단위 / 오류 전파 / 정답 추출",
            "- MMLU 오류: 지식 부족 / 선택지 혼동 / distractor / 소수 정답 무시 / 다수 편향",
            "",
            f"수동 분석 표본: {len(selected)}문항",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_case(task, item_id, path):
    data = load_result(path)
    if item_id not in data["debate"]:
        raise KeyError(f"{task}:{item_id} not found")
    compact = {
        "task": task,
        "item_id": item_id,
        "prompt": data["debate"][item_id].get("prompt"),
        "gold": data["debate"][item_id].get("gold"),
        "methods": {
            method: {
                key: data[method][item_id].get(key)
                for key in ("pred", "correct", "preds", "raw")
                if key in data[method][item_id]
            }
            for method in METHODS
        },
        "debate_trace": data["debate"][item_id].get("debate_trace"),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    paths = {"gsm8k": args.gsm8k_result, "mmlu": args.mmlu_result}
    if args.case:
        task, item_id = args.case.split(":", 1)
        if task not in paths:
            raise ValueError("--case task must be gsm8k or mmlu")
        print_case(task, item_id, paths[task])
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_task = {
        task: analyze_task(task, result_path)
        for task, result_path in paths.items()
    }
    for task, rows in rows_by_task.items():
        write_csv(args.output_dir / f"{task}_transition_index.csv", rows)

    selected = select_cases(rows_by_task["gsm8k"], args.gsm8k_sample)
    selected += select_cases(rows_by_task["mmlu"], args.mmlu_sample)
    sample_stem = f"qualitative_sample_{len(selected)}"
    write_csv(args.output_dir / f"{sample_stem}.csv", selected)
    (args.output_dir / f"{sample_stem}_ids.txt").write_text(
        "\n".join(
            f"{task}: "
            + ",".join(
                row["item_id"] for row in selected if row["task"] == task
            )
            for task in ("gsm8k", "mmlu")
        )
        + "\n",
        encoding="utf-8",
    )
    summary = args.output_dir / "summary.md"
    write_summary(summary, rows_by_task, selected)
    print(summary.read_text(encoding="utf-8"))
    print(f"[saved] {args.output_dir}")


if __name__ == "__main__":
    main()
