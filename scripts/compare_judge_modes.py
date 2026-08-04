"""Compare article-only and debate-trace selective-judge outputs."""
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article-only", required=True, help="article-only *.items.json")
    parser.add_argument("--debate-trace", required=True, help="debate-trace *.items.json")
    parser.add_argument("--output", default="results/selective_judge/judge_mode_comparison.json")
    return parser.parse_args()


def load(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["item_id"]): row for row in rows}


def score(rows):
    values = list(rows.values())
    return {
        "items": len(values),
        "accuracy": sum(row["final_prediction"] == row["gold"] for row in values) / max(1, len(values)),
        "judge_calls_including_retries": sum(len(row.get("judge_attempts", [])) for row in values),
        "input_tokens": sum(row.get("token_usage", {}).get("input_tokens", 0) for row in values),
        "output_tokens": sum(row.get("token_usage", {}).get("output_tokens", 0) for row in values),
        "latency_seconds": sum(row.get("latency_seconds", 0.0) for row in values),
    }


def main():
    args = parse_args()
    article, trace = load(args.article_only), load(args.debate_trace)
    if set(article) != set(trace):
        raise ValueError("judge modes do not contain the same item IDs")
    for item_id in article:
        left, right = article[item_id], trace[item_id]
        for field in ("judge_model", "trigger_mode", "judge_seed"):
            if left.get(field) != right.get(field):
                raise ValueError(f"item {item_id}: mismatched {field}")
        if left.get("gold") != right.get("gold"):
            raise ValueError(f"item {item_id}: mismatched gold label")
    article_score, trace_score = score(article), score(trace)
    transitions = {"article_only_wins": 0, "debate_trace_wins": 0, "same": 0}
    for item_id in article:
        a_ok = article[item_id]["final_prediction"] == article[item_id]["gold"]
        t_ok = trace[item_id]["final_prediction"] == trace[item_id]["gold"]
        key = "same" if a_ok == t_ok else ("article_only_wins" if a_ok else "debate_trace_wins")
        transitions[key] += 1
    report = {
        "same_item_ids": True,
        "article_only": article_score,
        "debate_trace": trace_score,
        "accuracy_delta_trace_minus_article": trace_score["accuracy"] - article_score["accuracy"],
        "paired_outcomes": transitions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
