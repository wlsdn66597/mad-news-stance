"""Run zero-shot EXAONE segment labelling followed by one article prediction."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.joa_zero_shot import annotate_markup, segment_prompt  # noqa: E402
from src.llm import build_messages, chat, load_model, model_context_window, strip_think  # noqa: E402
from src.metrics import LABELS_DEFAULT, format_report  # noqa: E402
from src.tasks.stance import Stance  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/phase2_exaone_stance_minimal_en.yaml"
    )
    parser.add_argument("--model", default="exaone")
    parser.add_argument(
        "--segments",
        default="data/k-news-stance_test1001_joa_segments_unlabeled.json",
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--run-seed", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--segment-max-new-tokens", type=int, default=32)
    parser.add_argument("--article-max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--parse-fallback",
        choices=["supportive", "neutral", "oppositional", "error"],
        default="neutral",
        help="post-processing for a malformed segment answer; no retry/model call",
    )
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def generation_seed(run_seed: int, *parts) -> int:
    token = "\0".join(str(part) for part in (run_seed,) + parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF


def save(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_overlay(path: Path, completed: dict):
    rows = []
    for item_id, result in completed.items():
        rows.append(
            {
                "id": int(item_id) if item_id.isdigit() else item_id,
                "event_name": result["issue"],
                "title_joa_icl": result["title_joa_icl"],
                "main_body_joa_icl": result["main_body_joa_icl"],
            }
        )
    rows.sort(key=lambda row: str(row["id"]))
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.model not in cfg["models"]:
        raise ValueError(f"model {args.model!r} is not defined in {args.config}")

    with open(args.segments, encoding="utf-8") as handle:
        prepared_rows = json.load(handle)
    prepared = {str(row["id"]): row for row in prepared_rows}
    if any("입장=" in json.dumps(row, ensure_ascii=False) for row in prepared_rows):
        raise ValueError("the segment-agent input still contains a stance attribute")

    split = args.split or cfg.get("split", "test")
    n = args.n if args.n is not None else cfg.get("n")
    data_seed = cfg.get("seed", 0) if args.data_seed is None else args.data_seed
    temperature = (
        args.temperature
        if args.temperature is not None
        else cfg.get("sampling", {}).get("temperature", 1.0)
    )
    article_max_new_tokens = args.article_max_new_tokens or cfg["max_new_tokens"]
    profile = cfg.get("prompt_profile", "stance_minimal_en")
    task = Stance(cfg["data_path"], prompt_profile=profile)
    items = task.load(split=split, n=n, seed=data_seed)
    missing = [str(item["id"]) for item in items if str(item["id"]) not in prepared]
    if missing:
        raise ValueError(f"{len(missing)} selected items have no segment data: {missing[:5]}")

    tag = f"_{args.tag}" if args.tag else ""
    output = Path(
        f"results/joa_zero_shot/{args.model}_{split}_n{len(items)}_"
        f"d{data_seed}_s{args.run_seed}{tag}.json"
    )
    overlay = output.with_name(output.stem + "_segment_labels.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        results = json.loads(output.read_text(encoding="utf-8"))
    else:
        results = {"_meta": {}, "items": {}}
    completed = results.setdefault("items", {})

    model_cfg = cfg["models"][args.model]
    print(f"[load] {model_cfg['id']}", flush=True)
    set_seed(args.run_seed)
    model, tokenizer = load_model(
        model_cfg["id"],
        load_in_4bit=cfg["load_in_4bit"],
        gpu_index=cfg["gpu_index"],
        mem_fraction=cfg.get("mem_fraction"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    results["_meta"] = {
        "pipeline": "zero_shot_segment_agent_then_zero_shot_article_agent",
        "config": args.config,
        "model_id": model_cfg["id"],
        "segments": args.segments,
        "split": split,
        "n": len(items),
        "data_seed": data_seed,
        "run_seed": args.run_seed,
        "temperature": temperature,
        "top_p": 0.8,
        "segment_max_new_tokens": args.segment_max_new_tokens,
        "article_max_new_tokens": article_max_new_tokens,
        "segment_parse_fallback": args.parse_fallback,
        "context_window": model_context_window(model, tokenizer),
        "debate": False,
        "segment_calls_per_span": 1,
        "article_calls_per_item": 1,
    }
    save(output, results)

    started = time.time()
    new_items = 0
    for position, item in enumerate(items, start=1):
        key = str(item["id"])
        if key in completed:
            continue
        row = prepared[key]
        segment_results = []
        fallback_count = 0
        for segment in row["segments"]:
            prompt = segment_prompt(item["issue"], segment["text"])
            seed = generation_seed(args.run_seed, key, segment["segment_id"])
            set_seed(seed)
            raw = strip_think(
                chat(
                    model,
                    tokenizer,
                    build_messages(prompt, system_prompt=model_cfg.get("system_prompt")),
                    max_new_tokens=args.segment_max_new_tokens,
                    temperature=temperature,
                )
            )
            pred = task.parse(raw)
            used_fallback = pred is None
            if used_fallback:
                if args.parse_fallback == "error":
                    raise ValueError(
                        f"segment parse failure for item {key} {segment['segment_id']}: {raw!r}"
                    )
                pred = args.parse_fallback
                fallback_count += 1
            segment_results.append(
                {
                    **segment,
                    "pred": pred,
                    "raw": raw,
                    "generation_seed": seed,
                    "parse_fallback": used_fallback,
                }
            )

        annotated_title = annotate_markup(
            row["title_joa_unlabeled"], "title", segment_results
        )
        annotated_body = annotate_markup(
            row["main_body_joa_unlabeled"], "body", segment_results
        )
        article_item = {
            **item,
            "headline": annotated_title,
            "article": annotated_body,
            "segment_labeled": True,
        }
        article_prompt = task.question(article_item, style="vanilla")
        article_seed = generation_seed(args.run_seed, key, "article")
        set_seed(article_seed)
        article_raw = strip_think(
            chat(
                model,
                tokenizer,
                build_messages(
                    article_prompt, system_prompt=model_cfg.get("system_prompt")
                ),
                max_new_tokens=article_max_new_tokens,
                temperature=temperature,
            )
        )
        article_pred = task.parse(article_raw)
        completed[key] = {
            "issue": item["issue"],
            "gold": item["gold"],
            "pred": article_pred,
            "correct": article_pred == item["gold"],
            "segment_parse_fallbacks": fallback_count,
            "segments": segment_results,
            "title_joa_icl": annotated_title,
            "main_body_joa_icl": annotated_body,
            "article_prompt": article_prompt,
            "article_raw": article_raw,
            "article_generation_seed": article_seed,
        }
        new_items += 1
        save(output, results)
        if new_items % 10 == 0 or position == len(items):
            elapsed = time.time() - started
            rate = new_items / elapsed if elapsed else 0.0
            remaining = (len(items) - len(completed)) / rate if rate else 0.0
            print(
                f"[progress] {len(completed)}/{len(items)} "
                f"elapsed={elapsed / 3600:.2f}h remaining={remaining / 3600:.2f}h",
                flush=True,
            )

    write_overlay(overlay, completed)
    ordered = [completed[str(item["id"])] for item in items if str(item["id"]) in completed]
    preds = [entry["pred"] for entry in ordered]
    golds = [entry["gold"] for entry in ordered]
    print("\n===== ZERO-SHOT JOA REPORT =====")
    print(format_report(preds, golds, LABELS_DEFAULT))
    print(f"segment_parse_fallbacks={sum(x['segment_parse_fallbacks'] for x in ordered)}")
    print(f"[saved] {output}")
    print(f"[segment-label overlay] {overlay}")


if __name__ == "__main__":
    main()
