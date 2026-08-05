"""Copy a finished run's shared Round 0 into a new result file.

`run_phase2.py` skips shared Round 0 items that already exist in the output
file, so pre-seeding lets a follow-up run (e.g. more rounds) start from the
exact same initial answers instead of re-sampling them. That makes the two runs
strictly paired and saves n_agents generations per item.

    python scripts/seed_shared_round0.py \
      --from results/phase2/<run>_a3_r2.json \
      --to   results/phase2/<run>_a3_r4.json

The target must not exist yet; nothing is ever overwritten.
"""
import argparse
import json
from pathlib import Path

# Everything that must agree for the initial answers to be reusable. n_rounds is
# deliberately absent: that is the setting the follow-up run changes.
COMPATIBLE_META = (
    "model_id",
    "load_in_4bit",
    "prompt_profile",
    "split",
    "n",
    "data_seed",
    "run_seed",
    "temperature",
    "max_new_tokens",
    "n_agents",
    "share_round0",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="source", required=True)
    parser.add_argument("--to", dest="target", required=True)
    parser.add_argument("--force", action="store_true", help="allow overwriting the target")
    return parser.parse_args()


def main():
    args = parse_args()
    source_path, target_path = Path(args.source), Path(args.target)
    if target_path.exists() and not args.force:
        raise SystemExit(
            f"{target_path} already exists; refusing to overwrite (pass --force to replace it)"
        )

    source = json.loads(source_path.read_text(encoding="utf-8"))
    meta = source.get("_meta", {})
    shared = source.get("_shared_round0") or {}
    if not shared:
        raise SystemExit(f"{source_path} has no _shared_round0 to copy")
    if not meta.get("share_round0"):
        raise SystemExit(f"{source_path} was not run with share_round0; the answers are not paired")
    missing = [key for key in COMPATIBLE_META if key not in meta]
    if missing:
        raise SystemExit(f"{source_path} _meta lacks {', '.join(missing)}")

    incomplete = [key for key, value in shared.items() if not value.get("raw")]
    if incomplete:
        raise SystemExit(f"{len(incomplete)} shared Round 0 entries have no answers")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps({"_shared_round0": shared}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[seeded] {len(shared)} shared Round 0 items -> {target_path}")
    print("[reused settings] " + "  ".join(f"{key}={meta[key]}" for key in COMPATIBLE_META))
    print("Run run_phase2.py with the settings above (n_rounds may differ); it will reuse "
          "these answers and generate only the later rounds.")


if __name__ == "__main__":
    main()
