"""Lightweight integrity check for a completed Lockout Relay Rescue run."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
REQUIRED = [
    "lockout_relay_demo.mp4",
    "lockout_relay_scene.xml",
    "metrics.json",
    "paired_ablation.json",
    "demo_trajectory.json",
    "sensor_policy_card.json",
    "benchmark.csv",
    "artifact_manifest.json",
]


def main() -> None:
    missing = [name for name in REQUIRED if not (OUTPUTS / name).is_file()]
    if missing:
        raise SystemExit(f"Missing generated artifacts: {', '.join(missing)}")
    if (OUTPUTS / "lockout_relay_demo.mp4").stat().st_size < 100_000:
        raise SystemExit("Demo video is unexpectedly small; rerun without --no-video.")
    metrics = json.loads((OUTPUTS / "metrics.json").read_text(encoding="utf-8"))
    feedback = metrics["feedback"]
    baseline = metrics["open_loop_baseline"]
    if feedback["trials"] < 1 or feedback["successes"] != feedback["trials"]:
        raise SystemExit("Feedback benchmark is incomplete or contains failed trials.")
    if feedback["success_rate"] <= baseline["success_rate"]:
        raise SystemExit("Feedback benchmark does not outperform the open-loop baseline.")
    print("PASS: artifacts present, video present, and feedback benchmark outperforms baseline.")


if __name__ == "__main__":
    main()
