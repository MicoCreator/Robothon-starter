"""Integrity check for a completed AURORA FRAME artifact set."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
REQUIRED = (
    "aurora_frame_demo.mp4",
    "aurora_frame_scene.xml",
    "metrics.json",
    "paired_ablation.json",
    "demo_trajectory.json",
    "sensor_policy_card.json",
    "benchmark.csv",
    "artifact_manifest.json",
)


def main() -> None:
    missing = [name for name in REQUIRED if not (OUTPUTS / name).is_file()]
    if missing:
        raise SystemExit(f"Missing generated artifacts: {', '.join(missing)}")
    video = OUTPUTS / "aurora_frame_demo.mp4"
    if video.stat().st_size < 250_000:
        raise SystemExit("Demo MP4 is unexpectedly small; rerun without --no-video.")
    metrics = json.loads((OUTPUTS / "metrics.json").read_text(encoding="utf-8"))
    feedback = metrics["feedback"]
    baseline = metrics["blind_timing_baseline"]
    if feedback["trials"] < 1 or feedback["successes"] != feedback["trials"]:
        raise SystemExit("Closed-loop benchmark is incomplete or contains failed trials.")
    if feedback["success_rate"] <= baseline["success_rate"]:
        raise SystemExit("Feedback does not outperform the blind-timing baseline.")
    scene = (OUTPUTS / "aurora_frame_scene.xml").read_text(encoding="utf-8")
    forbidden = ("<position name=\"frame", "<motor name=\"frame", "<position name=\"latch", "<motor name=\"latch")
    if any(token in scene for token in forbidden):
        raise SystemExit("Detected an unexpected task-object actuator in the generated MJCF.")
    print("PASS: artifacts, physical-actuation audit, and feedback-vs-baseline benchmark verified.")


if __name__ == "__main__":
    main()
