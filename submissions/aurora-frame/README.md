# AURORA FRAME 2.0 - Adaptive Bimanual Shelter Recovery Under Wind

**FFAI Robothon 2026 · Freestyle / real-world emergency-response scenario**

AURORA FRAME 2.0 upgrades the original shelter-deployment benchmark into a harder recovery task: a fixed-base bimanual robot catches a spring-loaded emergency shelter, physically presses three passive lock sliders, then stabilizes the frame through calm wind, crosswind, rotating gusts, and a sudden microburst recovery scenario.

The key idea is simple and judge-readable: the robot is not just opening a shelter. It detects when wind starts pulling the shelter out of the safe deployment arc, then moves both palms to recover the frame before the latches and guy-lines lose stability.

## What changed in 2.0

- **Bigger benchmark:** 128 paired closed-loop vs blind-timing trials instead of 30.
- **Four wind profiles:** calm verification, steady crosswind, rotating gust, and microburst recovery.
- **Adaptive recovery phase:** the controller now enters `ADAPTIVE_GUST_RECOVERY` when frame pitch error or a microburst event is detected.
- **Better video telemetry:** the MP4 overlay shows wind profile, wind direction, frame error, recovery events, latch state, guy-line load, contact force, and gust deflection.
- **Cleaner evidence package:** generated metrics include scenario counts, recovery events, stability margin, and per-trial physical measurements.

## The task

The shelter is not an animated prop. Its frame is an unactuated MuJoCo hinge with gravity, spring stiffness, damping, friction, collisions, and two spatial guy-line tendons. The robot controls only its six Cartesian arm axes. The frame, three latch sliders, and tendons have **no actuators** and are never moved through direct pose edits after initialization.

Terminal success requires all of the following measured MuJoCo gates:

1. All three passive latch sliders reach at least 39 mm of physical travel.
2. The shelter returns to within 10.5 degrees of its safe deployment angle after the disturbance.
3. Maximum gust-induced deflection remains below 6.0 degrees.

## Control and sensing

- Two force-limited XYZ Cartesian palms stabilize the passive frame through contact.
- A sensor-gated phase policy performs scan -> catch -> three latch presses -> gust stabilization -> adaptive recovery.
- Feedback uses measured frame hinge angle and velocity, guy-line tendon lengths, latch positions, and actual palm/handle contact force.
- During rotating gusts and microbursts, measured pitch error shifts the palm contact point to recover the shelter arc while tendon lengths are monitored as a passive load check.
- The blind-timing baseline runs the same seeds and wind profiles without that residual correction.

## Verified benchmark

The committed evidence is generated with:

```bash
python run_aurora_frame.py --cases 128 --demo-seed 4118 --demo-scenario microburst_recovery
python verify_submission.py
```

| Measurement | Adaptive closed loop | Blind timing baseline |
|---|---:|---:|
| Identical seeded wind trials | 128 | 128 |
| Wind profiles | calm / crosswind / rotating gust / microburst recovery | same |
| Full deploy / latch / gust success | generated in `outputs/metrics.json` | generated in `outputs/metrics.json` |
| Recovery telemetry | generated in `outputs/demo_trajectory.json` | none |

Exact results are intentionally machine-generated rather than hand-entered. See `outputs/metrics.json`, `outputs/paired_ablation.json`, and `outputs/benchmark.csv`.

## Run

```bash
python -m pip install -r requirements.txt
python run_aurora_frame.py --cases 128 --demo-seed 4118 --demo-scenario microburst_recovery
python verify_submission.py
```

Fast smoke test:

```bash
python run_aurora_frame.py --quick --no-video
```

## Generated evidence

```text
outputs/aurora_frame_demo.mp4     captioned physics demo + benchmark end-card
outputs/aurora_frame_scene.xml    generated self-contained MJCF scene
outputs/metrics.json              aggregate benchmark and integrity notes
outputs/paired_ablation.json      paired feedback / blind timing trials
outputs/benchmark.csv             flat per-trial results
outputs/demo_trajectory.json      state, force, tendon, latch, and recovery trace
outputs/sensor_policy_card.json   observation / action / safety-gate map
outputs/artifact_manifest.json    output manifest
```

## Honest limits

- This is a MuJoCo simulation, not a hardware deployment claim.
- Wind is represented as seeded external force/torque profiles, not CFD airflow.
- The robot is intentionally fixed-base so the benchmark isolates coordinated shelter deployment rather than walking.
- Guy-lines are modeled by spatial tendons rather than deformable rope meshes.

Registration UUID: `540cf2be-b01e-4eee-a77a-44e07b3591dc` · GitHub submission account: `MicoCreator`
