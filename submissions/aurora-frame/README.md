# AURORA FRAME — Bimanual Emergency Shelter Deployment Under Wind

**FFAI Robothon 2026 · Freestyle / real-world scenario**

AURORA FRAME is a physics-grounded emergency-response task: a fixed-base bimanual robot catches a spring-loaded shelter frame, physically presses three passive lock sliders, then holds the deployed structure through randomized wind gusts. It is designed around one judge-readable event: a collapsed field shelter becomes stable only after contact-driven latching and feedback-based gust correction.

## The task

The shelter is not an animated prop. Its frame is an unactuated MuJoCo hinge with gravity, spring stiffness, damping, friction, collisions, and two spatial guy-line tendons. The robot controls only its six Cartesian arm axes. The frame, three latch sliders, and tendons have **no actuators** and are never moved through direct pose edits after initialization.

Terminal success requires all of the following measured MuJoCo gates:

1. All three passive latch sliders reach at least 39 mm of physical travel.
2. The shelter returns to within 10.5° of its commanded safe deployment angle after the gust.
3. Maximum gust-induced deflection remains below 5°.

## Why this matters

Deploying temporary infrastructure is a real field-robotics problem. A robot must do more than lift an object: it has to manage a fast mechanical transformation, avoid an uncontrolled snap, verify a lock, and adapt its holding load while the environment changes. AURORA FRAME turns that challenge into a compact, auditable MuJoCo benchmark.

## Control and sensing

- Two force-limited XYZ Cartesian palms stabilize the passive frame through contact.
- A sensor-gated phase policy performs scan → catch → three latch presses → gust stabilization.
- Feedback uses measured frame hinge angle and velocity, guy-line tendon lengths, latch positions, and actual palm/handle contact force.
- During the storm phase, measured pitch error shifts the palm contact point toward the hinge to restore the shelter arc.
- The blind-timing baseline runs the same seeds and holds nominal palm poses without frame-pitch correction.

## Verified benchmark

The committed evidence was generated with:

```bash
python run_aurora_frame.py --cases 30 --demo-seed 4118
```

| Measurement | Closed-loop feedback | Blind timing baseline |
|---|---:|---:|
| Identical seeded storm trials | 30 | 30 |
| Full deploy / latch / gust success | **30 / 30 (100.0%)** | **12 / 30 (40.0%)** |
| Success-rate improvement | **+60.0 percentage points** | — |

The seed suite changes gust direction and magnitude, initial collapse angle, frame spring rate, and guy-line stiffness. Read the generated `outputs/metrics.json` for exact aggregate measurements and `outputs/paired_ablation.json` for every trial.

## Run

```bash
python -m pip install -r requirements.txt
python run_aurora_frame.py
python verify_submission.py
```

Fast smoke test:

```bash
python run_aurora_frame.py --quick --no-video
```

## Generated evidence

```text
outputs/aurora_frame_demo.mp4     77-second captioned physics demo + evidence card
outputs/aurora_frame_scene.xml    generated self-contained MJCF scene
outputs/metrics.json              aggregate benchmark and integrity notes
outputs/paired_ablation.json      all paired feedback / blind timing trials
outputs/benchmark.csv             flat per-trial results
outputs/demo_trajectory.json      20 Hz state, force, tendon and latch trace
outputs/sensor_policy_card.json   observation / action / safety-gate map
outputs/artifact_manifest.json    output manifest
```

## Honest limits

- This is a MuJoCo simulation, not a hardware deployment claim.
- Wind is represented as a seeded external force and pitch torque, not CFD airflow.
- The robot is intentionally fixed-base so the benchmark isolates coordinated shelter deployment rather than walking.
- Guy-lines are modeled by spatial tendons rather than deformable rope meshes.

Registration UUID: `540cf2be-b01e-4eee-a77a-44e07b3591dc` · GitHub submission account: `MicoCreator`
