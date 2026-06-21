# Lockout Relay Rescue — Tactile Emergency Isolation

**FFAI Robothon 2026 · Freestyle / real-world scenario**

Lockout Relay Rescue is a self-contained MuJoCo safety-maintenance benchmark. A 15-DOF five-finger hand on a Cartesian wrist must isolate a moving industrial relay, press its verification latch, and physically seat a tool-mounted lockout tag. The robot uses measured fingertip contact, a noisy panel-marker pose observation, isolator angle/velocity, latch travel, and tag/slot poses.

## Why this project exists

Emergency lockout procedures are safety-critical: the correct control must be isolated before a maintenance tag is placed. The difficult part is not reaching a static panel; it is maintaining a force-limited grasp when the panel shifts, correcting alignment from observations, and verifying the final safety state.

## What is implemented

- **15-DOF five-finger hand:** five radial closing joints and ten finger flex joints, plus a Cartesian XYZ wrist and wrist-yaw control.
- **Real MuJoCo contacts:** five fingertip touch sensors, collisions, actuator limits, gravity, panel jitter, a rotary isolator, spring-return latch, and a physical tool-mounted lockout tag.
- **Sensor-driven recovery:** the feedback controller re-targets the hand using a noisy calibrated panel-marker observation after a seeded two-axis panel shift. A 6 N tactile guard back-drives overloaded fingers.
- **Force-gated isolation:** the rotary mechanism receives torque only after at least three physical fingertip contacts and a measured alignment gate.
- **Safety sequence:** tactile grasp → isolation rotation → verification latch → tag seating → terminal checks.
- **Paired ablation:** feedback and open-loop control run the same seeded disturbances. Every metric is written by the submitted program.

## Verified benchmark result

The included `outputs/` directory was generated with:

```bash
python run_lockout_rescue.py --cases 36
```

| Measurement | Feedback controller | Open-loop baseline |
|---|---:|---:|
| Seeded panel-jitter trials | 36 | 36 |
| End-to-end safe-lockout success | **36 / 36 (100.0%)** | 2 / 36 (5.6%) |
| Mean final isolation angle | 86.9° | 38.8° |
| Mean final tag/slot error | 14.1 mm | 42.9 mm |
| Mean peak tactile reading | 50.7 N | 240.5 N |

That is a **94.4 percentage-point** success improvement from the measured feedback path. The tactile value is a simulation contact signal; the controller uses a 6 N command back-drive threshold, so it is not presented as hardware-safe force certification.

## Run

```bash
python -m pip install -r requirements.txt
python run_lockout_rescue.py
```

For a fast smoke test:

```bash
python run_lockout_rescue.py --quick --no-video
```

## Generated evidence

Running the command produces:

```text
outputs/lockout_relay_demo.mp4      generated 68-second telemetry demo
outputs/lockout_relay_scene.xml     generated MJCF scene
outputs/metrics.json                aggregate benchmark report
outputs/paired_ablation.json        seeded feedback vs open-loop trials
outputs/benchmark.csv               flat per-trial metrics
outputs/demo_trajectory.json        20 Hz sensor/action trace for the demo
outputs/sensor_policy_card.json     observation/action/safety map
outputs/artifact_manifest.json      output manifest
```

## Technical approach

The controller is an explicit phase policy with minimum-jerk phase transitions. It uses a **noisy panel-marker pose observation** to compensate panel motion and uses the five tactile readings to gate the rotary action and back-drive an overloaded radial finger command. The rotary switch is simulated as an actuated physical mechanism receiving a contact-gated wrist-torque proxy; it is not moved by directly setting its pose. The lockout tag is a physical tool mounted to the hand rather than a free grasped object.

## Honest limitations

- This is a simulation-only benchmark; it does not claim a physical robot experiment.
- The visual input is a noisy calibrated marker observation, not an end-to-end neural vision system.
- The rotary coupling is a disclosed torque proxy; a hardware version would require calibrated wrist-force transfer.

## Submission contents

`run_lockout_rescue.py` is intentionally self-contained and procedurally generates the MJCF, demo, reports, and data artifacts. No external robot meshes, pretrained models, or hidden assets are required.
