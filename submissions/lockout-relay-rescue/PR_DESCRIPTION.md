Registration UUID: `266a431c-c10d-4904-ad50-701c36cb70dd`

## Project Summary

- **Project name:** Lockout Relay Rescue — Tactile Emergency Isolation
- **Robot platform:** 15-DOF five-finger hand on a Cartesian XYZ + wrist-yaw MuJoCo wrist
- **Task goal:** Safely isolate a moving industrial relay, press a verification latch, and seat a lockout tag.
- **Technical approach:** Phase policy with noisy marker-pose servoing, physical fingertip contact gating, tactile overload back-drive, and contact-gated rotary torque.
- **Core evidence:** Generated 67.7-second MP4, 36/36 feedback success versus 2/36 open-loop success on identical seeded panel motion, machine-readable metrics, full demo trajectory, and generated MJCF.
- **Current limitations:** Simulation only; marker-based visual observation; disclosed torque proxy for the rotary mechanism.

## How to Run

```bash
python -m pip install -r requirements.txt
python run_lockout_rescue.py
```

## Demo Video

`outputs/lockout_relay_demo.mp4`
