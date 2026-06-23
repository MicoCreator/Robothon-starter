## Registration UUID

`540cf2be-b01e-4eee-a77a-44e07b3591dc`

## Project Summary

- **Project name:** AURORA FRAME — Bimanual Emergency Shelter Deployment Under Wind
- **Robot platform:** Fixed-base bimanual emergency-response rig with six force-limited Cartesian axes
- **Task goal:** Catch a passive spring-loaded shelter, physically engage three lock sliders, and stabilize the frame against randomized gusts.
- **Technical approach:** Sensor-gated bimanual phase policy using frame hinge residual, spatial-tendon length, latch position, and palm-handle contact force.
- **Core evidence:** Generated 77-second MP4, self-contained MJCF, 30 paired seeded trials, and machine-readable metric/trajectory artifacts.
- **Verified result:** **30/30 closed-loop passes** versus **12/30 blind-timing passes** on identical seeded conditions.
- **Current limits:** MuJoCo-only; external wind load rather than CFD; fixed-base robot; tendon rather than deformable-rope model.

## How to run

```bash
python -m pip install -r requirements.txt
python run_aurora_frame.py --cases 30 --demo-seed 4118
python verify_submission.py
```

## Demo video

`outputs/aurora_frame_demo.mp4`

## Checklist

- [x] `registration.json` contains this PR UUID.
- [x] This PR description contains the same UUID.
- [x] The task objects have no direct actuators.
- [x] Seeded benchmark, baseline, video, and verification script are included.
