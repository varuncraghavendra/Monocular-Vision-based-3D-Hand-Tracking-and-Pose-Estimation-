# Monocular-Vision-based-3D-Hand-Tracking-and-Pose-Estimation

**CS5330 Pattern Recognition and Computer Vision — Final Project**
**Varun Raghavendra | Spring 2026**

Real-time dual-hand 3D pose estimation pipeline for robot learning and fine dextrous manipulation, built on [MMPose](https://github.com/open-mmlab/mmpose) and [InterNet (ECCV 2020)](https://arxiv.org/abs/2008.09309).

---

## Demo

| Camera overlay | 3D pose graph |
|---|---|
| Per-finger coloured thin skeleton on 960×720 live feed | Real-time matplotlib 3D plot matching finger colours |

The window is split: **left** = camera + skeleton overlay, **right** = 3D hand pose graph updated every frame.

---

## Features

- **Dual-hand tracking** — detects and tracks both hands simultaneously using InterNet's 42-keypoint output (21 per hand), with independent per-slot EMA smoothing and 12-frame miss tolerance before a slot resets
- **3D pose graph** — real-time matplotlib `Axes3D` rendering of hand skeleton in 3D space, with the same per-finger colour coding as the camera overlay
- **Per-finger colour skeleton** — pink=thumb, blue=index, green=middle, orange=ring, purple=pinky, with a white palm bar connecting the knuckles
- **1€ Filter** — per-element adaptive smoothing on XY keypoints; fast motion gets low smoothing, stationary joints get heavy smoothing, eliminating jitter without adding lag
- **Gesture classification** — OPEN PALM vs GRASP, strongly biased toward GRASP: any single non-thumb finger curling below threshold triggers GRASP immediately; returning to OPEN PALM requires all fingers clearly extended for 4 consecutive frames
- **Monodepth2 depth estimation** — monocular depth map from the camera feed, displayed as a colourised inset
- **Palm depth calibration** — one-shot calibration: hold open palm at 40 cm, press `c`, wait 3 seconds; depth values become real metres; calibrated depth also improves hand tracking via depth-consistent slot matching
- **Depth-aided tracking** — after calibration, the tracker penalises slot assignments where the detected palm depth differs significantly from the slot's last known depth, preventing slots from swapping when two hands cross in 2D

---

## System Requirements

| Component | Requirement |
|---|---|
| OS | Ubuntu 20.04 / 22.04 |
| Python | 3.10 |
| GPU | NVIDIA GPU with CUDA (tested on RTX 4080 Laptop) |
| CUDA | 12.1 |
| PyTorch | 2.1.0+cu121 |
| MMCV | 2.1.0 |
| Webcam | Any USB webcam capable of 960×720 |

---

## Installation

### 1. Clone MMPose and set up the venv

```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
python3 -m venv ~/venvs/mmpose-hand3d
source ~/venvs/mmpose-hand3d/bin/activate
```

### 2. Install PyTorch with CUDA

```bash
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install MMCV

```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

### 4. Install MMPose and dependencies

```bash
pip install -e . --no-build-isolation
pip install "numpy<2" matplotlib
```

### 5. Fix NumPy compatibility

```bash
pip install "numpy<2" --force-reinstall
```

---

## Model Checkpoints

Place checkpoints in the following locations:

```
mmpose/
├── checkpoints/
│   └── res50.pth                          ← InterNet ResNet-50 checkpoint
└── projects/robot_learning_hand_demo/
    └── checkpoints/
        └── mono_640x192/
            ├── encoder.pth                ← Monodepth2 encoder
            └── depth.pth                  ← Monodepth2 decoder
```

**InterNet checkpoint** — download from [OpenMMLab](https://download.openmmlab.com/mmpose/v1/hand_3d_keypoint/internet/interhand3d/internet_res50_interhand3d-d6ff20d6_20230913.pth) and rename to `res50.pth`.

**Monodepth2 checkpoint** — download `mono_640x192` from the [Monodepth2 releases](https://github.com/nianticlabs/monodepth2) and extract into `checkpoints/mono_640x192/`.

---

## Running

Activate the venv first:

```bash
source ~/venvs/mmpose-hand3d/bin/activate
cd ~/path/to/mmpose
```

Run with GPU (recommended):

```bash
python3 projects/robot_learning_hand_demo/scripts/run_robot_learning_gui.py --device cuda
```

Run on CPU (slower):

```bash
python3 projects/robot_learning_hand_demo/scripts/run_robot_learning_gui.py --device cpu
```

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--device` | `cpu` | `cuda` or `cpu` |
| `--score-thr` | `0.08` | Minimum keypoint confidence threshold |
| `--depth-model` | `projects/robot_learning_hand_demo/checkpoints/mono_640x192` | Path to Monodepth2 checkpoint directory |
| `--infer-scale` | `0.55` | Frame resize factor before inference (lower = faster) |

---

## Keyboard Controls

| Key | Action |
|---|---|
| `ESC` | Quit |
| `c` | Start depth calibration countdown (3 seconds) |
| `r` | Reset depth calibration |

### Depth calibration procedure

1. Press `c`
2. Hold your open palm flat, facing the camera, at **exactly 40 cm** distance
3. Hold still for 3 seconds while the progress bar fills
4. The inset changes from **Depth UNCAL** (gray) to **Depth CAL** (green)
5. Depth readings next to each hand are now in real metres

---

## Project Structure

```
projects/robot_learning_hand_demo/
├── scripts/
│   └── run_robot_learning_gui.py     ← Entry point
├── src/
│   ├── pipeline.py                   ← Main loop, rendering, UI
│   ├── pose_backends.py              ← InterNet inference + dual-hand tracker
│   ├── depth_estimator.py            ← Monodepth2 wrapper + calibration
│   ├── gesture_abstraction.py        ← OPEN PALM / GRASP classifier
│   ├── one_euro_filter.py            ← Per-element 1€ adaptive filter
│   ├── camera.py                     ← Camera capture
│   └── monodepth2_networks.py        ← Monodepth2 encoder/decoder networks
└── checkpoints/
    └── mono_640x192/                 ← Monodepth2 weights
```

---

## Architecture

```
Camera (960×720)
    │
    ├──► Monodepth2 ──► Depth map (H×W float32)
    │         │
    │         └──► Palm calibration (40 cm reference)
    │
    └──► InterNet ResNet-50 ──► 42 keypoints (21 per hand, xyz)
              │
              ├──► Deduplication + side-label slot matching
              ├──► EMA tracking (depth-aided after calibration)
              ├──► 1€ Filter (per-element, XY only)
              ├──► Gesture classifier (GRASP / OPEN PALM)
              └──► matplotlib 3D renderer ──► pose graph panel
```

---

## Methodology

### Pose estimation
InterNet (Moon et al., ECCV 2020) is a top-down 3D hand pose estimator trained on InterHand2.6M (2.6 million annotated frames, multi-view 3D ground truth). It outputs 42 keypoints per inference — 21 for each hand in (x, y, z) coordinates. The keypoint ordering within each 21-kp block is **tip→wrist**: index 0 = thumb tip, index 20 = wrist.

### Tracking
Two slots are maintained (right=0, left=1). Slot assignment is done in two passes: first by the hand-side label predicted by InterNet, then by a proximity + depth consistency score for any unmatched slots. EMA smoothing (α=0.35) on slot centre, size, and depth prevents transient bad frames from teleporting the tracker. Slots survive up to 12 consecutive missed frames before resetting.

### Smoothing
The 1€ Filter computes a per-element adaptive alpha so fast-moving joints get high alpha (low lag) and stationary joints get heavy smoothing. This is applied to XY only; the z coordinate comes directly from InterNet.

### Gesture classification
The classifier uses the tip/MCP distance ratio: `dist(tip, wrist) / dist(MCP, wrist)`. A fully extended finger yields ratio ~2.8–4.0; a slightly curled finger (pen grip, knife hold) yields ~1.6–2.4. The classifier defaults to GRASP and requires all four non-thumb fingers to simultaneously exceed ratio 2.4 for 4 consecutive frames before switching to OPEN PALM.

### Depth
Monodepth2 (Godard et al., ICCV 2019) provides a relative monocular depth map. Since Monodepth2 was trained on driving data, the scale is meaningless without calibration. The one-shot palm calibration computes a scalar `calib_scale = 0.40 / raw_depth_at_palm`, converting subsequent maps to metric metres. After calibration, the tracker incorporates depth consistency into slot scoring.

---

## Known Limitations

- Monodepth2 was trained on outdoor driving scenes (KITTI). Indoor close-range hand footage is out-of-distribution, so absolute depth accuracy is limited even after calibration. The relative depth ordering (which hand is closer) is more reliable than absolute values.
- InterNet's z-coordinate output is relative, not metric. The 3D pose plot reflects the model's relative depth estimates, not true physical coordinates.
- Detection can drop under fast motion, heavy occlusion, or very dark/backlit conditions.

---

## References

- **InterNet**: Moon et al., "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image", ECCV 2020
- **MMPose**: OpenMMLab Pose Estimation Toolbox, https://github.com/open-mmlab/mmpose
- **Monodepth2**: Godard et al., "Digging into Self-Supervised Monocular Depth Prediction", ICCV 2019
- **1€ Filter**: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012
