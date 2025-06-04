# Perception-Aware Control for Autonomous Navigation

## Introduction

This repository contains the official implementation of the numerical experiments presented in the ECE 228 report *Perception-Aware Control for Autonomous Navigation*. 

### Abstract

Autonomous navigation in dynamic environments remains a challenging task due to rapidly changing conditions, sensor noise, and occlusions that impair perception and decision-making. This project introduces a robust perception-based hybrid control framework that integrates state-of-the-art object detection models—YOLOv10 and RT-DETR—with a learned perception map and a hybrid dynamical control system for real-time obstacle avoidance.

The perception map transforms image data into position estimates, enabling the controller to dynamically adapt to occlusions and adversarial disturbances. A hysteresis-based switching mechanism ensures smooth transitions between control modes and prevents Zeno behaviors.

Extensive simulations in a 2D grid environment show that the proposed system significantly improves accuracy. It achieves a Root Mean Square Error (RMSE) of 1.13 under normal conditions, 1.15 under occlusion, and 1.13 under adversarial control noise—outperforming a baseline CNN-based hybrid controller with RMSEs of 2.84 and 3.13 under comparable conditions. These results validate the system’s robustness, stability, and effectiveness for real-time autonomous navigation in uncertain and dynamic settings.

---

## Requirements

All dependencies are listed in the `requirements.txt` file.

---

## Usage

To run the experiments, navigate to the `control` directory and execute the `hybrid_control.py` by 

```bash
python hybrid_control.py
```
### Configuring the Controller

You can customize the controller behavior by modifying:
- `reference_params`: defines the target trajectory
- `tracker_params`: sets tracking controller gains

You may also adjust the vehicle's initial state using the `init_x` and `init_y` variables.

### Comparing Controllers

To compare the hybrid controller against alternatives:
1. Open `control/hybrid_control.py`.
2. Go to the function `waypoint_tracking` and modify the control selection logic.
3. Comments within the code offer guidance for running additional experiments or modifying controller configurations.
---

## Repository Structure

```text
├── control/                          # Controller scripts
│   ├── hybrid_control.py             # Main hybrid controller (used for core experiments)
│   ├── control.py                    # Baseline controller
│   ├── control2.py                   # Alternative controller
│   └── hybrid_control_multobstacles.py # Extension for multiple obstacles (WIP, future direction)

├── detect/                           # Object detection models and inference
│   ├── object_detect_cli.py
│   ├── object_detector.py
│   └── object_detector_model.pth     # Trained detector weights

├── clf/                              # Classifier models
│   ├── car_model.pth
│   ├── new_model.pth
│   ├── clf.py
│   ├── inference.py
│   ├── car_classes.txt
│   └── new_classes.txt

├── data/                             # Data loader utilities
│   └── data_load.py

├── Figure_outputs/                   # Visualizations of experiment results
│   └── *.png

├── frames_FollowerLevelSets_*/       # Saved simulation frame sequences

├── models/
│   └── linear_model.py               # Optional baseline model

├── no_occlusion/, occlusion/         # Simulation frames for different noise conditions

├── calculate_avg_inference.py        # Measures average model inference time
├── requirements.txt                  # Dependency list
