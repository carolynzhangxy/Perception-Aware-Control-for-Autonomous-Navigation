# Robust Obstacle Avoidance via Vision-Based Hybrid Learning Control



### Introduction

Code implementation of the numerical results presented in *Robust Obstacle Avoidance 
via Vision-Based Hybrid Learning Control*. The PDF with the supplementary materials
contains the mathematical proofs along with the results of the numerical experiments.


**Abstract** We study the problem of target stabilization with robust obstacle avoidance
in robots and vehicles that have access to vision-based sensors generating streams of 
complex nonlinear data. This problem is particularly challenging due to the topological
obstructions induced by the obstacle, which preclude the existence of smooth feedback 
controllers able to achieve simultaneous stabilization and robust obstacle avoidance. 
To overcome this issue, we develop a *hybrid* controller that switches between two
different feedback laws depending on the current position of the vehicle. The main
innovation of the paper is the incorporation of perception maps, learned from data
obtained from the camera, into the hybrid controller. Moreover, under suitable
assumptions on the perception map, we establish theoretical guarantees for the
trajectories of the vehicle in terms of convergence and obstacle avoidance. The
proposed hybrid controller is numerically tested under different scenarios, including
under noisy data, sensor failures, and camera occlusions. Mathematical proofs and
illustrative simulation videos are included in the supplemental material.


### Requirements is stored in the `requirements.txt` file.


### Usage

To execute the experiments go to the _control_ directory and execute each file to run
the experiments (`waypoint_tracking` function) for the controller corresponding to 
the file's name. To see a simulation of a base scenario just execute the file.

Inside the `waypoint_tracking` function it is possible to tweak the controller's
parameters by modifying the `reference_params` and `tracker_params` dictionaries.
The variables `init_x` and `init_y` can be overwritten to define a new starting
position for the vehicles.

To test the _hybrid_controller_ against other controllers, inside the `waypoint_tracking`
function of the `hybrid_control.py` file it is possible to replace them (line 298). Further
information on how to do that, as well as, how to tweak the settings for execute the
remaining experiments is readily available in the comments within the Python files.

### layout of the repository

directory structure of the repository is as follows:

```.
├── control # Contains the code for the hybrid controller and the experiments
│   ├── hybrid_control.py # Hybrid controller implementation
│   ├── coontrol.py # Base controller implementation
│   ├── control_2.py # Second controller implementation

<!-- utils -->