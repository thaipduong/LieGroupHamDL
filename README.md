# Port-Hamiltonian Neural ODE Networks on Lie Groups For Robot Dynamics Learning and Control
This repo provides code for our paper "Port-Hamiltonian Neural ODE Networks on Lie Groups For Robot Dynamics Learning and Control".
Please check out our project website for more details: https://thaipduong.github.io/LieGroupHamDL/.

This branch contains ROS packages for our experiments with PX4 drones. It's tested with:
- Ubuntu 18.04
- ROS Melodic
- Gazebo 9
- PX4 (release version 1.10.0 for Ubuntu 18.04)

# Setup in Ubuntu 18.04
Installation instructions for Ubuntu 18.04. Ubuntu 18.04 requires ROS Melodic, Gazebo9 and PX4 Firmware release 1.10.0
_________________________________________________________________


## First time setup:
1. Install ROS Melodic under Ubuntu 18.04 http://wiki.ros.org/melodic/Installation/Ubuntu
2. Run ubuntu_sim_ros_melodic.sh from https://dev.px4.io/v1.10/en/setup/dev_env_linux_ubuntu.html#rosgazebo
     - This installs additional tools (mavros, px4 sitl) necessary for integration with px4 and running/visualizing drones.
     ```
     wget https://raw.githubusercontent.com/PX4/Devguide/v1.10/build_scripts/ubuntu_sim_ros_melodic.sh
     chmod +x ubuntu_sim_ros_melodic.sh
     source ubuntu_sim_ros_melodic.sh
     ```
3. Dependencies
    - Install mavros
    ```dependencies
    sudo apt-get install ros-melodic-mavros ros-melodic-mavros-extras
    ```
    - This installs python tools for the GUI
    ```
    sudo apt install libqt4-dev
    sudo apt install python-qt4 pyqt4-dev-tools
    sudo apt install python-pyside pyside-tools
    ```

    - This install ```torchode``` python package for training with dataset that has irregular time steps, which is common for data from real systems such as PX4 quadrotor.
    ```
    pip install torchode
    ```
4. Install our modified PX4 firmware
    - Clone the PX4 firmware repository: git@github.com:ExistentialRobotics/erl_quadrotor_firmware.git
    - The default PX4 location for this package: ~/PX4/Firmware
    ```
    git clone --branch release/1.10 git@github.com:ExistentialRobotics/erl_quadrotor_firmware.git --recursive
    cd Firmware
    git submodule update --init --recursive
    bash ./Tools/setup/ubuntu.sh
    make px4_sitl_default gazebo
    ```
    - Upgrade ```sudo apt upgrade libignition-math2``` and ```sudo apt upgrade libsdformat6``` before ```bash ./Tools/setup/ubuntu.sh``` if running into symbol errors with gazebo
    - Change ```#define HAS_GYRO TRUE``` to ```#define HAS_GYRO true``` in this file ```Tools/sitl_gazebo/include/gazebo_opticalflow_plugin.h``` if running into errors with higher version of gcc.

## ERL Quadrotor Control 
* Clone our repository ```git@github.com:thaipduong/LieGroupHamDL.git```, checkout ```PX4``` branch and copy the packages in ```ros``` folder in the ```~/catkin_ws/src``` directory:

```angular2html
git clone git@github.com:thaipduong/LieGroupHamDL.git
git checkout PX4
cp ros/* ~/catkin_ws/src/
```


## Geometric Control (KumarRobotics)
* Clone the repository ```kr_mav_control.git``` in the ```~/catkin_ws/src``` directory: 

```
$ cd ~/catkin_ws/src/
$ git clone git@github.com:KumarRobotics/kr_mav_control.git
```

* Next, change the line ```<library path="lib/libso3cmd_to_mavros_nodelet">``` to ```<library path="lib/libkr_mavros_interface">``` in the file ```/interfaces/kr_mavros_interface/nodelet_plugin.xml```. 
* Finally, run the following commands:

```
$ cd ~/catkin_ws
$ catkin config -DCMAKE_BUILD_TYPE=Release
$ catkin build
```

## Collect data using the PX4 simulator 
* Make sure the path ```px4_dir``` to your PX4 Firmware is updated in ```erl_quadrotor_control/scripts/launch-common.sh```
* Run ```data_collection.sh``` to collect data
* Run ```convert_data_from_rosbag.launch``` to convert the collected rosbag to a dataset in .npz format.
```
$ cd ~/catkin_ws/src/erl_quadrotor_control/scripts
$ chmod +x data_collection.sh
$ ./data_collection.sh 
$ roslaunch convert_data_from_rosbag.launch
```


https://github.com/thaipduong/LieGroupHamDL/assets/40247151/7c062b83-70a8-4516-9b15-982d5193af2b



## Training model:
* Run ```training/examples/quadrotor_px4/train_quadrotor_SE3_PX4.py``` with the .npz file above to train. A pre-collected dataset is provided.
```
python training/examples/quadrotor_px4/train_quadrotor_SE3_PX4.py
```
* Run ```training/examples/quadrotor_px4/save_torchscript_quadrotor_SE3_PX4.py``` to save the trained model to a torchscript file, which is used for our controller in C++.

## Trajectory tracking with the trained model in C++
* Update the trained model path in ```~\catkin_ws\src\erl_quadrotor_control\launch\sim_actuator.launch```. A pretrained model is provided in the package.
* Run ```trajectory_tracking.sh``` to test our controller with the learned model:
```angular2html
$ cd ~/catkin_ws/src/erl_quadrotor_control/scripts
$ chmod +x trajectory_tracking.sh
$ ./trajectory_tracking.sh
```


https://github.com/thaipduong/LieGroupHamDL/assets/40247151/25a5788c-4711-4e80-954b-a301120bacd3


## Citation
If you find our papers/code useful for your research, please cite our work as follows.

1. T. Duong, N. Atanasov. [Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control](https://thaipduong.github.io/SE3HamDL/). RSS, 2021

 ```bibtex
@inproceedings{duong21hamiltonian,
author = {Thai Duong AND Nikolay Atanasov},
title = {{Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control}},
booktitle = {Proceedings of Robotics: Science and Systems},
year = {2021},
address = {Virtual},
month = {July},
DOI = {10.15607/RSS.2021.XVII.086} 
}
```
