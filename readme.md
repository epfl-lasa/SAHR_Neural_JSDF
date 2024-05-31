

### Installation
- `pip install faiss-gpu`
- python=3.9




## Using ROS in a conda env
- `conda config --env --add channels conda-forge
# and the robostack channel, https://robostack.github.io/GettingStarted.html
conda config --env --add channels robostack-staging`
- `conda install ros-noetic-desktop`


## Bringup the robots
- Position control `roslaunch iiwa_driver iiwa_bringup.launch model:=7 controller:=PositionController`
- Torque control `roslaunch iiwa_driver iiwa_bringup.launch model:=7 controller:=TorqueController`
