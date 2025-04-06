## Installation
please follow the instruction of the original tutorial for the docker image build and catkin package creation. Then build this package
```bash
python -m pip install scikit-robot
cd ~/catkin_ws/src
git clone https://github.com/HiroIshida/mycobot_teleop
cd mycobot_teleop
catkin bt
```

## Usage
To launch the simulator
```bash
roslaunch mycobot_teleop demo.launch
```

To run the teleop node (in a new terminal)
```bash
rosrun mycobot_teleop leader_teleop.py
```
