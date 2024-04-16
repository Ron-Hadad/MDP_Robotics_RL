# Q Learner for Turtlebot3 in Gazebo Simulation

## Overview
This package contains a Python script (`q_learner.py`) that implements a Q learning algorithm for controlling a Turtlebot3 Burger robot in a Gazebo simulation environment. The `q_table.csv` file contains a table 3133 X 7 that represent the current policy of the robot.

## Setup Instructions
1. **Installation**:
- Clone the `task4_env` folder into your `catkin_ws/` , build the code using `catkin_make`, and source the setup file (`source devel/setup.bash`).

3. **Launching the Environment**:
- Use the provided launch file to start the simulation environment:
   ```
    roslaunch task4_env task4_env.launch
   ```

## Usage
- Execute the Q learner script by running:
    ```
  rosrun task4_env q_learner.py <learning_mode>
    ```
- Replace `<learning_mode>` with `0` to execute the current policy(by `q_table.csv`) 10 times and print the average reward, or `1` to enable learning(defualt 1000 runs).

## Features
- Implements tabular Q learning to control the Turtlebot3 robot.
- Receives environment state and reward information from the `skills_server.py` script.
- Performs Q learning updates based on actions taken and rewards received.
- Uses epsilon-greedy policy for action selection during learning.


This project was developed by Guy Ginat (ID: 206922544) and Ron Hadad (ID: 209260645).
