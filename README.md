# Turtlesim Reinforcement Learning Environment

This project is a custom OpenAI Gymnasium environment built using ROS 2 and the Turtlesim simulator. The agent (turtle1) learns to catch randomly spawned turtles in the simulated environment.

## Features

- Gymnasium-compatible environment (`TurtleEnv`)
- ROS 2-based communication with Turtlesim
- Reward structure:
  - +10 for catching a turtle
  - -5 for hitting the boundary
  - -0.01 time penalty per step
- Turtle is respawned after each catch
- Episode ends upon catching or failure

## Requirements

- ROS 2 (e.g. Jazzy, Humble, etc.)
- Python 3.10+
- [Turtlesim](https://docs.ros.org/en/ros2_packages/galactic/api/turtlesim.html)
- Gymnasium

Install Python dependencies:

```bash
pip install -r requirements.txt
```

# ⚠️ Ensure that ROS 2 and Turtlesim are installed and sourced before running.

# How to Run
Make sure ROS 2 is sourced:
```
source /opt/ros/jazzy/setup.bash
```

#Launch Turtlesim:
```
ros2 run turtlesim turtlesim_node
```
