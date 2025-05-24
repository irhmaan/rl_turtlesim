import gymnasium as gym
from gymnasium import spaces
from enum import IntEnum
import numpy as np
import rclpy
from rclpy.node import Node
from turtlesim.srv import Spawn, Kill
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
import time


class Direction(IntEnum):
    UP = 0       # 0 radians
    RIGHT = 1    # -π/2
    DOWN = 2     # π
    LEFT = 3     # π/2

DIRECTION_TO_ANGLE = {
    Direction.UP: 0.0,
    Direction.RIGHT: -np.pi / 2,
    Direction.DOWN: np.pi,
    Direction.LEFT: np.pi / 2,
}


class TurtleRLNode(Node):
    def __init__(self):
        super().__init__('turtle_rl_node')

        self.pose_sub = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)

        self.spawn_client = self.create_client(Spawn, '/spawn')
        self.kill_client = self.create_client(Kill, '/kill')

        self.turtle1_pose = None
        self.enemy_pose = None
        self.enemy_name = 'enemy_turtle'

        self.enemy_pose_sub = None
        self.counter = 0
        self.turtles_caught = 0

    def pose_callback(self, msg):
        self.turtle1_pose = msg

    def enemy_pose_callback(self, msg):
        self.enemy_pose = msg

    def spawn_enemy(self):
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for spawn service...')
        request = Spawn.Request()
        request.x = float(np.random.uniform(1.0, 10.0))
        request.y = float(np.random.uniform(1.0, 10.0))
        request.theta = 0.0
        self.counter += 1
        request.name = self.enemy_name
        self.spawn_client.call_async(request)

        # Subscribe to enemy pose
        if self.enemy_pose_sub is None:
            self.enemy_pose_sub = self.create_subscription(Pose, f'/{self.enemy_name}/pose', self.enemy_pose_callback, 10)

    def kill_enemy(self, log=False):
        while not self.kill_client.wait_for_service(timeout_sec=1.0):
            if log:
                self.get_logger().info('Waiting for kill service...')

        request = Kill.Request()
        request.name = self.enemy_name

        future = self.kill_client.call_async(request)

        if log:
            def callback(future):
                try:
                    response = future.result()
                    self.get_logger().info(f'Successfully killed turtle: {request.name}')
                except Exception as e:
                    self.get_logger().error(f'Failed to kill turtle {request.name}: {str(e)}')

            future.add_done_callback(callback)


class TurtleEnv(gym.Env):
    def __init__(self):
        rclpy.init()
        self.node = TurtleRLNode()
        self.previous_obs = None
        # Discrete enum-based action space: UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(4)

        # Observation: [x1, y1, x_enemy, y_enemy]
        self.observation_space = spaces.Box(low=0.0, high=11.0, shape=(4,), dtype=np.float32)

        self.done = False
        self.turtles_caught = 0

    def step(self, action):
            rclpy.spin_once(self.node)

            if self.node.turtle1_pose is None or self.node.enemy_pose is None:
                return self._get_obs(), 0.0, False, False, {}

            desired_angle = DIRECTION_TO_ANGLE[Direction(action)]
            current_angle = self.node.turtle1_pose.theta
            angle_diff = (desired_angle - current_angle + np.pi) % (2 * np.pi) - np.pi

            twist_msg = Twist()
            twist_msg.angular.z = float(np.clip(6.0 * angle_diff, -2.0, 2.0))
            twist_msg.linear.x = 1.5 if abs(angle_diff) < 0.1 else 0.0


            if action == Direction.UP:
                twist_msg.linear.x = 1.0
                twist_msg.angular.z = 0.0
            elif action == Direction.DOWN:
                twist_msg.linear.x = -1.0
                twist_msg.angular.z = 0.0
            elif action == Direction.LEFT:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 1.5
            elif action == Direction.RIGHT:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = -1.5

            self.node.cmd_pub.publish(twist_msg)
            time.sleep(0.1)
            rclpy.spin_once(self.node)

            obs = self._get_obs()
            dist = np.linalg.norm(obs[0:2] - obs[2:4])
            reward = -0.01
            terminated = False

            # Reward for getting closer
            prev_dist = np.linalg.norm(self.previous_obs[0:2] - self.previous_obs[2:4]) if hasattr(self, 'previous_obs') and self.previous_obs is not None else dist
            reward += float((prev_dist - dist) * 0.3 ) # Adjust the scaling factor as needed

            if dist < 0.5:
                reward += 10.0
                self.turtles_caught += 1
                self._reset_enemy()
            elif obs[0] < 0.5 or  obs[1] < 0.5 :
                reward -= 1.0  # Negative reward for going out of bounds
            elif obs[0] > 10.5 or obs[1] > 10.5:
                reward -= 3.5 # more aggresive reward for going near to the boundary

            self.previous_obs = obs
            return obs, reward,  {"turtles_caught": self.turtles_caught}

    def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.turtles_caught = 0
            self._reset_enemy()
            while self.node.turtle1_pose is None or self.node.enemy_pose is None:
                rclpy.spin_once(self.node)
            initial_obs = self._get_obs()
            self.previous_obs = initial_obs
            return initial_obs, {}

    def _reset_enemy(self):
        try:
            self.node.kill_enemy()
            time.sleep(0.2)
        except Exception:
            pass
        self.node.spawn_enemy()
        time.sleep(0.2)
        while self.node.enemy_pose is None:
            rclpy.spin_once(self.node)

    def _get_obs(self):
        t = self.node.turtle1_pose
        e = self.node.enemy_pose
        return np.array([t.x, t.y, e.x, e.y], dtype=np.float32)

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
