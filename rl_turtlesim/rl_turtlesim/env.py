import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from turtlesim.srv import Spawn, Kill
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist, Vector3
import time


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

        # Normalized action space [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32))

        # Observation: [x1, y1, x_enemy, y_enemy]
        self.observation_space = spaces.Box(low=0.0, high=11.0, shape=(4,), dtype=np.float32)

        self.done = False
        # self.episode_steps = 0
        # self.max_steps = 200

    def step(self, action):
        rclpy.spin_once(self.node)
        obs = self._get_obs()
        turtle_x, turtle_y, enemy_x, enemy_y = obs

        # Calculate the vector to the enemy
        to_enemy = np.array([enemy_x - turtle_x, enemy_y - turtle_y], dtype=np.float32)
        distance_to_enemy = np.linalg.norm(to_enemy)

        # Determine the desired linear velocity vector
        linear_speed_factor = 0.5  # Tune this value
        if distance_to_enemy > 0.0:
            linear_velocity_x = linear_speed_factor * (to_enemy[0] / distance_to_enemy) # Normalize and scale
            linear_velocity_y = linear_speed_factor * (to_enemy[1] / distance_to_enemy) # Normalize and scale
        else:
            linear_velocity_x = 0.0
            linear_velocity_y = 0.0

        # Calculate the angle to the enemy for angular movement (optional, but good for alignment)
        angle_to_enemy = np.arctan2(enemy_y - turtle_y, enemy_x - turtle_x)
        # Assuming turtle's orientation is 0, adjust if needed
        angle_diff = angle_to_enemy
        angular_speed_factor = 1.0 # Tune this value
        angular = angular_speed_factor * angle_diff

        twist_msg = Twist()
        twist_msg.linear = Vector3(x=float(linear_velocity_x), y=float(linear_velocity_y), z=0.0)
        twist_msg.angular = Vector3(x=0.0, y=0.0, z=float(angular))
        self.node.cmd_pub.publish(twist_msg)

        rclpy.spin_once(self.node)
        obs = self._get_obs()

        dist = np.linalg.norm(obs[0:2] - obs[2:4])
        reward = -0.01  # Time penalty
        terminated = False

        if dist < 0.5:
            reward += 10.0
            self.turtles_caught += 1
            self._reset_enemy()
            terminated = True
        elif obs[0] < 0.5 or obs[0] > 10.5 or obs[1] < 0.5 or obs[1] > 10.5:
            reward -= 5.0  # Boundary penalty
            terminated = True

        return obs, reward, terminated, False, {"turtles_caught": self.turtles_caught}
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # self.episode_steps = 0
        self.turtles_caught = 0
        self.prev_obs = None # Initialize previous observation
        self._reset_enemy()
        while self.node.turtle1_pose is None or self.node.enemy_pose is None:
            rclpy.spin_once(self.node)
        return self._get_obs(), {}

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

 