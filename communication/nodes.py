import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist, Point, Quaternion
from std_msgs.msg import String, Float64MultiArray, Int32
import numpy as np
from .config import (
    NAMESPACE, QOS_RELIABLE, QOS_SENSOR,
    TOPIC_POSE, TOPIC_COMMAND, TOPIC_STATE, TOPIC_VELOCITY,
    PUBLISH_RATE_POSE, PUBLISH_RATE_STATE
)


class MSRRNode(Node):
    
    def __init__(self, node_name, module_id=0):
        super().__init__(node_name, namespace=NAMESPACE)
        self.module_id = module_id
        self.qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=QOS_RELIABLE
        )
        self.qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=QOS_SENSOR
        )
        self.publishers_ = {}
        self.subscribers_ = {}
        self.timers_ = {}
        self.services_ = {}
        self.clients_ = {}
    
    def create_pose_publisher(self, topic_suffix=''):
        topic = f'{TOPIC_POSE}_{self.module_id}{topic_suffix}'
        pub = self.create_publisher(PoseStamped, topic, self.qos_reliable)
        self.publishers_[topic] = pub
        return pub
    
    def create_velocity_publisher(self, topic_suffix=''):
        topic = f'{TOPIC_VELOCITY}_{self.module_id}{topic_suffix}'
        pub = self.create_publisher(Twist, topic, self.qos_reliable)
        self.publishers_[topic] = pub
        return pub
    
    def create_command_publisher(self, topic_suffix=''):
        topic = f'{TOPIC_COMMAND}_{self.module_id}{topic_suffix}'
        pub = self.create_publisher(Float64MultiArray, topic, self.qos_reliable)
        self.publishers_[topic] = pub
        return pub
    
    def create_pose_subscriber(self, callback, module_id=None, topic_suffix=''):
        mid = module_id if module_id is not None else self.module_id
        topic = f'{TOPIC_POSE}_{mid}{topic_suffix}'
        sub = self.create_subscription(PoseStamped, topic, callback, self.qos_reliable)
        self.subscribers_[topic] = sub
        return sub
    
    def create_velocity_subscriber(self, callback, module_id=None, topic_suffix=''):
        mid = module_id if module_id is not None else self.module_id
        topic = f'{TOPIC_VELOCITY}_{mid}{topic_suffix}'
        sub = self.create_subscription(Twist, topic, callback, self.qos_reliable)
        self.subscribers_[topic] = sub
        return sub
    
    def create_command_subscriber(self, callback, module_id=None, topic_suffix=''):
        mid = module_id if module_id is not None else self.module_id
        topic = f'{TOPIC_COMMAND}_{mid}{topic_suffix}'
        sub = self.create_subscription(Float64MultiArray, topic, callback, self.qos_reliable)
        self.subscribers_[topic] = sub
        return sub
    
    def publish_pose(self, position, orientation, topic_suffix=''):
        topic = f'{TOPIC_POSE}_{self.module_id}{topic_suffix}'
        if topic in self.publishers_:
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = f'module_{self.module_id}'
            msg.pose.position.x = float(position[0])
            msg.pose.position.y = float(position[1])
            msg.pose.position.z = float(position[2])
            msg.pose.orientation.x = float(orientation[0])
            msg.pose.orientation.y = float(orientation[1])
            msg.pose.orientation.z = float(orientation[2])
            msg.pose.orientation.w = float(orientation[3])
            self.publishers_[topic].publish(msg)
    
    def publish_velocity(self, linear, angular, topic_suffix=''):
        topic = f'{TOPIC_VELOCITY}_{self.module_id}{topic_suffix}'
        if topic in self.publishers_:
            msg = Twist()
            msg.linear.x = float(linear[0])
            msg.linear.y = float(linear[1])
            msg.linear.z = float(linear[2])
            msg.angular.x = float(angular[0])
            msg.angular.y = float(angular[1])
            msg.angular.z = float(angular[2])
            self.publishers_[topic].publish(msg)
    
    def publish_command(self, command_data, topic_suffix=''):
        topic = f'{TOPIC_COMMAND}_{self.module_id}{topic_suffix}'
        if topic in self.publishers_:
            msg = Float64MultiArray()
            msg.data = [float(x) for x in command_data]
            self.publishers_[topic].publish(msg)


class ControllerNode(MSRRNode):
    
    def __init__(self, module_id=0):
        super().__init__('controller_node', module_id)
        self.target_pose = None
        self.current_pose = None
        self.control_gains = {'kp': 1.0, 'ki': 0.0, 'kd': 0.1}
        self.error_integral = np.zeros(6)
        self.last_error = np.zeros(6)
        
        self.cmd_pub = self.create_command_publisher()
        self.pose_sub = self.create_pose_subscriber(self._pose_callback)
        self.target_sub = self.create_subscription(
            PoseStamped, f'target_{self.module_id}',
            self._target_callback, self.qos_reliable
        )
        
        self.control_timer = self.create_timer(
            1.0 / PUBLISH_RATE_STATE, self._control_loop
        )
    
    def _pose_callback(self, msg):
        self.current_pose = {
            'position': [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            'orientation': [msg.pose.orientation.x, msg.pose.orientation.y,
                          msg.pose.orientation.z, msg.pose.orientation.w]
        }
    
    def _target_callback(self, msg):
        self.target_pose = {
            'position': [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            'orientation': [msg.pose.orientation.x, msg.pose.orientation.y,
                          msg.pose.orientation.z, msg.pose.orientation.w]
        }
    
    def _control_loop(self):
        if self.current_pose is None or self.target_pose is None:
            return
        
        error = np.array(self.target_pose['position']) - np.array(self.current_pose['position'])
        error = np.concatenate([error, np.zeros(3)])
        
        self.error_integral += error
        error_derivative = error - self.last_error
        self.last_error = error.copy()
        
        control = (self.control_gains['kp'] * error +
                   self.control_gains['ki'] * self.error_integral +
                   self.control_gains['kd'] * error_derivative)
        
        self.publish_command(control.tolist())
    
    def set_gains(self, kp, ki, kd):
        self.control_gains = {'kp': kp, 'ki': ki, 'kd': kd}
    
    def reset_controller(self):
        self.error_integral = np.zeros(6)
        self.last_error = np.zeros(6)


class PerceptionNode(MSRRNode):
    
    def __init__(self, module_id=0):
        super().__init__('perception_node', module_id)
        self.detected_poses = {}
        self.pose_pub = self.create_pose_publisher()
        self.publish_timer = self.create_timer(
            1.0 / PUBLISH_RATE_POSE, self._publish_loop
        )
    
    def update_pose(self, position, orientation):
        self.detected_poses[self.module_id] = {
            'position': position,
            'orientation': orientation
        }
    
    def _publish_loop(self):
        if self.module_id in self.detected_poses:
            pose = self.detected_poses[self.module_id]
            self.publish_pose(pose['position'], pose['orientation'])
    
    def get_all_poses(self):
        return self.detected_poses.copy()
