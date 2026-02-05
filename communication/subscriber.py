import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseArray, Twist
from std_msgs.msg import String, Float64MultiArray, Int32MultiArray
from sensor_msgs.msg import JointState
import numpy as np
from collections import deque
from .config import NAMESPACE, QOS_RELIABLE, TOPIC_POSE, TOPIC_COMMAND, TOPIC_STATE, MAX_QUEUE_SIZE


class BaseSubscriber(Node):
    
    def __init__(self, node_name, topic_name, msg_type, callback=None, qos_depth=QOS_RELIABLE):
        super().__init__(node_name, namespace=NAMESPACE)
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth
        )
        self._user_callback = callback
        self.subscription = self.create_subscription(
            msg_type, topic_name, self._internal_callback, self.qos
        )
        self.last_msg = None
        self.msg_buffer = deque(maxlen=MAX_QUEUE_SIZE)
    
    def _internal_callback(self, msg):
        self.last_msg = msg
        self.msg_buffer.append(msg)
        if self._user_callback:
            self._user_callback(msg)
    
    def get_last_message(self):
        return self.last_msg
    
    def get_buffered_messages(self, count=None):
        if count is None:
            return list(self.msg_buffer)
        return list(self.msg_buffer)[-count:]
    
    def clear_buffer(self):
        self.msg_buffer.clear()


class PoseSubscriber(BaseSubscriber):
    
    def __init__(self, module_id=0, callback=None):
        topic = f'{TOPIC_POSE}_{module_id}'
        super().__init__(f'pose_sub_{module_id}', topic, PoseStamped, callback)
        self.module_id = module_id
    
    def get_pose(self):
        if self.last_msg is None:
            return None, None
        pos = self.last_msg.pose.position
        orn = self.last_msg.pose.orientation
        return (
            [pos.x, pos.y, pos.z],
            [orn.x, orn.y, orn.z, orn.w]
        )
    
    def get_position(self):
        pose = self.get_pose()
        return pose[0] if pose[0] else None
    
    def get_orientation(self):
        pose = self.get_pose()
        return pose[1] if pose[1] else None


class MultiPoseSubscriber(BaseSubscriber):
    
    def __init__(self, callback=None):
        super().__init__('multi_pose_sub', 'poses', PoseArray, callback)
        self.poses = {}
    
    def _internal_callback(self, msg):
        super()._internal_callback(msg)
        self.poses.clear()
        for i, pose in enumerate(msg.poses):
            self.poses[i] = {
                'position': [pose.position.x, pose.position.y, pose.position.z],
                'orientation': [pose.orientation.x, pose.orientation.y,
                              pose.orientation.z, pose.orientation.w]
            }
    
    def get_all_poses(self):
        return self.poses.copy()
    
    def get_pose(self, module_id):
        return self.poses.get(module_id)


class CommandSubscriber(BaseSubscriber):
    
    def __init__(self, module_id=0, callback=None):
        topic = f'{TOPIC_COMMAND}_{module_id}'
        super().__init__(f'cmd_sub_{module_id}', topic, Float64MultiArray, callback)
        self.module_id = module_id
    
    def get_command(self):
        if self.last_msg is None:
            return None
        return list(self.last_msg.data)
    
    def get_velocity_command(self):
        cmd = self.get_command()
        if cmd is None or len(cmd) < 6:
            return None, None
        return cmd[:3], cmd[3:6]
    
    def get_force_command(self):
        cmd = self.get_command()
        if cmd is None or len(cmd) < 6:
            return None, None
        return cmd[:3], cmd[3:6]


class StateSubscriber(BaseSubscriber):
    
    def __init__(self, module_id=0, callback=None):
        topic = f'{TOPIC_STATE}_{module_id}'
        super().__init__(f'state_sub_{module_id}', topic, Float64MultiArray, callback)
        self.module_id = module_id
    
    def get_state(self):
        if self.last_msg is None:
            return None
        data = list(self.last_msg.data)
        if len(data) < 13:
            return None
        return {
            'position': data[0:3],
            'orientation': data[3:7],
            'linear_velocity': data[7:10],
            'angular_velocity': data[10:13]
        }
    
    def get_position(self):
        state = self.get_state()
        return state['position'] if state else None
    
    def get_velocity(self):
        state = self.get_state()
        if state is None:
            return None, None
        return state['linear_velocity'], state['angular_velocity']


class JointStateSubscriber(BaseSubscriber):
    
    def __init__(self, callback=None):
        super().__init__('joint_state_sub', 'joint_states', JointState, callback)
    
    def get_joint_positions(self):
        if self.last_msg is None:
            return None
        return dict(zip(self.last_msg.name, self.last_msg.position))
    
    def get_joint_velocities(self):
        if self.last_msg is None or not self.last_msg.velocity:
            return None
        return dict(zip(self.last_msg.name, self.last_msg.velocity))
    
    def get_joint_efforts(self):
        if self.last_msg is None or not self.last_msg.effort:
            return None
        return dict(zip(self.last_msg.name, self.last_msg.effort))


class ConnectionSubscriber(BaseSubscriber):
    
    def __init__(self, callback=None):
        super().__init__('connection_sub', 'connections', Int32MultiArray, callback)
    
    def get_connections(self):
        if self.last_msg is None:
            return []
        data = self.last_msg.data
        pairs = []
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                pairs.append((data[i], data[i + 1]))
        return pairs
