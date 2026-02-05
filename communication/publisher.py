import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseArray, Twist, TwistStamped
from std_msgs.msg import String, Float64MultiArray, Int32MultiArray, Header
from sensor_msgs.msg import JointState
import numpy as np
from .config import NAMESPACE, QOS_RELIABLE, TOPIC_POSE, TOPIC_COMMAND, TOPIC_STATE


class BasePublisher(Node):
    
    def __init__(self, node_name, topic_name, msg_type, qos_depth=QOS_RELIABLE):
        super().__init__(node_name, namespace=NAMESPACE)
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth
        )
        self.publisher_ = self.create_publisher(msg_type, topic_name, self.qos)
        self.msg_type = msg_type
    
    def publish(self, msg):
        self.publisher_.publish(msg)
    
    def get_header(self, frame_id='world'):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        return header


class PosePublisher(BasePublisher):
    
    def __init__(self, module_id=0):
        topic = f'{TOPIC_POSE}_{module_id}'
        super().__init__(f'pose_pub_{module_id}', topic, PoseStamped)
        self.module_id = module_id
    
    def publish_pose(self, position, orientation, frame_id=None):
        msg = PoseStamped()
        msg.header = self.get_header(frame_id or f'module_{self.module_id}')
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(orientation[0])
        msg.pose.orientation.y = float(orientation[1])
        msg.pose.orientation.z = float(orientation[2])
        msg.pose.orientation.w = float(orientation[3])
        self.publish(msg)


class MultiPosePublisher(BasePublisher):
    
    def __init__(self):
        super().__init__('multi_pose_pub', 'poses', PoseArray)
    
    def publish_poses(self, poses_dict, frame_id='world'):
        msg = PoseArray()
        msg.header = self.get_header(frame_id)
        for module_id, pose_data in poses_dict.items():
            from geometry_msgs.msg import Pose
            pose = Pose()
            pose.position.x = float(pose_data['position'][0])
            pose.position.y = float(pose_data['position'][1])
            pose.position.z = float(pose_data['position'][2])
            pose.orientation.x = float(pose_data['orientation'][0])
            pose.orientation.y = float(pose_data['orientation'][1])
            pose.orientation.z = float(pose_data['orientation'][2])
            pose.orientation.w = float(pose_data['orientation'][3])
            msg.poses.append(pose)
        self.publish(msg)


class CommandPublisher(BasePublisher):
    
    def __init__(self, module_id=0):
        topic = f'{TOPIC_COMMAND}_{module_id}'
        super().__init__(f'cmd_pub_{module_id}', topic, Float64MultiArray)
        self.module_id = module_id
    
    def publish_command(self, command_data):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in command_data]
        self.publish(msg)
    
    def publish_velocity_command(self, linear_vel, angular_vel):
        command = list(linear_vel) + list(angular_vel)
        self.publish_command(command)
    
    def publish_force_command(self, force, torque):
        command = list(force) + list(torque)
        self.publish_command(command)


class StatePublisher(BasePublisher):
    
    def __init__(self, module_id=0):
        topic = f'{TOPIC_STATE}_{module_id}'
        super().__init__(f'state_pub_{module_id}', topic, Float64MultiArray)
        self.module_id = module_id
    
    def publish_state(self, position, orientation, linear_vel, angular_vel):
        state = (list(position) + list(orientation) + 
                 list(linear_vel) + list(angular_vel))
        msg = Float64MultiArray()
        msg.data = [float(x) for x in state]
        self.publish(msg)


class JointStatePublisher(BasePublisher):
    
    def __init__(self, robot_name='msrr'):
        super().__init__(f'{robot_name}_joint_pub', 'joint_states', JointState)
        self.robot_name = robot_name
    
    def publish_joint_states(self, joint_names, positions, velocities=None, efforts=None):
        msg = JointState()
        msg.header = self.get_header()
        msg.name = joint_names
        msg.position = [float(p) for p in positions]
        if velocities is not None:
            msg.velocity = [float(v) for v in velocities]
        if efforts is not None:
            msg.effort = [float(e) for e in efforts]
        self.publish(msg)


class ConnectionPublisher(BasePublisher):
    
    def __init__(self):
        super().__init__('connection_pub', 'connections', Int32MultiArray)
    
    def publish_connections(self, connection_pairs):
        msg = Int32MultiArray()
        data = []
        for pair in connection_pairs:
            data.extend([int(pair[0]), int(pair[1])])
        msg.data = data
        self.publish(msg)
