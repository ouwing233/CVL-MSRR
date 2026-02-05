import pybullet as p
import pybullet_data
import numpy as np
import time
from .config import (
    GRAVITY, TIME_STEP, SIMULATION_MODE,
    GROUND_PLANE_HEIGHT, GROUND_FRICTION, GROUND_RESTITUTION,
    CAMERA_DISTANCE, CAMERA_YAW, CAMERA_PITCH, CAMERA_TARGET
)
from .robot import MSRRRobot


class SimulationEnvironment:
    
    def __init__(self, mode=None, enable_rendering=True):
        self.mode = mode if mode else SIMULATION_MODE
        self.enable_rendering = enable_rendering
        self.physics_client = None
        self.ground_id = None
        self.robot = None
        self.obstacles = []
        self.time_elapsed = 0.0
        self._initialize()
    
    def _initialize(self):
        if self.mode == "GUI" and self.enable_rendering:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, GRAVITY, physicsClientId=self.physics_client)
        p.setTimeStep(TIME_STEP, physicsClientId=self.physics_client)
        
        self.ground_id = p.loadURDF(
            "plane.urdf",
            [0, 0, GROUND_PLANE_HEIGHT],
            physicsClientId=self.physics_client
        )
        p.changeDynamics(
            self.ground_id, -1,
            lateralFriction=GROUND_FRICTION,
            restitution=GROUND_RESTITUTION,
            physicsClientId=self.physics_client
        )
        
        if self.mode == "GUI" and self.enable_rendering:
            p.resetDebugVisualizerCamera(
                CAMERA_DISTANCE,
                CAMERA_YAW,
                CAMERA_PITCH,
                CAMERA_TARGET,
                physicsClientId=self.physics_client
            )
        
        self.robot = MSRRRobot(self.physics_client)
    
    def step(self):
        p.stepSimulation(physicsClientId=self.physics_client)
        self.time_elapsed += TIME_STEP
    
    def step_seconds(self, seconds):
        steps = int(seconds / TIME_STEP)
        for _ in range(steps):
            self.step()
    
    def get_time(self):
        return self.time_elapsed
    
    def add_obstacle(self, shape_type, position, size, mass=0, color=None):
        if color is None:
            color = [0.5, 0.5, 0.5, 1.0]
        
        if shape_type == "box":
            col_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[s/2 for s in size],
                physicsClientId=self.physics_client
            )
            vis_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[s/2 for s in size],
                rgbaColor=color,
                physicsClientId=self.physics_client
            )
        elif shape_type == "sphere":
            col_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=size[0],
                physicsClientId=self.physics_client
            )
            vis_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=size[0],
                rgbaColor=color,
                physicsClientId=self.physics_client
            )
        elif shape_type == "cylinder":
            col_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=size[0],
                height=size[1],
                physicsClientId=self.physics_client
            )
            vis_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=size[0],
                length=size[1],
                rgbaColor=color,
                physicsClientId=self.physics_client
            )
        else:
            return None
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position,
            physicsClientId=self.physics_client
        )
        self.obstacles.append(body_id)
        return body_id
    
    def remove_obstacle(self, obstacle_id):
        if obstacle_id in self.obstacles:
            p.removeBody(obstacle_id, physicsClientId=self.physics_client)
            self.obstacles.remove(obstacle_id)
            return True
        return False
    
    def get_camera_image(self, width=640, height=480, view_matrix=None, projection_matrix=None):
        if view_matrix is None:
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                CAMERA_TARGET,
                CAMERA_DISTANCE,
                CAMERA_YAW,
                CAMERA_PITCH,
                0,
                2,
                physicsClientId=self.physics_client
            )
        if projection_matrix is None:
            projection_matrix = p.computeProjectionMatrixFOV(
                60, width/height, 0.1, 100,
                physicsClientId=self.physics_client
            )
        
        _, _, rgb, depth, seg = p.getCameraImage(
            width, height,
            view_matrix, projection_matrix,
            physicsClientId=self.physics_client
        )
        return np.array(rgb), np.array(depth), np.array(seg)
    
    def set_camera(self, distance, yaw, pitch, target):
        if self.mode == "GUI":
            p.resetDebugVisualizerCamera(
                distance, yaw, pitch, target,
                physicsClientId=self.physics_client
            )
    
    def enable_real_time(self, enable=True):
        p.setRealTimeSimulation(enable, physicsClientId=self.physics_client)
    
    def get_contact_points(self, body_a=None, body_b=None):
        if body_a is not None and body_b is not None:
            return p.getContactPoints(body_a, body_b, physicsClientId=self.physics_client)
        elif body_a is not None:
            return p.getContactPoints(body_a, physicsClientId=self.physics_client)
        return p.getContactPoints(physicsClientId=self.physics_client)
    
    def save_state(self):
        return p.saveState(physicsClientId=self.physics_client)
    
    def restore_state(self, state_id):
        p.restoreState(state_id, physicsClientId=self.physics_client)
    
    def reset(self):
        self.robot.reset()
        for obs_id in self.obstacles:
            p.removeBody(obs_id, physicsClientId=self.physics_client)
        self.obstacles.clear()
        self.time_elapsed = 0.0
    
    def close(self):
        p.disconnect(physicsClientId=self.physics_client)
