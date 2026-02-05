import pybullet as p
import numpy as np
from .config import (
    MODULE_MASS, MODULE_SIZE, MODULE_COLOR,
    JOINT_MAX_FORCE, JOINT_MAX_VELOCITY, JOINT_DAMPING, JOINT_FRICTION,
    DEFAULT_POSITION, DEFAULT_ORIENTATION, CONNECTION_DISTANCE
)


class MSRRModule:
    
    def __init__(self, physics_client, module_id, position=None, orientation=None):
        self.physics_client = physics_client
        self.module_id = module_id
        self.position = position if position else DEFAULT_POSITION.copy()
        self.orientation = orientation if orientation else DEFAULT_ORIENTATION.copy()
        self.body_id = None
        self.connected_modules = []
        self.constraints = []
        self._create_module()
    
    def _create_module(self):
        col_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[s/2 for s in MODULE_SIZE],
            physicsClientId=self.physics_client
        )
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[s/2 for s in MODULE_SIZE],
            rgbaColor=MODULE_COLOR,
            physicsClientId=self.physics_client
        )
        self.body_id = p.createMultiBody(
            baseMass=MODULE_MASS,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=self.position,
            baseOrientation=self.orientation,
            physicsClientId=self.physics_client
        )
        p.changeDynamics(
            self.body_id, -1,
            lateralFriction=JOINT_FRICTION,
            linearDamping=JOINT_DAMPING,
            angularDamping=JOINT_DAMPING,
            physicsClientId=self.physics_client
        )
    
    def get_pose(self):
        pos, orn = p.getBasePositionAndOrientation(
            self.body_id,
            physicsClientId=self.physics_client
        )
        return np.array(pos), np.array(orn)
    
    def set_pose(self, position, orientation):
        p.resetBasePositionAndOrientation(
            self.body_id,
            position,
            orientation,
            physicsClientId=self.physics_client
        )
        self.position = list(position)
        self.orientation = list(orientation)
    
    def get_velocity(self):
        lin_vel, ang_vel = p.getBaseVelocity(
            self.body_id,
            physicsClientId=self.physics_client
        )
        return np.array(lin_vel), np.array(ang_vel)
    
    def apply_force(self, force, position=None):
        if position is None:
            position = self.position
        p.applyExternalForce(
            self.body_id, -1,
            force, position,
            p.WORLD_FRAME,
            physicsClientId=self.physics_client
        )
    
    def apply_torque(self, torque):
        p.applyExternalTorque(
            self.body_id, -1,
            torque,
            p.WORLD_FRAME,
            physicsClientId=self.physics_client
        )
    
    def connect_to(self, other_module, joint_type=p.JOINT_FIXED):
        constraint_id = p.createConstraint(
            self.body_id, -1,
            other_module.body_id, -1,
            joint_type,
            [0, 0, 1],
            [MODULE_SIZE[0]/2, 0, 0],
            [-MODULE_SIZE[0]/2, 0, 0],
            physicsClientId=self.physics_client
        )
        p.changeConstraint(
            constraint_id,
            maxForce=JOINT_MAX_FORCE,
            physicsClientId=self.physics_client
        )
        self.constraints.append(constraint_id)
        self.connected_modules.append(other_module.module_id)
        other_module.connected_modules.append(self.module_id)
        return constraint_id
    
    def disconnect_from(self, other_module):
        for i, cid in enumerate(self.constraints):
            info = p.getConstraintInfo(cid, physicsClientId=self.physics_client)
            if info[2] == other_module.body_id:
                p.removeConstraint(cid, physicsClientId=self.physics_client)
                self.constraints.pop(i)
                self.connected_modules.remove(other_module.module_id)
                other_module.connected_modules.remove(self.module_id)
                return True
        return False
    
    def remove(self):
        for cid in self.constraints:
            p.removeConstraint(cid, physicsClientId=self.physics_client)
        p.removeBody(self.body_id, physicsClientId=self.physics_client)


class MSRRRobot:
    
    def __init__(self, physics_client):
        self.physics_client = physics_client
        self.modules = {}
        self.next_module_id = 0
    
    def add_module(self, position=None, orientation=None):
        module = MSRRModule(
            self.physics_client,
            self.next_module_id,
            position,
            orientation
        )
        self.modules[self.next_module_id] = module
        self.next_module_id += 1
        return module
    
    def remove_module(self, module_id):
        if module_id in self.modules:
            self.modules[module_id].remove()
            del self.modules[module_id]
            return True
        return False
    
    def get_module(self, module_id):
        return self.modules.get(module_id)
    
    def connect_modules(self, module_id_1, module_id_2, joint_type=p.JOINT_FIXED):
        m1 = self.get_module(module_id_1)
        m2 = self.get_module(module_id_2)
        if m1 and m2:
            return m1.connect_to(m2, joint_type)
        return None
    
    def disconnect_modules(self, module_id_1, module_id_2):
        m1 = self.get_module(module_id_1)
        m2 = self.get_module(module_id_2)
        if m1 and m2:
            return m1.disconnect_from(m2)
        return False
    
    def get_all_poses(self):
        poses = {}
        for mid, module in self.modules.items():
            pos, orn = module.get_pose()
            poses[mid] = {'position': pos, 'orientation': orn}
        return poses
    
    def get_center_of_mass(self):
        if not self.modules:
            return np.zeros(3)
        positions = [m.get_pose()[0] for m in self.modules.values()]
        return np.mean(positions, axis=0)
    
    def apply_force_to_all(self, force):
        for module in self.modules.values():
            module.apply_force(force)
    
    def reset(self):
        for module in list(self.modules.values()):
            module.remove()
        self.modules.clear()
        self.next_module_id = 0
