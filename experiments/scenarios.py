import numpy as np
from abc import ABC, abstractmethod
from .config import (
    DEFAULT_EXPERIMENT_DURATION, DEFAULT_NUM_MODULES,
    METRICS_POSITION_ERROR, METRICS_ORIENTATION_ERROR, METRICS_COMPLETION_TIME
)


class ScenarioBase(ABC):
    
    def __init__(self, name='base_scenario'):
        self.name = name
        self.parameters = {}
        self.initial_state = {}
        self.target_state = {}
        self.current_state = {}
        self.is_setup = False
        self._complete = False
    
    @abstractmethod
    def setup(self, simulation_env):
        pass
    
    @abstractmethod
    def step(self, simulation_env):
        pass
    
    @abstractmethod
    def get_state(self, simulation_env):
        pass
    
    def set_parameters(self, params):
        self.parameters.update(params)
    
    def get_parameters(self):
        return self.parameters.copy()
    
    def set_target_state(self, state):
        self.target_state = state.copy()
    
    def get_target_state(self):
        return self.target_state.copy()
    
    def is_complete(self):
        return self._complete
    
    def get_metrics(self):
        return [METRICS_POSITION_ERROR, METRICS_COMPLETION_TIME]
    
    def reset(self):
        self.current_state = {}
        self.is_setup = False
        self._complete = False


class ReconfigurationScenario(ScenarioBase):
    
    def __init__(self):
        super().__init__('reconfiguration')
        self.parameters = {
            'num_modules': DEFAULT_NUM_MODULES,
            'initial_config': 'line',
            'target_config': 'square',
            'tolerance': 0.01
        }
        self.modules = []
        self.connections = []
        self.target_positions = []
    
    def setup(self, simulation_env):
        if simulation_env is None:
            self.is_setup = True
            return
        
        num_modules = self.parameters.get('num_modules', DEFAULT_NUM_MODULES)
        initial_config = self.parameters.get('initial_config', 'line')
        
        self._create_initial_config(simulation_env, num_modules, initial_config)
        self._compute_target_positions()
        
        self.target_state = {
            'positions': self.target_positions,
            'configuration': self.parameters.get('target_config', 'square')
        }
        
        self.is_setup = True
    
    def _create_initial_config(self, sim_env, num_modules, config_type):
        self.modules = []
        spacing = 0.06
        
        if config_type == 'line':
            for i in range(num_modules):
                pos = [i * spacing, 0, 0.1]
                module = sim_env.robot.add_module(position=pos)
                self.modules.append(module)
                if i > 0:
                    sim_env.robot.connect_modules(i - 1, i)
        
        elif config_type == 'square':
            side = int(np.ceil(np.sqrt(num_modules)))
            for i in range(num_modules):
                row = i // side
                col = i % side
                pos = [col * spacing, row * spacing, 0.1]
                module = sim_env.robot.add_module(position=pos)
                self.modules.append(module)
    
    def _compute_target_positions(self):
        num_modules = self.parameters.get('num_modules', DEFAULT_NUM_MODULES)
        target_config = self.parameters.get('target_config', 'square')
        spacing = 0.06
        
        self.target_positions = []
        
        if target_config == 'line':
            for i in range(num_modules):
                self.target_positions.append([i * spacing, 0, 0.1])
        
        elif target_config == 'square':
            side = int(np.ceil(np.sqrt(num_modules)))
            for i in range(num_modules):
                row = i // side
                col = i % side
                self.target_positions.append([col * spacing, row * spacing, 0.1])
        
        elif target_config == 'snake':
            for i in range(num_modules):
                x = (i % 4) * spacing
                y = (i // 4) * spacing
                if (i // 4) % 2 == 1:
                    x = 3 * spacing - x
                self.target_positions.append([x, y, 0.1])
    
    def step(self, simulation_env):
        if not self.is_setup:
            return
        
        tolerance = self.parameters.get('tolerance', 0.01)
        all_reached = True
        
        for i, module in enumerate(self.modules):
            if i >= len(self.target_positions):
                break
            
            current_pos, _ = module.get_pose()
            target_pos = np.array(self.target_positions[i])
            error = np.linalg.norm(current_pos - target_pos)
            
            if error > tolerance:
                all_reached = False
                direction = (target_pos - current_pos) / (error + 1e-6)
                force = direction * 10.0
                module.apply_force(force.tolist())
        
        self._complete = all_reached
    
    def get_state(self, simulation_env):
        if simulation_env is None:
            return {}
        
        state = {'modules': {}}
        poses = simulation_env.robot.get_all_poses()
        
        for mid, pose in poses.items():
            state['modules'][mid] = {
                'position': pose['position'].tolist(),
                'orientation': pose['orientation'].tolist()
            }
        
        if poses:
            first_pose = list(poses.values())[0]
            state['position'] = first_pose['position'].tolist()
            state['orientation'] = first_pose['orientation'].tolist()
        
        return state
    
    def get_metrics(self):
        return [METRICS_POSITION_ERROR, METRICS_COMPLETION_TIME]


class LocomotionScenario(ScenarioBase):
    
    def __init__(self):
        super().__init__('locomotion')
        self.parameters = {
            'num_modules': DEFAULT_NUM_MODULES,
            'gait': 'wave',
            'target_distance': 1.0,
            'speed': 0.1
        }
        self.modules = []
        self.start_position = None
        self.gait_phase = 0.0
    
    def setup(self, simulation_env):
        if simulation_env is None:
            self.is_setup = True
            return
        
        num_modules = self.parameters.get('num_modules', DEFAULT_NUM_MODULES)
        spacing = 0.06
        
        self.modules = []
        for i in range(num_modules):
            pos = [i * spacing, 0, 0.1]
            module = simulation_env.robot.add_module(position=pos)
            self.modules.append(module)
            if i > 0:
                simulation_env.robot.connect_modules(i - 1, i)
        
        self.start_position = simulation_env.robot.get_center_of_mass()
        
        target_distance = self.parameters.get('target_distance', 1.0)
        self.target_state = {
            'position': [self.start_position[0] + target_distance, 
                        self.start_position[1], 
                        self.start_position[2]],
            'distance': target_distance
        }
        
        self.gait_phase = 0.0
        self.is_setup = True
    
    def step(self, simulation_env):
        if not self.is_setup or simulation_env is None:
            return
        
        gait = self.parameters.get('gait', 'wave')
        speed = self.parameters.get('speed', 0.1)
        target_distance = self.parameters.get('target_distance', 1.0)
        
        current_com = simulation_env.robot.get_center_of_mass()
        traveled = np.linalg.norm(current_com[:2] - self.start_position[:2])
        
        if traveled >= target_distance:
            self._complete = True
            return
        
        self.gait_phase += 0.05
        
        if gait == 'wave':
            self._wave_gait(speed)
        elif gait == 'caterpillar':
            self._caterpillar_gait(speed)
    
    def _wave_gait(self, speed):
        for i, module in enumerate(self.modules):
            phase = self.gait_phase + i * 0.5
            force_x = speed * np.cos(phase) * 5.0
            force_z = np.sin(phase) * 2.0
            module.apply_force([force_x, 0, force_z])
    
    def _caterpillar_gait(self, speed):
        for i, module in enumerate(self.modules):
            phase = self.gait_phase + i * 0.3
            contract = np.sin(phase) > 0
            if contract:
                force_x = speed * 10.0
            else:
                force_x = 0
            module.apply_force([force_x, 0, 0])
    
    def get_state(self, simulation_env):
        if simulation_env is None:
            return {}
        
        com = simulation_env.robot.get_center_of_mass()
        
        state = {
            'position': com.tolist(),
            'orientation': [0, 0, 0, 1],
            'distance_traveled': np.linalg.norm(com[:2] - self.start_position[:2]) if self.start_position is not None else 0,
            'gait_phase': self.gait_phase
        }
        
        return state
    
    def get_metrics(self):
        return [METRICS_POSITION_ERROR, METRICS_COMPLETION_TIME, 'distance_traveled']
