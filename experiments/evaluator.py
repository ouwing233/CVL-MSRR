import numpy as np
from .config import (
    METRICS_POSITION_ERROR, METRICS_ORIENTATION_ERROR,
    METRICS_COMPLETION_TIME, METRICS_ENERGY_CONSUMPTION, METRICS_SUCCESS_RATE
)


class MetricsCalculator:
    
    @staticmethod
    def position_error(actual, target):
        actual = np.array(actual)
        target = np.array(target)
        return np.linalg.norm(actual - target)
    
    @staticmethod
    def orientation_error(actual, target):
        actual = np.array(actual)
        target = np.array(target)
        dot = np.abs(np.dot(actual, target))
        dot = np.clip(dot, -1.0, 1.0)
        return 2 * np.arccos(dot)
    
    @staticmethod
    def trajectory_error(actual_trajectory, target_trajectory):
        errors = []
        min_len = min(len(actual_trajectory), len(target_trajectory))
        for i in range(min_len):
            err = MetricsCalculator.position_error(actual_trajectory[i], target_trajectory[i])
            errors.append(err)
        return np.mean(errors) if errors else 0.0
    
    @staticmethod
    def smoothness(trajectory):
        if len(trajectory) < 3:
            return 0.0
        trajectory = np.array(trajectory)
        velocities = np.diff(trajectory, axis=0)
        accelerations = np.diff(velocities, axis=0)
        jerk = np.diff(accelerations, axis=0)
        return np.mean(np.linalg.norm(jerk, axis=1)) if len(jerk) > 0 else 0.0
    
    @staticmethod
    def path_length(trajectory):
        if len(trajectory) < 2:
            return 0.0
        trajectory = np.array(trajectory)
        diffs = np.diff(trajectory, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    @staticmethod
    def energy_consumption(forces, velocities, dt=0.01):
        forces = np.array(forces)
        velocities = np.array(velocities)
        power = np.sum(np.abs(forces * velocities), axis=1)
        return np.sum(power) * dt
    
    @staticmethod
    def settling_time(values, target, threshold=0.02):
        values = np.array(values)
        target = float(target)
        
        for i in range(len(values) - 1, -1, -1):
            if abs(values[i] - target) > threshold * abs(target):
                return i + 1
        return 0
    
    @staticmethod
    def overshoot(values, target):
        values = np.array(values)
        target = float(target)
        
        if target > values[0]:
            max_val = np.max(values)
            if max_val > target:
                return (max_val - target) / (target - values[0]) * 100
        else:
            min_val = np.min(values)
            if min_val < target:
                return (target - min_val) / (values[0] - target) * 100
        return 0.0
    
    @staticmethod
    def success_rate(results_list, threshold):
        successes = sum(1 for r in results_list if r < threshold)
        return successes / len(results_list) if results_list else 0.0


class Evaluator:
    
    def __init__(self):
        self.metrics = {}
        self.weights = {}
    
    def evaluate(self, recorded_data, target_state, metric_names=None):
        results = {}
        
        if metric_names is None:
            metric_names = [
                METRICS_POSITION_ERROR,
                METRICS_ORIENTATION_ERROR,
                METRICS_COMPLETION_TIME
            ]
        
        for metric in metric_names:
            if metric == METRICS_POSITION_ERROR:
                results[metric] = self._evaluate_position_error(recorded_data, target_state)
            elif metric == METRICS_ORIENTATION_ERROR:
                results[metric] = self._evaluate_orientation_error(recorded_data, target_state)
            elif metric == METRICS_COMPLETION_TIME:
                results[metric] = self._evaluate_completion_time(recorded_data)
            elif metric == METRICS_ENERGY_CONSUMPTION:
                results[metric] = self._evaluate_energy(recorded_data)
        
        return results
    
    def _evaluate_position_error(self, data, target):
        if 'position_0' in data and 'position_1' in data and 'position_2' in data:
            final_pos = [
                data['position_0'][-1] if data['position_0'] else 0,
                data['position_1'][-1] if data['position_1'] else 0,
                data['position_2'][-1] if data['position_2'] else 0
            ]
            target_pos = target.get('position', [0, 0, 0])
            return MetricsCalculator.position_error(final_pos, target_pos)
        return 0.0
    
    def _evaluate_orientation_error(self, data, target):
        if all(f'orientation_{i}' in data for i in range(4)):
            final_orn = [
                data[f'orientation_{i}'][-1] if data[f'orientation_{i}'] else 0
                for i in range(4)
            ]
            target_orn = target.get('orientation', [0, 0, 0, 1])
            return MetricsCalculator.orientation_error(final_orn, target_orn)
        return 0.0
    
    def _evaluate_completion_time(self, data):
        if 'timestamp' in data and data['timestamp']:
            return data['timestamp'][-1]
        return 0.0
    
    def _evaluate_energy(self, data):
        force_keys = [k for k in data.keys() if 'force' in k]
        vel_keys = [k for k in data.keys() if 'velocity' in k]
        
        if force_keys and vel_keys:
            forces = np.array([data[k] for k in sorted(force_keys)]).T
            velocities = np.array([data[k] for k in sorted(vel_keys)]).T
            
            min_len = min(len(forces), len(velocities))
            if min_len > 0:
                return MetricsCalculator.energy_consumption(
                    forces[:min_len], velocities[:min_len]
                )
        return 0.0
    
    def compute_score(self, results, weights=None):
        if weights is None:
            weights = {k: 1.0 for k in results.keys()}
        
        total_weight = sum(weights.values())
        score = 0.0
        
        for metric, value in results.items():
            w = weights.get(metric, 0.0)
            score += w * value
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def compare(self, results_list, metric_name):
        values = [r.get(metric_name, float('inf')) for r in results_list]
        best_idx = np.argmin(values)
        worst_idx = np.argmax(values)
        
        return {
            'best_index': best_idx,
            'worst_index': worst_idx,
            'best_value': values[best_idx],
            'worst_value': values[worst_idx],
            'mean': np.mean(values),
            'std': np.std(values)
        }
