import time
import numpy as np
from collections import defaultdict
from .config import (
    DEFAULT_EXPERIMENT_DURATION, DEFAULT_SAMPLE_RATE, DEFAULT_TIMEOUT,
    DEFAULT_NUM_TRIALS, SEED_BASE, get_experiment_name, ensure_dirs
)
from .logger import ExperimentLogger, DataRecorder
from .evaluator import Evaluator


class ExperimentRunner:
    
    def __init__(self, scenario, simulation_env=None, communication_node=None):
        self.scenario = scenario
        self.simulation_env = simulation_env
        self.communication_node = communication_node
        self.logger = ExperimentLogger(scenario.name)
        self.recorder = DataRecorder(scenario.name)
        self.evaluator = Evaluator()
        self.is_running = False
        self.start_time = None
        self.elapsed_time = 0.0
        self.results = {}
        ensure_dirs()
    
    def setup(self, params=None):
        self.logger.log_info('Setting up experiment')
        if params:
            self.scenario.set_parameters(params)
        self.scenario.setup(self.simulation_env)
        self.recorder.start_recording()
        self.results = {}
    
    def run(self, duration=None, sample_rate=None):
        if duration is None:
            duration = DEFAULT_EXPERIMENT_DURATION
        if sample_rate is None:
            sample_rate = DEFAULT_SAMPLE_RATE
        
        self.is_running = True
        self.start_time = time.time()
        sample_interval = 1.0 / sample_rate
        last_sample_time = 0.0
        
        self.logger.log_info(f'Starting experiment: duration={duration}s, rate={sample_rate}Hz')
        
        while self.is_running:
            current_time = time.time() - self.start_time
            self.elapsed_time = current_time
            
            if current_time >= duration:
                self.logger.log_info('Experiment duration reached')
                break
            
            if self.scenario.is_complete():
                self.logger.log_info('Scenario completed')
                break
            
            if current_time - last_sample_time >= sample_interval:
                self._sample_data(current_time)
                last_sample_time = current_time
            
            self.scenario.step(self.simulation_env)
            
            if self.simulation_env:
                self.simulation_env.step()
        
        self.is_running = False
        self._finalize()
    
    def _sample_data(self, timestamp):
        state = self.scenario.get_state(self.simulation_env)
        if state:
            self.recorder.record(timestamp, state)
    
    def _finalize(self):
        self.recorder.stop_recording()
        self.logger.log_info('Finalizing experiment')
        
        recorded_data = self.recorder.get_data()
        self.results = self.evaluator.evaluate(
            recorded_data,
            self.scenario.get_target_state(),
            self.scenario.get_metrics()
        )
        
        self.logger.log_results(self.results)
        self.recorder.save()
        self.logger.save()
    
    def stop(self):
        self.is_running = False
    
    def get_results(self):
        return self.results.copy()
    
    def reset(self):
        self.scenario.reset()
        self.recorder.clear()
        self.results = {}
        self.elapsed_time = 0.0


class BatchRunner:
    
    def __init__(self, scenario_class, simulation_env_class=None):
        self.scenario_class = scenario_class
        self.simulation_env_class = simulation_env_class
        self.all_results = []
        self.summary = {}
        ensure_dirs()
    
    def run_batch(self, param_sets, num_trials=None):
        if num_trials is None:
            num_trials = DEFAULT_NUM_TRIALS
        
        self.all_results = []
        
        for param_idx, params in enumerate(param_sets):
            param_results = []
            
            for trial in range(num_trials):
                seed = SEED_BASE + param_idx * num_trials + trial
                np.random.seed(seed)
                
                scenario = self.scenario_class()
                sim_env = self.simulation_env_class() if self.simulation_env_class else None
                
                runner = ExperimentRunner(scenario, sim_env)
                runner.setup(params)
                runner.run()
                
                result = runner.get_results()
                result['trial'] = trial
                result['seed'] = seed
                result['params'] = params.copy()
                param_results.append(result)
                
                if sim_env:
                    sim_env.close()
            
            self.all_results.append(param_results)
        
        self._compute_summary()
        return self.all_results
    
    def run_grid_search(self, param_grid, num_trials=None):
        param_sets = self._expand_grid(param_grid)
        return self.run_batch(param_sets, num_trials)
    
    def _expand_grid(self, param_grid):
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        from itertools import product
        param_sets = []
        for combo in product(*values):
            params = dict(zip(keys, combo))
            param_sets.append(params)
        return param_sets
    
    def _compute_summary(self):
        self.summary = {}
        
        for param_idx, param_results in enumerate(self.all_results):
            if not param_results:
                continue
            
            metrics = defaultdict(list)
            for result in param_results:
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        metrics[key].append(value)
            
            param_summary = {}
            for key, values in metrics.items():
                param_summary[f'{key}_mean'] = np.mean(values)
                param_summary[f'{key}_std'] = np.std(values)
                param_summary[f'{key}_min'] = np.min(values)
                param_summary[f'{key}_max'] = np.max(values)
            
            self.summary[param_idx] = param_summary
        
        return self.summary
    
    def get_summary(self):
        return self.summary.copy()
    
    def get_best_params(self, metric_name, minimize=True):
        best_idx = None
        best_value = float('inf') if minimize else float('-inf')
        
        for param_idx, summary in self.summary.items():
            key = f'{metric_name}_mean'
            if key in summary:
                value = summary[key]
                if minimize and value < best_value:
                    best_value = value
                    best_idx = param_idx
                elif not minimize and value > best_value:
                    best_value = value
                    best_idx = param_idx
        
        if best_idx is not None and self.all_results:
            return self.all_results[best_idx][0].get('params', {})
        return None
