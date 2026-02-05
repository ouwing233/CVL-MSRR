import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

DEFAULT_EXPERIMENT_DURATION = 60.0
DEFAULT_SAMPLE_RATE = 100.0
DEFAULT_TIMEOUT = 300.0

METRICS_POSITION_ERROR = 'position_error'
METRICS_ORIENTATION_ERROR = 'orientation_error'
METRICS_COMPLETION_TIME = 'completion_time'
METRICS_ENERGY_CONSUMPTION = 'energy_consumption'
METRICS_SUCCESS_RATE = 'success_rate'

SCENARIO_RECONFIGURATION = 'reconfiguration'
SCENARIO_LOCOMOTION = 'locomotion'
SCENARIO_MANIPULATION = 'manipulation'
SCENARIO_SELF_ASSEMBLY = 'self_assembly'

LOG_LEVEL_DEBUG = 0
LOG_LEVEL_INFO = 1
LOG_LEVEL_WARNING = 2
LOG_LEVEL_ERROR = 3

DEFAULT_NUM_TRIALS = 10
DEFAULT_NUM_MODULES = 4

SEED_BASE = 42

PARAM_NAMES = [
    'num_modules',
    'target_configuration',
    'initial_configuration',
    'control_frequency',
    'max_velocity',
    'max_force'
]

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def get_experiment_name(scenario, timestamp=None):
    if timestamp is None:
        timestamp = get_timestamp()
    return f'{scenario}_{timestamp}'

def ensure_dirs():
    for d in [DATA_DIR, LOG_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
