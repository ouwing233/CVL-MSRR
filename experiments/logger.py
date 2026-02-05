import os
import json
import csv
import time
from collections import defaultdict
from .config import (
    LOG_DIR, DATA_DIR, RESULTS_DIR, get_timestamp,
    LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARNING, LOG_LEVEL_ERROR
)


class ExperimentLogger:
    
    def __init__(self, experiment_name, log_level=LOG_LEVEL_INFO):
        self.experiment_name = experiment_name
        self.log_level = log_level
        self.timestamp = get_timestamp()
        self.log_entries = []
        self.start_time = time.time()
        os.makedirs(LOG_DIR, exist_ok=True)
    
    def _log(self, level, message):
        if level >= self.log_level:
            elapsed = time.time() - self.start_time
            level_names = {
                LOG_LEVEL_DEBUG: 'DEBUG',
                LOG_LEVEL_INFO: 'INFO',
                LOG_LEVEL_WARNING: 'WARNING',
                LOG_LEVEL_ERROR: 'ERROR'
            }
            entry = {
                'time': elapsed,
                'level': level_names.get(level, 'UNKNOWN'),
                'message': message
            }
            self.log_entries.append(entry)
            print(f'[{entry["level"]}] {elapsed:.3f}s: {message}')
    
    def log_debug(self, message):
        self._log(LOG_LEVEL_DEBUG, message)
    
    def log_info(self, message):
        self._log(LOG_LEVEL_INFO, message)
    
    def log_warning(self, message):
        self._log(LOG_LEVEL_WARNING, message)
    
    def log_error(self, message):
        self._log(LOG_LEVEL_ERROR, message)
    
    def log_results(self, results):
        self.log_info('Experiment Results:')
        for key, value in results.items():
            if isinstance(value, float):
                self.log_info(f'  {key}: {value:.6f}')
            else:
                self.log_info(f'  {key}: {value}')
    
    def save(self):
        filename = os.path.join(LOG_DIR, f'{self.experiment_name}_{self.timestamp}.log')
        with open(filename, 'w') as f:
            for entry in self.log_entries:
                f.write(f'[{entry["level"]}] {entry["time"]:.3f}s: {entry["message"]}\n')
        return filename
    
    def save_json(self):
        filename = os.path.join(LOG_DIR, f'{self.experiment_name}_{self.timestamp}.json')
        with open(filename, 'w') as f:
            json.dump(self.log_entries, f, indent=2)
        return filename
    
    def get_entries(self, level=None):
        if level is None:
            return self.log_entries.copy()
        level_names = {
            LOG_LEVEL_DEBUG: 'DEBUG',
            LOG_LEVEL_INFO: 'INFO',
            LOG_LEVEL_WARNING: 'WARNING',
            LOG_LEVEL_ERROR: 'ERROR'
        }
        level_name = level_names.get(level, '')
        return [e for e in self.log_entries if e['level'] == level_name]


class DataRecorder:
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.timestamp = get_timestamp()
        self.data = defaultdict(list)
        self.is_recording = False
        self.record_count = 0
        os.makedirs(DATA_DIR, exist_ok=True)
    
    def start_recording(self):
        self.is_recording = True
        self.record_count = 0
    
    def stop_recording(self):
        self.is_recording = False
    
    def record(self, timestamp, state_dict):
        if not self.is_recording:
            return
        
        self.data['timestamp'].append(timestamp)
        
        for key, value in state_dict.items():
            if isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    self.data[f'{key}_{i}'].append(v)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (list, tuple)):
                        for i, vv in enumerate(v):
                            self.data[f'{key}_{k}_{i}'].append(vv)
                    else:
                        self.data[f'{key}_{k}'].append(v)
            else:
                self.data[key].append(value)
        
        self.record_count += 1
    
    def get_data(self):
        return dict(self.data)
    
    def get_column(self, column_name):
        return self.data.get(column_name, [])
    
    def clear(self):
        self.data = defaultdict(list)
        self.record_count = 0
    
    def save(self, format='csv'):
        if format == 'csv':
            return self._save_csv()
        elif format == 'json':
            return self._save_json()
        return None
    
    def _save_csv(self):
        filename = os.path.join(DATA_DIR, f'{self.experiment_name}_{self.timestamp}.csv')
        if not self.data:
            return None
        
        headers = list(self.data.keys())
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            num_rows = len(self.data[headers[0]]) if headers else 0
            for i in range(num_rows):
                row = [self.data[h][i] if i < len(self.data[h]) else '' for h in headers]
                writer.writerow(row)
        
        return filename
    
    def _save_json(self):
        filename = os.path.join(DATA_DIR, f'{self.experiment_name}_{self.timestamp}.json')
        with open(filename, 'w') as f:
            json.dump(dict(self.data), f, indent=2)
        return filename
    
    def load(self, filename):
        if filename.endswith('.csv'):
            return self._load_csv(filename)
        elif filename.endswith('.json'):
            return self._load_json(filename)
        return None
    
    def _load_csv(self, filename):
        self.clear()
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    try:
                        self.data[key].append(float(value))
                    except ValueError:
                        self.data[key].append(value)
        return dict(self.data)
    
    def _load_json(self, filename):
        self.clear()
        with open(filename, 'r') as f:
            loaded = json.load(f)
            for key, values in loaded.items():
                self.data[key] = values
        return dict(self.data)


class ResultsWriter:
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.timestamp = get_timestamp()
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def write(self, results, params=None, metadata=None):
        output = {
            'experiment': self.experiment_name,
            'timestamp': self.timestamp,
            'results': results
        }
        if params:
            output['parameters'] = params
        if metadata:
            output['metadata'] = metadata
        
        filename = os.path.join(RESULTS_DIR, f'{self.experiment_name}_{self.timestamp}.json')
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        return filename
    
    def write_summary(self, summary, experiment_ids=None):
        output = {
            'summary': summary,
            'timestamp': self.timestamp
        }
        if experiment_ids:
            output['experiments'] = experiment_ids
        
        filename = os.path.join(RESULTS_DIR, f'summary_{self.timestamp}.json')
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        return filename
