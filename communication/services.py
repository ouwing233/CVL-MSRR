import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger, SetBool
from rcl_interfaces.srv import SetParameters
from geometry_msgs.msg import Pose
import numpy as np
from .config import NAMESPACE, SERVICE_RECONFIGURE, SERVICE_CALIBRATE, TIMEOUT_SERVICE


class ReconfigurationService(Node):
    
    def __init__(self):
        super().__init__('reconfiguration_service', namespace=NAMESPACE)
        self.callback_group = ReentrantCallbackGroup()
        self.srv = self.create_service(
            Trigger, SERVICE_RECONFIGURE,
            self.reconfigure_callback,
            callback_group=self.callback_group
        )
        self.reconfiguration_handlers = []
        self.current_configuration = {}
        self.target_configuration = {}
    
    def reconfigure_callback(self, request, response):
        try:
            success = self._execute_reconfiguration()
            response.success = success
            response.message = 'Reconfiguration completed' if success else 'Reconfiguration failed'
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response
    
    def _execute_reconfiguration(self):
        for handler in self.reconfiguration_handlers:
            if not handler(self.current_configuration, self.target_configuration):
                return False
        self.current_configuration = self.target_configuration.copy()
        return True
    
    def register_handler(self, handler):
        self.reconfiguration_handlers.append(handler)
    
    def set_target_configuration(self, config):
        self.target_configuration = config.copy()
    
    def get_current_configuration(self):
        return self.current_configuration.copy()


class CalibrationService(Node):
    
    def __init__(self):
        super().__init__('calibration_service', namespace=NAMESPACE)
        self.callback_group = ReentrantCallbackGroup()
        self.srv = self.create_service(
            Trigger, SERVICE_CALIBRATE,
            self.calibrate_callback,
            callback_group=self.callback_group
        )
        self.calibration_data = {}
        self.is_calibrated = False
        self.calibration_handlers = []
    
    def calibrate_callback(self, request, response):
        try:
            success = self._execute_calibration()
            response.success = success
            response.message = 'Calibration completed' if success else 'Calibration failed'
            self.is_calibrated = success
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response
    
    def _execute_calibration(self):
        for handler in self.calibration_handlers:
            result = handler()
            if result is None:
                return False
            self.calibration_data.update(result)
        return True
    
    def register_handler(self, handler):
        self.calibration_handlers.append(handler)
    
    def get_calibration_data(self):
        return self.calibration_data.copy()
    
    def set_calibration_data(self, data):
        self.calibration_data = data.copy()
        self.is_calibrated = True


class ConnectionService(Node):
    
    def __init__(self):
        super().__init__('connection_service', namespace=NAMESPACE)
        self.callback_group = ReentrantCallbackGroup()
        self.connect_srv = self.create_service(
            SetBool, 'connect_modules',
            self.connect_callback,
            callback_group=self.callback_group
        )
        self.connections = set()
        self.connection_handler = None
        self.disconnection_handler = None
        self.pending_connection = None
    
    def connect_callback(self, request, response):
        try:
            if request.data:
                success = self._connect()
                response.success = success
                response.message = 'Connected' if success else 'Connection failed'
            else:
                success = self._disconnect()
                response.success = success
                response.message = 'Disconnected' if success else 'Disconnection failed'
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response
    
    def _connect(self):
        if self.pending_connection is None:
            return False
        m1, m2 = self.pending_connection
        if self.connection_handler:
            if self.connection_handler(m1, m2):
                self.connections.add((min(m1, m2), max(m1, m2)))
                return True
        return False
    
    def _disconnect(self):
        if self.pending_connection is None:
            return False
        m1, m2 = self.pending_connection
        key = (min(m1, m2), max(m1, m2))
        if key in self.connections:
            if self.disconnection_handler:
                if self.disconnection_handler(m1, m2):
                    self.connections.remove(key)
                    return True
        return False
    
    def set_connection_handler(self, handler):
        self.connection_handler = handler
    
    def set_disconnection_handler(self, handler):
        self.disconnection_handler = handler
    
    def set_pending_connection(self, module_1, module_2):
        self.pending_connection = (module_1, module_2)
    
    def get_all_connections(self):
        return list(self.connections)


class ServiceClient(Node):
    
    def __init__(self, node_name='service_client'):
        super().__init__(node_name, namespace=NAMESPACE)
        self.clients = {}
    
    def create_trigger_client(self, service_name):
        client = self.create_client(Trigger, service_name)
        self.clients[service_name] = client
        return client
    
    def create_setbool_client(self, service_name):
        client = self.create_client(SetBool, service_name)
        self.clients[service_name] = client
        return client
    
    def call_trigger(self, service_name, timeout=TIMEOUT_SERVICE):
        if service_name not in self.clients:
            self.create_trigger_client(service_name)
        client = self.clients[service_name]
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        request = Trigger.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()
    
    def call_setbool(self, service_name, value, timeout=TIMEOUT_SERVICE):
        if service_name not in self.clients:
            self.create_setbool_client(service_name)
        client = self.clients[service_name]
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        request = SetBool.Request()
        request.data = value
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()
    
    def reconfigure(self, timeout=TIMEOUT_SERVICE):
        return self.call_trigger(SERVICE_RECONFIGURE, timeout)
    
    def calibrate(self, timeout=TIMEOUT_SERVICE):
        return self.call_trigger(SERVICE_CALIBRATE, timeout)
    
    def connect_modules(self, timeout=TIMEOUT_SERVICE):
        return self.call_setbool('connect_modules', True, timeout)
    
    def disconnect_modules(self, timeout=TIMEOUT_SERVICE):
        return self.call_setbool('connect_modules', False, timeout)
