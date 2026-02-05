NAMESPACE = 'msrr'

TOPIC_POSE = 'pose'
TOPIC_VELOCITY = 'velocity'
TOPIC_COMMAND = 'command'
TOPIC_STATE = 'state'
TOPIC_SENSOR = 'sensor'
TOPIC_TARGET = 'target'

SERVICE_RECONFIGURE = 'reconfigure'
SERVICE_CALIBRATE = 'calibrate'
SERVICE_CONNECT = 'connect'
SERVICE_DISCONNECT = 'disconnect'

ACTION_MOVE = 'move'
ACTION_ROTATE = 'rotate'
ACTION_RECONFIGURE = 'reconfigure_action'

QOS_RELIABLE = 10
QOS_BEST_EFFORT = 1
QOS_SENSOR = 5

PUBLISH_RATE_POSE = 30.0
PUBLISH_RATE_STATE = 10.0
PUBLISH_RATE_COMMAND = 50.0

TIMEOUT_SERVICE = 5.0
TIMEOUT_ACTION = 30.0

MAX_QUEUE_SIZE = 100

MSG_TYPE_POSE = 'geometry_msgs/msg/PoseStamped'
MSG_TYPE_TWIST = 'geometry_msgs/msg/Twist'
MSG_TYPE_POINT = 'geometry_msgs/msg/Point'
MSG_TYPE_QUATERNION = 'geometry_msgs/msg/Quaternion'

NODE_NAME_CONTROLLER = 'msrr_controller'
NODE_NAME_PERCEPTION = 'msrr_perception'
NODE_NAME_PLANNER = 'msrr_planner'
NODE_NAME_SIMULATOR = 'msrr_simulator'

PARAM_MODULE_COUNT = 'module_count'
PARAM_UPDATE_RATE = 'update_rate'
PARAM_DEBUG_MODE = 'debug_mode'
