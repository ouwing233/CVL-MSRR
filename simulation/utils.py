import pybullet as p
import numpy as np


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


def quaternion_to_euler(quaternion):
    x, y, z, w = quaternion
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    return [roll, pitch, yaw]


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]


def quaternion_inverse(q):
    x, y, z, w = q
    norm_sq = x*x + y*y + z*z + w*w
    return [-x/norm_sq, -y/norm_sq, -z/norm_sq, w/norm_sq]


def rotate_vector(vector, quaternion):
    q_vec = [vector[0], vector[1], vector[2], 0]
    q_inv = quaternion_inverse(quaternion)
    q_rot = quaternion_multiply(quaternion_multiply(quaternion, q_vec), q_inv)
    return [q_rot[0], q_rot[1], q_rot[2]]


def transform_point(point, position, orientation):
    rotated = rotate_vector(point, orientation)
    return [rotated[i] + position[i] for i in range(3)]


def inverse_transform_point(point, position, orientation):
    translated = [point[i] - position[i] for i in range(3)]
    q_inv = quaternion_inverse(orientation)
    return rotate_vector(translated, q_inv)


def distance_between_points(p1, p2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def angle_between_vectors(v1, v2):
    v1_norm = np.array(v1) / np.linalg.norm(v1)
    v2_norm = np.array(v2) / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(dot)


def interpolate_position(start, end, t):
    return [start[i] + t * (end[i] - start[i]) for i in range(3)]


def slerp_quaternion(q1, q2, t):
    dot = sum(a * b for a, b in zip(q1, q2))
    if dot < 0:
        q2 = [-x for x in q2]
        dot = -dot
    
    if dot > 0.9995:
        result = [q1[i] + t * (q2[i] - q1[i]) for i in range(4)]
        norm = np.sqrt(sum(x * x for x in result))
        return [x / norm for x in result]
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    q_perp = [q2[i] - dot * q1[i] for i in range(4)]
    norm = np.sqrt(sum(x * x for x in q_perp))
    q_perp = [x / norm for x in q_perp]
    
    return [np.cos(theta) * q1[i] + np.sin(theta) * q_perp[i] for i in range(4)]


def create_grid_positions(rows, cols, spacing, start_position=None):
    if start_position is None:
        start_position = [0, 0, 0]
    positions = []
    for i in range(rows):
        for j in range(cols):
            pos = [
                start_position[0] + j * spacing,
                start_position[1] + i * spacing,
                start_position[2]
            ]
            positions.append(pos)
    return positions


def create_line_positions(count, spacing, direction, start_position=None):
    if start_position is None:
        start_position = [0, 0, 0]
    direction = np.array(direction) / np.linalg.norm(direction)
    positions = []
    for i in range(count):
        pos = [start_position[j] + i * spacing * direction[j] for j in range(3)]
        positions.append(pos)
    return positions


def compute_inertia_tensor(mass, shape_type, dimensions):
    if shape_type == "box":
        w, h, d = dimensions
        Ixx = mass / 12.0 * (h*h + d*d)
        Iyy = mass / 12.0 * (w*w + d*d)
        Izz = mass / 12.0 * (w*w + h*h)
    elif shape_type == "sphere":
        r = dimensions[0]
        Ixx = Iyy = Izz = 2.0/5.0 * mass * r * r
    elif shape_type == "cylinder":
        r, h = dimensions[0], dimensions[1]
        Ixx = Iyy = mass / 12.0 * (3*r*r + h*h)
        Izz = mass / 2.0 * r * r
    else:
        Ixx = Iyy = Izz = 1.0
    return [Ixx, Iyy, Izz]
