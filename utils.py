import numpy as np
import torch


def quaternion_to_euler(q):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw) using PyTorch.

    Args:
    q (torch.Tensor): A tensor of shape (..., 4) representing quaternions.

    Returns:
    torch.Tensor: A tensor of shape (..., 3) representing the Euler angles
                  (roll, pitch, yaw) in radians.
    """
    # Normalize the quaternion
    q = q / q.norm(p=2, dim=-1, keepdim=True)

    # Extract the quaternion components
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Compute the roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Compute the pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1, 1)  # Clamp to handle numerical errors
    pitch = torch.asin(sinp)

    # Compute the yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


def quaternion_conjugate(q):
    """Compute the conjugate of a quaternion."""
    q_conj = q.clone()
    q_conj[..., 1:] = -q_conj[..., 1:]  # negate the vector part
    return q_conj


def quaternion_multiply(q1, q2):
    """Compute the product of two quaternions."""
    # Extract scalar and vector parts of the quaternions
    s1, v1 = q1[..., 0], q1[..., 1:]
    s2, v2 = q2[..., 0], q2[..., 1:]


    # Compute the product
    s = s1 * s2 - torch.dot(v1, v2)
    v = s1 * v2 + s2 * v1 + torch.cross(v1, v2)
    return torch.cat((s.unsqueeze(-1), v), dim=-1)


def quaternion_difference(q1, q2):
    """Compute the quaternion representing the rotation from q1 to q2."""
    q1_inv = quaternion_conjugate(q1)
    q_diff = quaternion_multiply(q2, q1_inv)
    return q_diff


def rotation_matrix_to_euler_xyz(R):
    """
    Convert a rotation matrix to Euler angles in the XYZ sequence.
    R is a 3x3 numpy array representing the rotation matrix.
    """
    # Ensure the matrix is a proper rotation matrix
    assert np.allclose(np.dot(R, R.T), np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    
    if R[2, 1] < 1:
        if R[2, 1] > -1:
            theta_y = np.arcsin(-R[2, 1])
            theta_z = np.arctan2(R[2, 0], R[2, 2])
            theta_x = np.arctan2(R[0, 1], R[1, 1])
        else:
            # Gimbal lock: theta_y = -90 degrees
            theta_y = np.pi / 2
            theta_z = np.arctan2(-R[0, 2], R[0, 0])
            theta_x = 0
    else:
        # Gimbal lock: theta_y = 90 degrees
        theta_y = -np.pi / 2
        theta_z = np.arctan2(-R[0, 2], R[0, 0])
        theta_x = 0

    return np.array([theta_x, theta_y, theta_z])