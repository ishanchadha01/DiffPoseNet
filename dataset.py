import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

import os
import math


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


class TartanAirDataset(Dataset):
    def __init__(self):
        self.data_path = None
        self.img_path = os.path.join(self.data_path, 'image_left')
        self.flow_path = os.path.join(self.data_path, 'flow')
        self.pose_file = os.path.join(self.data_path, 'pose_left.txt')
        self.focal = 320.0

        # Get image pairs
        self.image_pairs = []
        self.flows = []
        img_files = sorted(os.listdir(self.img_path))
        for img1, img2 in zip(img_files[:-1], img_files[1:]):
            img_data1 = cv2.imread(os.path.join(self.img_path, img1))
            img_data2 = cv2.imread(os.path.join(self.img_path, img2))
            self.image_pairs.append([img_data1, img_data2])
        
        # Get flows
        for img in sorted(os.listdir(self.flow_path)):
            flow_data = cv2.imread(os.path.join(self.flow_path, img))
            self.flows.append(flow_data)
        self.flows = sorted(self.flows)

        # Get poses
        self.poses = []
        f = open(self.pose_file, 'r')
        lines = f.readlines()
        for pose1, pose2 in zip(lines[:-1], lines[1:]):
            nums1 = [float(num) for num in pose1.split(' ')]
            nums2 = [float(num) for num in pose2.split(' ')]
            q1 = torch.tensor(nums1[3:])
            q2 = torch.tensor(nums2[3:])
            q_diff = quaternion_difference(q1, q2)
            euler_rot = quaternion_to_euler(q_diff)
            trans_diff = torch.tensor(nums2[:3]) - torch.tensor(nums1[:3])
            self.poses.append(torch.cat((trans_diff, euler_rot), dim=1))
        f.close()


    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        return self.image_pairs[idx], self.flows[idx], self.poses[idx]
        # return pairs of imgs, pairs of poses, and one flow


class C3VDDataset(Dataset):
    pass