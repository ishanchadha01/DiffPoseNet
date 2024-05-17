import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from tqdm import tqdm

import os
import math
import json

from utils import rot_to_euler, quaternion_to_euler, quaternion_difference



class TartanAirDataset(Dataset):
    def __init__(self):
        super().__init__()
        # self.data_path = "/storage/home/hcoda1/3/ichadha3/scratch/data/abandonedfactory/Hard/abandonedfactory/Hard/P000/"
        self.data_path = "/Users/ishan/Documents/research/DiffPoseNet/small_data"
        self.img_path = os.path.join(self.data_path, 'image_left')
        self.flow_path = os.path.join(self.data_path, 'flow')
        self.pose_file = os.path.join(self.data_path, 'pose_left.txt')
        self.focal = 320.0

        # Get image pairs
        print("Loading dataset images")
        self.image_pairs = []
        self.flows = []
        img_files = sorted(os.listdir(self.img_path))
        for img1, img2 in tqdm(zip(img_files[:-1], img_files[1:])):
            img_data1 = torch.tensor(cv2.imread(os.path.join(self.img_path, img1)), dtype=torch.float32).permute(2,0,1) # put channels first
            img_data2 = torch.tensor(cv2.imread(os.path.join(self.img_path, img2)), dtype=torch.float32).permute(2,0,1)
            self.image_pairs.append(torch.stack([img_data1, img_data2]))
        
        # Get flows
        print("Loading dataset flows")
        flow_files = sorted(os.listdir(self.flow_path))[:10]
        for img in tqdm(flow_files):
            flow_data = np.load(os.path.join(self.flow_path, img))
            flow_data = np.linalg.norm(flow_data, axis=-1) # only get magnitudes
            self.flows.append(torch.tensor(flow_data))

        # Get poses
        print("Loading dataset poses")
        self.poses = []
        f = open(self.pose_file, 'r')
        lines = f.readlines()
        line_pairs = list(zip(lines[:-1], lines[1:]))[:10]
        for pose1, pose2 in tqdm(line_pairs):
            nums1 = [float(num) for num in pose1.split(' ')]
            nums2 = [float(num) for num in pose2.split(' ')]
            q1 = torch.tensor(nums1[3:])
            q2 = torch.tensor(nums2[3:])
            q_diff = quaternion_difference(q1, q2)
            euler_rot = quaternion_to_euler(q_diff)
            trans_diff = torch.tensor(nums2[:3]) - torch.tensor(nums1[:3])
            self.poses.append(torch.cat((trans_diff, euler_rot), dim=0))
        f.close()


    def __len__(self):
        # return len(self.image_pairs)
        return 10 # temp for debugging
    
    def __getitem__(self, idx):
        return self.image_pairs[idx], self.flows[idx], self.poses[idx]
        # return pairs of imgs, pairs of poses, and one flow


class C3VDDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data_path = "/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/DiffPoseNet/data/cecum_t1_a"
        self.img_path = os.path.join(self.data_path, 'images')
        config_path = os.path.join(self.data_path, 'transforms_train.json')

        config_data = None
        with open(config_path, "r") as f:
            config_data = json.load(f)
        self.focal = (config_data['fx'] + config_data['fy'])/2

        # Get image pairs
        print("Loading dataset images")
        self.image_pairs = []
        img_files = sorted(os.listdir(self.img_path))
        for img1, img2 in tqdm(zip(img_files[:-1], img_files[1:])):
            img_data1 = torch.tensor(cv2.imread(os.path.join(self.img_path, img1)), dtype=torch.float32).permute(2,0,1) # put channels first
            img_data2 = torch.tensor(cv2.imread(os.path.join(self.img_path, img2)), dtype=torch.float32).permute(2,0,1)
            self.image_pairs.append(torch.stack([img_data1, img_data2])) # [2,C,H,W]

        # Get poses
        print("Loading dataset poses")
        self.poses = []
        frames = config_data['frames']
        for frame1, frame2 in zip(frames[:-1], frames[1:]):
            pose1 = self.get_w2c_pose(frame1)
            pose2 = self.get_w2c_pose(frame2)
            self.poses.append(pose2 - pose1)

    def get_w2c_pose(self, frame):
        c2w = np.array(frame['transform_matrix'])
        w2c = np.linalg.inv(c2w)
        rot_mat = w2c[:3,:3]
        rpy = rot_to_euler(rot_mat)
        translation = w2c[:3, 3]
        return torch.cat((translation, rpy), dim=0)

    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        return self.image_pairs[idx], self.poses[idx]