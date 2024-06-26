
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
import numpy as np

from dataset import TartanAirDataset, C3VDDataset
from ddn import DeclarativeLayer
from models import PoseNet, NFlowNet, ChiralityNode2, BilevelOptimization
from utils import normalize_pose, euler_to_homogeneous

import os
from tqdm import tqdm


def compute_A_B_matrices(batch_size, channels, height, width, f):
    # Create a meshgrid of x and y coordinates
    y, x = torch.meshgrid(torch.arange(height).float() - (height - 1) / 2.0,
                          torch.arange(width).float() - (width - 1) / 2.0)

    # Normalize coordinates with the focal length
    x = x / f
    y = y / f

    # Reshape the grid so that it matches the BCHW format
    x_grid = x.unsqueeze(0).repeat(batch_size, 1, 1)
    y_grid = y.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Compute matrices A and B for each pixel
    A = torch.stack([
            torch.stack([-1 * torch.ones_like(x_grid), torch.zeros_like(x_grid), x_grid], dim=-1),
            torch.stack([torch.zeros_like(x_grid), -1 * torch.zeros_like(y_grid), y_grid], dim=-1)
        ], dim=-2)

    B = torch.stack([
            torch.stack([x_grid * y_grid, -(1 + x_grid**2), y_grid], dim=-1),
            torch.stack([1 + y_grid**2, -x_grid * y_grid, -1 * x_grid], dim=-1),
        ], dim=-2)

    return A, B


def compute_image_gradient(image):
    """
    Compute the gradient of an image using Sobel operators.

    Args:
    image (torch.Tensor): A 4D tensor of shape (B, C, H, W) where B is the batch size,
                          C is the number of channels, H is the height, and W is the width.

    Returns:
    torch.Tensor: The gradient directions of the image.
    """
    # Define Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    # Replicate Sobel filters for each channel
    C = image.size(1)
    sobel_x = sobel_x.repeat(C, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1)

    # Assume image is of shape (B, C, H, W)
    # Need to ensure the filters are on the same device and of the same data type as the image
    sobel_x = sobel_x.to(device=image.device, dtype=image.dtype)
    sobel_y = sobel_y.to(device=image.device, dtype=image.dtype)

    # Apply filters to compute gradients
    gradient_x = F.conv2d(image, sobel_x, padding=1, groups=C)
    gradient_y = F.conv2d(image, sobel_y, padding=1, groups=C)

    # Compute the gradient magnitude and average over channels
    gradient_x = gradient_x.mean(dim=1, keepdim=False)
    gradient_y = gradient_y.mean(dim=1, keepdim=False)
    gradients = torch.stack((gradient_x, gradient_y), dim=-1)

    return gradients


def trans_rot_vel_loss(predicted, true, lambda_weight=0.5):
    predicted_translation = predicted[:, :3]
    predicted_rotation = predicted[:, 3:]

    true_translation = true[:, :3]
    true_rotation = true[:, 3:]

    # Calculate the mean squared error separately
    loss_translation = torch.nn.MSELoss()(predicted_translation, true_translation)
    loss_rotation = torch.nn.MSELoss()(predicted_rotation, true_rotation)

    # Combine the losses with the weighting factor
    loss = loss_translation + lambda_weight * loss_rotation
    return loss


def train_flow_net(device='cpu'):
    learning_rate = 1e-4
    batch_size = 8
    num_epochs = 400
    train_dataset = TartanAirDataset()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    flow_net = NFlowNet()
    flow_net = flow_net.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(flow_net.parameters(), lr=learning_rate)
    for epoch in tqdm(range(num_epochs)):
        for i, (images, normal_flow_gt, _) in enumerate(train_loader): # should be pairs of imgs and pairs of flows
            images = images.to(device)
            normal_flow_gt = normal_flow_gt.to(device)
            outputs = flow_net(images[:,0], images[:,1]) # Forward pass
            loss = criterion(outputs, normal_flow_gt) # Compute loss
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients of all variables wrt loss
            optimizer.step()  # Update all parameters

            # Print loss every 100 batches
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
    torch.save(flow_net, "models/flow_net.pth")
    print("NFlowNet pre-training complete!")
    return flow_net


def train_pose_net(device='cpu'):
    learning_rate = 1e-5
    num_epochs = 30
    batch_size = 8
    train_dataset = TartanAirDataset()  
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    pose_net = PoseNet()
    pose_net = pose_net.to(device)
    criterion = trans_rot_vel_loss
    optimizer = torch.optim.Adam(pose_net.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        pose_net.train()
        for batch_idx, (images, _, true_poses) in enumerate(train_loader):
            images = images.to(device)
            true_poses = true_poses.to(device)
            predicted_poses = pose_net(images) # Forward pass
            loss = criterion(predicted_poses, true_poses) # Compute loss
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients of all variables wrt loss
            optimizer.step()  # Update all parameters

            # Print loss every 10 batches
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}')
    torch.save(pose_net, "models/pose_net.pth")
    print("PoseNet pre-training complete!")
    return pose_net


def train_refined_net(pose_net, flow_net, device='cpu'):
    learning_rate = 1e-5
    num_epochs = 120
    batch_size = 8
    train_dataset = TartanAirDataset()
    focal_len = train_dataset.focal
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    optimizer_nn = torch.optim.Adam(pose_net.parameters(), lr=learning_rate)
    for param in flow_net.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        pose_net.train()
        for batch_idx, (images, _, _) in enumerate(train_loader): # self-supervised

            ### First do chirality layer pass
            images = images.to(device) # [B, 2, C, H, W] where B is batch size
            predicted_poses_init = pose_net(images) # [B,6]
            normal_flow = flow_net(images[:,0], images[:,1]).detach().clone() # [B, H, W]
            normal_flow.requires_grad = True
            gradients = compute_image_gradient(images[:,0,...]) # get batch gradients direction and magnitude for first img in each pair, [B,H,W,2]
            
            V_coarse = predicted_poses_init[:,:3] # accounting for batches
            omega_coarse = predicted_poses_init[:,3:]

            # get x,y of pixels with max gradients (should be at edges)
            # get normals, V, omega for those pixels
            # compute A and B matrices for sampled pixels
            A, B = compute_A_B_matrices(images.shape[0], images.shape[2], images.shape[3], images.shape[4], focal_len)
        
            chirality_node = ChiralityNode2(normal_flow, gradients, A, B)
            refined_poses = chirality_node(V_coarse, omega_coarse) # process in batches TODO does this work? check dims
            V_refined = refined_poses[:,0] # refined_poses of shape B,2,pose
            omega_refined = refined_poses[:,1]
            # nn_loss = torch.no_grad()(chirality_node.objective)(V_refined, omega_refined)
            nn_loss = chirality_node.objective(V_refined, omega_refined)

            optimizer_nn.zero_grad()  # Clear previous gradients
            # for name, param in pose_net.named_parameters():
            #     print(name, param.requires_grad)

            nn_loss.backward()  # Compute gradients of all variables wrt loss
            optimizer_nn.step()  # Update all parameters

            # # Print loss every 10 batches
            # if batch_idx % 10 == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {nn_loss.item()}')


            ### Perform adaptive refinement at the end with bilayer optimization
            ## Get coarse pose initiaization after previous bwd pass, while using prev refined poses as Y labels
            V_coarse = predicted_poses_init[:,:3] # accounting for batches
            omega_coarse = predicted_poses_init[:,3:]
            
            ## Solve bilayer optimization
            P_r_init = torch.stack([V_refined, omega_refined], dim=1)
            P_c = torch.stack([V_coarse, omega_coarse], dim=1)
            bilevel_opt = DeclarativeLayer(BilevelOptimization(normal_flow, gradients, A, B, P_r_init=P_r_init))
            P_r = bilevel_opt(P_c)
            ddn_loss = torch.no_grad()(bilevel_opt)(P_c, y=P_r)
            ddn_loss.backward()

            # Print loss every 10 batches
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], NN Loss: {nn_loss.item()}, DDN Loss: {ddn_loss.item()}')
            

    # torch.save(pose_net, "models/refined_pose_net.pth")
    print("PoseNet pre-training complete!")
    return pose_net


def train(train_flow=False, train_pose=False, refine_pose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Train NFlowNet
    if train_flow:
        flow_net = train_flow_net(device)
    else:
        print("Loading flow net")
        flow_net = torch.load('models/flow_net.pth', map_location=device).to(device)

    ## Train PoseNet
    if train_pose:
        pose_net = train_pose_net(device)
    else:
        print("Loading pose net")
        pose_net = torch.load('models/pose_net.pth', map_location=device).to(device)

    # c3vd_inference(pose_net)

    # PoseNet + Chirality Layer training with Refinement
    if refine_pose:
        refined_pose_net = train_refined_net(pose_net, flow_net, device)


#TODO fix this for batched inference
def inference():
    device = 'cpu'
    pose_net = torch.load('models/pose_net.pth', map_location=torch.device('cpu')).to(device)
    # flow_net = torch.load('models/flow_net.pth', map_location=torch.device('cpu')).to(device)
    pose_net.eval()
    # flow_net.eval()
    transforms = []
    imgs = sorted(os.listdir('./pipe_data/image_left'))
    for img_pair_files in tqdm(zip(imgs[:-1], imgs[1:])):
        img_pair_paths = [os.path.join('./pipe_data/image_left', fname) for fname in img_pair_files]
        img_pair = torch.stack((
            torch.from_numpy(cv2.imread(img_pair_paths[0])[::2, ::2, :]).permute(2,0,1).float(), 
            torch.from_numpy(cv2.imread(img_pair_paths[1])[::2, ::2, :]).permute(2,0,1).float()
        ), dim=0).unsqueeze(0)
        transform = pose_net(img_pair).detach().numpy()
        # flow = flow_net(img_pair[:,0], img_pair[:,1]).detach().squeeze(0).numpy()
        print(f"Images: {img_pair_files[0]} {img_pair_files[1]}")
        print(f"Transform: {transform}")
        transforms.append(transform)
        # cv2.imwrite(f"./pipe_data/flows_pred/flow_{img_pair_files[0]}_{img_pair_files[1]}", flow)

    # Starting at origin, map where camera moves
    points = np.zeros((len(transforms)+1, 3, 1))
    for i, transform in enumerate(transforms):
        homogenous_point = np.ones((4,1))
        homogenous_point[:3] = points[i]
        T = euler_to_homogeneous(transform[0])
        new_homogenous_point = T @ homogenous_point
        points[i+1] = new_homogenous_point[:3]
    print(points)



def c3vd_inference(pose_net):
    batch_size = 8
    dataset = C3VDDataset()
    test_loader = DataLoader(dataset, batch_size=batch_size)
    loss_func = torch.MSELoss()
    # create test dataloader and get loss? then, downsample one full runthrough and plot poses. might need to train posenet more first
    # start with just pose net, flows are important for refinement really anyways
    # flows = flow_net(imgs)
    losses = []
    pose_net.eval()
    for batch_idx, (img_pair, pose_true) in tqdm(test_loader):
        pose_estimate = pose_net(img_pair)
        pose_true = normalize_pose(pose_true)
        pose_estimate = normalize_pose(pose_estimate)
        loss = loss_func(pose_estimate, pose_true)
        losses.append(loss)





if __name__=='__main__':
    train()
    # inference()