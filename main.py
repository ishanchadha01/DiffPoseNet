
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import TartanAirDataset
from ddn import DeclarativeLayer
from models import PoseNet, NFlowNet, AdaptivePoseNode, ChiralityNode

from tqdm import tqdm


def compute_A_B_matrices(batch_size, channels, height, width, f):
    # Create a meshgrid of x and y coordinates
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    
    # Normalize x and y coordinates to be centered and in camera coordinates
    x = x.float() - width // 2
    y = y.float() - height // 2

    # Reshape the grid so that it matches the BCHW format
    x_grid = x.view(1, 1, height, width).repeat(batch_size, channels, 1, 1)
    y_grid = y.view(1, 1, height, width).repeat(batch_size, channels, 1, 1)
    
    # Compute matrices A and B for each pixel
    A = torch.stack((-f * torch.ones_like(x_grid), 
                     torch.zeros_like(x_grid), 
                     x_grid,
                     torch.zeros_like(x_grid),
                     -f * torch.zeros_like(y_grid), 
                     y_grid,
                     torch.zeros_like(x_grid), 
                     torch.zeros_like(x_grid), 
                     torch.ones_like(x_grid)), dim=-1).reshape(batch_size, channels, height, width, 3, 3)

    B = torch.stack((x_grid * y_grid, 
                     - (f**2 + x_grid**2), 
                     f * y_grid,
                     f**2 + y_grid**2, 
                     -x_grid * y_grid, 
                     -f * x_grid,
                     -y_grid, 
                     x_grid, 
                     torch.zeros_like(x_grid)), dim=-1).reshape(batch_size, channels, height, width, 3, 3)

    return A, B


def compute_image_gradient(image):
    """
    Compute the gradient of an image using Sobel operators.

    Args:
    image (torch.Tensor): A 4D tensor of shape (B, C, H, W) where B is the batch size,
                          C is the number of channels, H is the height, and W is the width.

    Returns:
    torch.Tensor: The gradient magnitude of the image.
    """
    # Define Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    # Assume image is of shape (B, C, H, W)
    # Need to ensure the filters are on the same device and of the same data type as the image
    sobel_x = sobel_x.to(device=image.device, dtype=image.dtype)
    sobel_y = sobel_y.to(device=image.device, dtype=image.dtype)

    # Apply filters to compute gradients
    gradient_x = F.conv2d(image, sobel_x, padding=1)
    gradient_y = F.conv2d(image, sobel_y, padding=1)

    # Compute the gradient magnitude
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
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
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
    optimizer = torch.optim.Adam(pose_net.parameters(), lr=learning_rate)
    for param in flow_net.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        pose_net.train()
        for batch_idx, (image_pairs, _) in enumerate(train_loader): # self-supervised
            images = image_pairs.to(device) # [B, 2, C, H, W] where B is batch size
            predicted_poses_init = pose_net(images) # [B,6]
            normal_flow = flow_net(images) # [B, C, H, W]
            gradients = compute_image_gradient(images[:,0,...]) # get batch gradients direction and magnitude for first img in each pair, [B,C,H,W,2]
            
            V_coarse = predicted_poses_init[:,:3] # accounting for batches
            omega_coarse = predicted_poses_init[:,3:]

            # get x,y of pixels with max gradients (should be at edges)
            # get normals, V, omega for those pixels
            # compute A and B matrices for sampled pixels
            A, B = compute_A_B_matrices(images.shape[0], images.shape[1], images.shape[2], images.shape[3], focal_len)
        
            chirality_node = ChiralityNode(normal_flow, gradients, A, B)
            chirality_net = DeclarativeLayer(chirality_node)
            refined_poses = chirality_net((V_coarse, omega_coarse)) # process in batches TODO does this work? check dims
            V_refined = refined_poses[:,0,...] # account for batches
            omega_refined = refined_poses[:,1,...]

            ## Bi-layer declarative deep net optimization
            # use coarse, refined poses and normal flow to solve
            adaptive_pose_node = AdaptivePoseNode(normal_flow, gradients, A, B)
            adaptive_pose_net = DeclarativeLayer(adaptive_pose_node)
            V_coarse, omega_coarse = adaptive_pose_net((V_refined, omega_refined, V_coarse, omega_coarse))
            predicted_poses = torch.cat((V_coarse, omega_coarse), dim=1)

            loss = adaptive_pose_node.objective(predicted_poses) # Compute loss
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients of all variables wrt loss
            optimizer.step()  # Update all parameters

            # Print loss every 10 batches
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}')
    torch.save(pose_net, "models/refined_pose_net.pth")
    print("PoseNet pre-training complete!")
    return pose_net


def train(train_flow=True, train_pose=True, refine_pose=True):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    ## Train NFlowNet
    if train_flow:
        flow_net = train_flow_net(device)
        out = flow_net

    ## Train PoseNet
    if train_pose:
        pose_net = train_pose_net(device)
        out = pose_net

    ## PoseNet + Chirality Layer training with Refinement
    if refine_pose:
        refined_pose_net = train_refined_net(pose_net, flow_net, device)
        out = refined_pose_net

    return out


#TODO fix this for batched inference
def inference():
    ## Chirality optimization step
    # load pose model output for coarse pose
    # load nflownet output for normal flow
    # solve in 1 step using l-bfgs for v, then omega, to get refined pose
    images = None
    N = .05 * len(images) # use 5% pixels sample each time
    poses = []
    for img1, img2 in zip(images[:-1], images[1:]):

        # only need to do 5 iterations of this

        V_coarse, omega_coarse = None # get output from posenet
        normal_flow = None # get output from NFlowNet
        gradients = None # get pixel gradients
        # depths = None # use depth anything for rough depth estimate

        # get x,y of pixels with max gradients (should be at edges)
        # get normals, V, omega for those pixels
        # compute A and B matrices for sampled pixels
        A = None
        B = None
        
        chirality_net = DeclarativeLayer(ChiralityNode(normal_flow, gradients, A, B))
        V_refined, omega_refined = chirality_net((V_coarse, omega_coarse))

        ## Bi-layer declarative deep net optimization
        # use coarse, refined poses and normal flow to solve
        adaptive_pose_net = DeclarativeLayer(AdaptivePoseNode(normal_flow, gradients, A, B))
        V_coarse, omega_coarse = adaptive_pose_net((V_refined, omega_refined, V_coarse, omega_coarse))
        poses.append([V_coarse, omega_coarse])
    
    # write poses to npy and compare to c3vd


if __name__=='__main__':
    train()