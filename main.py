
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TartanAirDataset, NFlowDataset
from models import PoseNet, NFlowNet


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


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Train NFlowNet
    learning_rate = 1e-4
    batch_size = 8
    num_epochs = 400
    train_dataset = NFlowDataset()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    model = NFlowNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (images, normal_flow_gt) in enumerate(train_loader):
            images = images.to(device)
            normal_flow_gt = normal_flow_gt.to(device)
            outputs = model(images) # Forward pass
            loss = criterion(outputs, normal_flow_gt) # Compute loss
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients of all variables wrt loss
            optimizer.step()  # Update all parameters

            # Print loss every 100 batches
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
    #TODO: save model
    print("NFlowNet training complete!")

    ## Train PoseNet
    learning_rate = 1e-5
    num_epochs = 30
    batch_size = 8
    train_dataset = TartanAirDataset()  
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    model = PoseNet()
    model = model.to(device)
    criterion = trans_rot_vel_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, true_poses) in enumerate(train_loader):
            images = images.to(device)
            true_poses = true_poses.to(device)
            predicted_poses = model(images) # Forward pass
            loss = criterion(predicted_poses, true_poses) # Compute loss
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients of all variables wrt loss
            optimizer.step()  # Update all parameters

            # Print loss every 10 batches
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}')
    # TODO: save model
    print("PoseNet training complete!")


def chirality_objective(V, omega, G_x, N_x, A, B):
    """
    Compute the chirality objective for optimization over sampled pixels.

    Args:
    - V: Tensor representing the constant translational velocity.
    - omega: Tensor representing the rotational velocity.
    - G_x: Matrix representing the direction of the image gradients for all pixels.
    - N_x: Vector representing the magnitude of the normal flow for all pixels.
    - A: Tensor representing the matrix involved in the motion flow field projection due to translation.
    - B: Tensor representing the matrix involved in the motion flow field projection due to rotation.

    Returns:
    - A tensor representing the value of the objective function R for all pixels.
    """
    # Ensure that G_x, Beta, Z_x, V, and Omega have the right shapes for batch matrix operations
    # G_x is [N,2]
    # N_x is [N, 1]
    # A and B are [2, 3]
    # V and Omega are [3, 1]
    cost = (G_x @ A @ V) * (N_x - (G_x @ B @ omega))
    smooth_cost = -F.gelu(cost)
    total_cost = torch.sum(smooth_cost)
    return total_cost


def inference():
    ## Chirality optimization step
    # load pose model output for coarse pose
    # load nflownet output for normal flow
    # solve in 1 step using l-bfgs for v, then omega, to get refined pose
    # use depth anything to get initial depth estimate too? for Z_x
    images = None
    N = .05 # use 5% pixels sample each time
    for img1, img2 in zip(images[:-1], images[1:]):

        V_coarse, omega_coarse = None # get output from posenet
        normal_flow = None # get output from 
        gradients = None # get pixel gradients
        depths = None # use depth anything for rough depth estimate

        # get x,y of pixels with max gradients (should be at edges)
        # get normals, V, omega for those pixels
        # compute A and B matrices for all pixels
        

        # Set input tensors to require grad
        V_coarse = torch.tensor(V_coarse, requires_grad=True)
        omega_coarse = torch.tensor(omega_coarse, requires_grad=True)

        # Define the LBFGS optimizer
        optimizer = torch.optim.LBFGS([V_coarse, omega_coarse], line_search_fn='strong_wolfe')

        # Define the closure function required for LBFGS optimization
        def closure():
            optimizer.zero_grad()
            cost = chirality_objective(V_coarse, omega_coarse)
            cost.backward()
            return cost

        # Perform the optimization step
        optimizer.step(closure)

        ## Bi-layer declarative deep net optimization
        # use coarse, refined poses and normal flow to solve
        pass