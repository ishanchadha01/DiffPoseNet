
import torch
from torch.utils.data import DataLoader

from dataset import TartanAirDataset, NFlowDataset
from models import PoseNet, NFlowNet

def main():
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


    ## Train PoseNet
    learning_rate = 1e-5
    num_epochs = 30
    batch_size = 8
    train_dataset = TartanAirDataset()  
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    model = PoseNet()
    model = model.to(device)
    criterion = torch.nn.MSELoss()
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

    print("PoseNet training complete!")
