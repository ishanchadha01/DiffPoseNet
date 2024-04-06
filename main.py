
import torch
from torch.utils.data import DataLoader

from dataset import TartanAirDataset
from models import PoseNet

def main():
    device = 'cuda'

    ## Train NFlowNet

    ## Train PoseNet
    learning_rate = 1e-5
    num_epochs = 30
    batch_size = 8
    train_dataset = TartanAirDataset()  
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    model = PoseNet()
    model = model.to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, true_poses) in enumerate(train_loader):
            images = images.to(device)
            true_poses = true_poses.to(device)
            predicted_poses = model(images) # Forward pass
            loss = loss_function(predicted_poses, true_poses) # Compute loss
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients of all variables wrt loss
            optimizer.step()  # Update all parameters

            # Print loss every 10 batches
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}')

    print("PoseNet training complete!")
