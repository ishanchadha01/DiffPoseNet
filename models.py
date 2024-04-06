import torch
from torchvision.models import vgg16


class NFlowNet(torch.nn.Module):
    pass

class PoseNet(torch.nn.Module):
    """
    One CNN stage, then 2 LSTM stages
    Trained on Tartain Air for 30 epochs
    """
    def __init__(self, H, W, num_hidden_units=250, num_lstm_layers=2):
        super().__init__(self)
        self.cnn = vgg16(pretrained=True).features
        
        # Freeze VGG parameters to avoid training them
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Size of the output tensor from VGG-16 before the LSTM layers
        # This will depend on the size of your input images
        # 512 channels in final VGG-16 output
        cnn_output_size = 512 * H/32 * W/32
        
        # Define the recurrent layers
        self.lstm = torch.nn.LSTM(input_size=cnn_output_size, 
                            hidden_size=num_hidden_units,
                            num_layers=num_lstm_layers,
                            batch_first=True)
        
        # Fully connected layer to get the pose
        # Pose has 6 degrees of freedom (x, y, z, roll, pitch, yaw)
        self.fc_pose = torch.nn.Linear(num_hidden_units, 6)


    def forward(self, x):
        batch_size, sequence_size, C, H, W = x.size()
        
        # Pass input through the VGG-16 encoder
        cnn_encoded = []
        for t in range(sequence_size):
            # Pass each frame through the VGG encoder individually
            frame = self.cnn(x[:, t, :, :, :])
            frame = frame.view(batch_size, -1)  # Flatten the output for the LSTM
            cnn_encoded.append(frame)
        
        # Concatenate the CNN outputs for the LSTM layers
        cnn_encoded = torch.stack(cnn_encoded, dim=1)
        
        # Pass the CNN encoded frames through the LSTM layers
        lstm_out, _ = self.lstm(cnn_encoded)
        
        # We take the output of the LSTM at the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Pass the LSTM output through the fully connected layer to get the pose
        pose = self.fc_pose(lstm_out)
        
        return pose