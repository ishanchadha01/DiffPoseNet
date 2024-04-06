import torch
from torchvision.models import vgg16


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv_transpose3x3(in_planes, out_planes, stride=1, output_padding=0):
    "3x3 transpose convolution with padding"
    return torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, output_padding=output_padding, bias=False)


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class NFlowNet(torch.nn.Module):
    """
    Encoder - two residual blocks
    Decoder - two transposed residual blocks
    """
    def __init__(self, expansion_factor=2):
        super().__init__(self)
        self.in_channels = 64
        self.encoder = torch.nn.Sequential(
            ResidualBlock(3, 64),
            ResidualBlock(64, 64 * expansion_factor)
        )
        self.decoder = torch.nn.Sequential(
            ResidualBlock(64 * expansion_factor, 64 * expansion_factor),
            ResidualBlock(64 * expansion_factor, 64),
            conv_transpose3x3(64, 3, stride=2, output_padding=1)  # Output layer to match the original image size
        )


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
        
        # Update the fully connected layer for translation and rotation predictions
        # Assuming translational velocity V is 3 elements and rotational velocity Omega is 3 elements
        self.fc_translation = torch.nn.Linear(num_hidden_units, 3)
        self.fc_rotation = torch.nn.Linear(num_hidden_units, 3)


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
        
        # LSTM output for the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Predict translation and rotation separately
        V = self.fc_translation(lstm_out)
        omega = self.fc_rotation(lstm_out)
        
        # Combine the predictions
        predicted_pose = torch.cat((V, omega), dim=1)
        
        return predicted_pose