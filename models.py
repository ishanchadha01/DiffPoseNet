import torch
from torch.autograd import Function
import torch.nn.functional as F
from torchvision.models import vgg16
import numpy as np

from ddn import AbstractDeclarativeNode


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
        super().__init__()
        self.in_channels = 64
        self.encoder = torch.nn.Sequential(
            ResidualBlock(3*2, 64), # 6 channels to handle both images
            ResidualBlock(64, 64 * expansion_factor)
        )
        # Output layer to match the original image size
        self.decoder = torch.nn.Sequential(
            ResidualBlock(64 * expansion_factor, 64 * expansion_factor),
            ResidualBlock(64 * expansion_factor, 64),
            conv_transpose3x3(64, 1, stride=2, output_padding=1)
        )
    
    def forward(self, img1, img2):
        x = torch.cat((img1, img2), dim=1) # along first non-batchsize dim to have 6 channels
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(img1.shape[-2], img1.shape[-1]), mode='bilinear', align_corners=False)
        x = x.squeeze(1)
        return x


class PoseNet(torch.nn.Module):
    """
    One CNN stage, then 2 LSTM stages
    Trained on Tartain Air for 30 epochs
    """
    def __init__(self, H=480, W=640, num_hidden_units=250, num_lstm_layers=2):
        super().__init__()
        self.cnn = vgg16(pretrained=True).features
        
        # Freeze VGG parameters to avoid training them
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Size of the output tensor from VGG-16 before the LSTM layers
        # This will depend on the size of your input images
        # 512 channels in final VGG-16 output
        cnn_output_size = 512 * int(H/32) * int(W/32)
        
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


class ChiralityNode2(torch.nn.Module):
    """
    Node that solves pose chirality optimization problem
    """
    def __init__(self):
        super().__init__()

    def __init__(self, N_x, G_x, A, B):
        super().__init__()
        self.N_x = N_x # Magnitude of the normal flow for all pixels
        self.G_x = G_x # The image gradients for all pixels
        self.A = A # Matrix for the motion flow field due to translation
        self.B = B # Matrix for the motion flow field due to rotation
        #TODO: adjust eps based on convergence

    def objective(self, V, omega, y=0):
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
        # G_x is [N, 2]
        # N_x is [N, 1]
        # A and B are [2, 3]
        # V and Omega are [N,3]
        cost = torch.einsum('bhwi,bhwij,bj->bhw', self.G_x, self.A, V) * (self.N_x - torch.einsum('bhwi,bhwij,bj->bhw', self.G_x, self.B, omega)) # [N,1]
        smooth_cost = -F.gelu(cost) # twice differentiable and bias towards positive value, enforcing constraint
        total_cost = torch.mean(smooth_cost) #TODO: maybe take expected value?
        return total_cost - y
    

        
    def forward(self, V_refined, omega_refined):
        V = V_refined.detach().requires_grad_()
        omega = omega_refined.detach().requires_grad_()
        with torch.enable_grad():
            optimizer = torch.optim.LBFGS(
                [V, omega], 
                lr=1, 
                max_iter=20, 
                line_search_fn='strong_wolfe')

            def closure():
                optimizer.zero_grad()
                f = self.objective(V, omega)
                f.backward(retain_graph=True)
                return f

            optimizer.step(closure)

        # return output, ctx
        y = torch.stack([V, omega], dim=1)
        return y.detach()
    


class BilevelOptimization(AbstractDeclarativeNode):
    def __init__(self, N_x, G_x, A, B, P_r_init=None):
        super().__init__()
        self.N_x = N_x # Magnitude of the normal flow for all pixels
        self.G_x = G_x # The image gradients for all pixels
        self.A = A # Matrix for the motion flow field due to translation
        self.B = B # Matrix for the motion flow field due to rotation
        self.P_r_init = P_r_init

    def solve(self, P_c):
        # should start with random Pr and given Pc from adaptive pose and return Pr after optimization

        P_c = P_c.clone().requires_grad_()
        with torch.enable_grad():
            if self.P_r_init is None:
                P_r = torch.ones_like(P_c)
            else:
                P_r = self.P_r_init
            optimizer = torch.optim.LBFGS(
                [P_r], 
                lr=1, 
                max_iter=20, 
                line_search_fn='strong_wolfe')

            def closure():
                optimizer.zero_grad()
                f = self.objective(P_c, P_r)
                f.backward(retain_graph=True)
                return f

            optimizer.step(closure)

        # return output, ctx
        return P_r.detach(), None

    def objective(self, P_c, y):
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
        # G_x is [B,H,W,2]
        # N_x is [B,H,W]
        # A and B are [B,H,W,2,3]
        # V and Omega are [B,3]

        # compute P_c from adaptive pose function
        P_r = y.detach().clone()
        P_c = P_c.detach().clone().requires_grad_() # use as x value for lower level cost
        optimizer_ddn = torch.optim.LBFGS([P_c], lr=1, max_iter=20, line_search_fn='strong_wolfe')
        def closure():
            optimizer_ddn.zero_grad()
            loss = AdaptivePoseUpperCost(P_r, P_c, self.N_x, self.G_x, self.A, self.B)/1e6
            loss.backward(retain_graph=True) #TODO: this is causing loss to become Nonetype, make sure everything is detached
        optimizer_ddn.step(closure)

        # compute objective of bottom level
        # fwd pass so that gradient of dPr/dPc computed
        V_coarse = P_c[:,0].detach().clone()
        omega_coarse = P_c[:,1].detach().clone()
        cost = torch.einsum('bhwi,bhwij,bj->bhw', self.G_x, self.A, V_coarse) \
              * (self.N_x - torch.einsum('bhwi,bhwij,bj->bhw', self.G_x, self.B, omega_coarse)) # [B,H,W]
        smooth_cost = -F.gelu(cost) # twice differentiable and bias towards positive value, enforcing constraint
        total_cost = torch.mean(smooth_cost) # take expected value
        print(total_cost)
        return total_cost


def AdaptivePoseUpperCost(P_r, P_c, N_x, G_x, A, B):
    V_refined = P_r[:,0]
    omega_refined = P_r[:,1]
    V_coarse = P_c[:,0]
    omega_coarse = P_c[:,1]

    numerator = N_x - torch.einsum('bhwi,bhwij,bj->bhw', G_x, B, omega_refined) # bchw bc N_x is bchw and is subtracted from
    denominator = torch.einsum('bhwi,bhwij,bj->bhw', G_x, A, V_refined) # bchw
    quotient = numerator/denominator
    quotient[torch.isnan(quotient)] = 0
    quotient[torch.isinf(quotient)] = 0
    coarse_error = torch.einsum('bhw,bhwij,bj->bhwi', quotient, A, V_coarse) - torch.einsum('bhwij,bj->bhwi', B, omega_coarse) # bchw2
    cost = N_x - torch.einsum('bhwi,bhwi->bhw', G_x, coarse_error) # [N]
    total_cost = torch.mean(cost) # TODO might need to take the average, but does it matter? only difference is that it'll take longer to converge
    print(f"upper cost: {total_cost}")
    return total_cost