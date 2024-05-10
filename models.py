import torch
from torch.autograd import Function
import torch.nn.functional as F
from torchvision.models import vgg16
import numpy as np

from ddn import AbstractDeclarativeNode, ComposedNode


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


class ChiralityNode(AbstractDeclarativeNode):
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
        total_cost = torch.mean(smooth_cost) # take expected value
        return total_cost - y
    
    def solve(self, V_refined, omega_refined):
        V = V_refined.clone().requires_grad_()
        omega = omega_refined.clone().requires_grad_()
        y = self._run_optimization(V, omega)
        return y
        
    def _run_optimization(self, V, omega):
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
        return y.detach(), None


class AdaptivePoseNode(AbstractDeclarativeNode):
    """
    Node that solves Adaptive pose optimization problem
    """
    def __init__(self):
        super().__init__()

    def __init__(self, N_x, G_x, A, B):
        super().__init__()
        self.N_x = N_x # Magnitude of the normal flow for all pixels
        self.G_x = G_x # The image gradients for all pixels
        self.A = A # Matrix for the motion flow field due to translation
        self.B = B # Matrix for the motion flow field due to rotation
        chirality_node = ChiralityNode(N_x, G_x, A, B)
        self.chirality_net = DeclarativeLayer(chirality_node)
        #TODO: adjust eps based on convergence

    def objective(self, *xs, y=0):
        """
        Compute the upper-level global objective function for optimization.
        
        Args:
        - V_coarse: Global translational velocity vector.
        - omega_coarse: Global rotational velocity vector.
        - V_refined: Refined translational velocity vector.
        - omega_refined: Refined rotational velocity vector.
        - G_x: Image gradient direction for all pixels/keypoints.
        - A: Global matrix A_tilde from the equation.
        - B: Global matrix B_tilde from the equation.

        Returns:
        - The value of the upper-level global objective function.
        """
        # Compute n_x - g_x' * B_tilde * Omega_tilde globally
        # Note: g_x, B_tilde, and A_tilde should be formulated to handle the entire image or set of keypoints
        V_refined, omega_refined, V_coarse_init, omega_coarse_init = xs # inputs
        numerator = self.N_x - torch.einsum('bhwi,bhwij,bj->bhw', self.G_x, self.B, omega_refined) # bchw bc N_x is bchw and is subtracted from
        denominator = torch.einsum('bhwi,bhwij,bj->bhw', self.G_x, self.A, V_refined) # bchw
        quotient = numerator/denominator
        quotient[torch.isnan(quotient)] = 0
        quotient[torch.isinf(quotient)] = 0
        coarse_error = torch.einsum('bhw,bhwij,bj->bhwi', quotient, self.A, V_coarse_init) - torch.einsum('bhwij,bj->bhwi', self.B, omega_coarse_init) # bchw2
        cost = self.N_x - torch.einsum('bhwi,bhwi->bhw', self.G_x, coarse_error) # [N]
        total_cost = torch.mean(cost) # TODO might need to take the average, but does it matter? only difference is that it'll take longer to converge
        return total_cost - y
    
    def solve(self, *xs):
        V_refined, omega_refined, V_coarse, omega_coarse = xs
        optimizer = torch.optim.LBFGS([V_coarse.detach(), omega_coarse.detach()], lr=1, max_iter=20, line_search_fn='strong_wolfe')

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad() # Zero out gradients

            # pose_refined = self.chirality_net(V_coarse, omega_coarse)
            # V_refined = pose_refined[:,0]
            # omega_refined = pose_refined[:,1]
            
            loss = self.objective(V_refined.detach(), omega_refined.detach(), V_coarse, omega_coarse)

            # if loss.requires_grad: #TODO: is this necessary?
            #     loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)
        return torch.stack([V_coarse, omega_coarse], dim=1), None


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
    

class AdaptivePoseNode2(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class ComposedNode(AbstractDeclarativeNode):
    """Composes two deep declarative nodes f and g to produce y = g(f(x)). The resulting composition
    behaves exactly like a single deep declarative node, that is, it has the same interface and
    as such can be further composed with other nodes to form a chain."""

    def __init__(self, nodeA, nodeB):
        assert (nodeA.dim_y == nodeB.dim_x)
        super().__init__(nodeA.dim_x, nodeB.dim_y)
        self.nodeA = nodeA
        self.nodeB = nodeB

    def solve(self, x):
        """Overrides the solve method to first compute z = nodeA.solve(x) and then y = nodeB.solve(z). Returns
        the dual variables from both nodes."""
        z, ctxA = self.nodeA.solve(x)
        y, ctxB = self.nodeB.solve(z)
        return y, {'ctxA': ctxA, 'ctxB': ctxB, 'z': z}

    def gradient(self, x, y=None, ctx=None):
        """Overrides the gradient method to compute the composed gradient by the chain rule."""

        # we need to resolve for z since there is currently no way to store this
        if ctx is None:
            z, _ = self.nodeA.solve(x)
        else:
            z = ctx['z']
        Dz = self.nodeA.gradient(x, z).reshape(self.nodeA.dim_y, self.nodeA.dim_x)
        Dy = self.nodeB.gradient(z, y).reshape(self.nodeB.dim_y, self.nodeB.dim_x)
        return np.dot(Dy, Dz)