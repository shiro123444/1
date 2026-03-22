"""
HierNBeats: Hierarchical Neural Basis Expansion Analysis
Based on paper: Sun et al. 2024 (ICANN 2024)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NBeatsBlock(nn.Module):
    """Basic NBeats block with doubly residual stacking"""

    def __init__(self, input_size, theta_size, basis_function, layers, layer_size):
        super(NBeatsBlock, self).__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function
        self.layers = layers
        self.layer_size = layer_size

        # Fully connected layers
        self.fc_layers = nn.ModuleList([nn.Linear(input_size, layer_size)] +
                                       [nn.Linear(layer_size, layer_size) for _ in range(layers - 1)])

        # Theta layers for backcast and forecast
        self.theta_b = nn.Linear(layer_size, theta_size)
        self.theta_f = nn.Linear(layer_size, theta_size)

    def forward(self, x):
        """
        Args:
            x: input [batch, input_size]
        Returns:
            backcast: [batch, input_size]
            forecast: [batch, output_size]
        """
        # Fully connected stack
        h = x
        for layer in self.fc_layers:
            h = F.relu(layer(h))

        # Generate theta parameters
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)

        # Apply basis functions
        backcast = self.basis_function(theta_b, is_forecast=False)
        forecast = self.basis_function(theta_f, is_forecast=True)

        return backcast, forecast


class HierarchicalBasis(nn.Module):
    """Hierarchical neural basis for interpretable forecasting"""

    def __init__(self, backcast_size, forecast_size, hierarchy_structure):
        super(HierarchicalBasis, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.hierarchy_structure = hierarchy_structure

    def forward(self, theta, is_forecast=False):
        """
        Generate basis expansion
        Args:
            theta: parameters [batch, theta_size]
            is_forecast: whether generating forecast or backcast
        Returns:
            basis expansion [batch, size]
        """
        size = self.forecast_size if is_forecast else self.backcast_size
        # Simple linear basis (can be extended to trend/seasonality)
        return theta[:, :size]


class GenericBasis(nn.Module):
    """Generic basis function for flexible learning"""

    def __init__(self, backcast_size, forecast_size):
        super(GenericBasis, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta, is_forecast=False):
        size = self.forecast_size if is_forecast else self.backcast_size
        return theta[:, :size]


class TrendBasis(nn.Module):
    """Trend basis with polynomial expansion"""

    def __init__(self, backcast_size, forecast_size, degree=3):
        super(TrendBasis, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.degree = degree

    def forward(self, theta, is_forecast=False):
        size = self.forecast_size if is_forecast else self.backcast_size
        batch_size = theta.size(0)

        # Polynomial basis
        t = torch.arange(0, size, dtype=torch.float32, device=theta.device) / size
        T = torch.stack([t ** i for i in range(self.degree + 1)], dim=0)  # [degree+1, size]

        # theta: [batch, degree+1]
        theta_trunc = theta[:, :self.degree + 1]
        return torch.matmul(theta_trunc, T)  # [batch, size]


class SeasonalityBasis(nn.Module):
    """Seasonality basis with Fourier expansion"""

    def __init__(self, backcast_size, forecast_size, num_harmonics=5):
        super(SeasonalityBasis, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.num_harmonics = num_harmonics

    def forward(self, theta, is_forecast=False):
        size = self.forecast_size if is_forecast else self.backcast_size
        batch_size = theta.size(0)

        # Fourier basis
        t = torch.arange(0, size, dtype=torch.float32, device=theta.device) / size
        S = []
        for i in range(1, self.num_harmonics + 1):
            S.append(torch.sin(2 * np.pi * i * t))
            S.append(torch.cos(2 * np.pi * i * t))
        S = torch.stack(S, dim=0)  # [2*num_harmonics, size]

        # theta: [batch, 2*num_harmonics]
        theta_trunc = theta[:, :2 * self.num_harmonics]
        return torch.matmul(theta_trunc, S)  # [batch, size]


class HierarchicalStack(nn.Module):
    """Multi-branch stack for hierarchical structure"""

    def __init__(self, input_size, output_size, num_blocks, layer_size,
                 basis_type='generic', hierarchy_info=None):
        super(HierarchicalStack, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_blocks = num_blocks

        # Create basis function
        if basis_type == 'trend':
            basis_fn = lambda: TrendBasis(input_size, output_size, degree=3)
            theta_size = 4  # degree + 1
        elif basis_type == 'seasonality':
            basis_fn = lambda: SeasonalityBasis(input_size, output_size, num_harmonics=5)
            theta_size = 10  # 2 * num_harmonics
        else:  # generic
            basis_fn = lambda: GenericBasis(input_size, output_size)
            theta_size = max(input_size, output_size)

        # Create blocks
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, theta_size, basis_fn(), layers=4, layer_size=layer_size)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Args:
            x: input [batch, input_size]
        Returns:
            residual: [batch, input_size]
            forecast: [batch, output_size]
        """
        residual = x
        forecast = torch.zeros(x.size(0), self.output_size, device=x.device)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return residual, forecast


class HierNBeats(nn.Module):
    """
    Hierarchical Neural Basis Expansion Analysis
    Multi-branch structure for hierarchical time series forecasting
    """

    def __init__(self, backcast_length, forecast_length, hierarchy_structure,
                 hierarchy_matrix, num_stacks=3, num_blocks=3, layer_size=512,
                 interpretable=True):
        """
        Args:
            backcast_length: input sequence length
            forecast_length: output forecast horizon
            hierarchy_structure: dict with level information
            hierarchy_matrix: S matrix for coherence [N x M]
            num_stacks: number of stacks per branch
            num_blocks: number of blocks per stack
            layer_size: hidden layer size
            interpretable: use interpretable basis (trend/seasonality)
        """
        super(HierNBeats, self).__init__()

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.hierarchy_structure = hierarchy_structure
        self.interpretable = interpretable

        # Register S matrix
        self.register_buffer('S', torch.FloatTensor(hierarchy_matrix))

        # Multi-branch structure for different levels
        self.branches = nn.ModuleDict()

        # Individual forecast branch (generic)
        self.branches['individual'] = HierarchicalStack(
            backcast_length, forecast_length, num_blocks, layer_size,
            basis_type='generic'
        )

        if interpretable:
            # Top-down branch (trend)
            self.branches['topdown'] = HierarchicalStack(
                backcast_length, forecast_length, num_blocks, layer_size,
                basis_type='trend'
            )

            # Bottom-up branch (seasonality)
            self.branches['bottomup'] = HierarchicalStack(
                backcast_length, forecast_length, num_blocks, layer_size,
                basis_type='seasonality'
            )
        else:
            # All generic branches
            self.branches['topdown'] = HierarchicalStack(
                backcast_length, forecast_length, num_blocks, layer_size,
                basis_type='generic'
            )
            self.branches['bottomup'] = HierarchicalStack(
                backcast_length, forecast_length, num_blocks, layer_size,
                basis_type='generic'
            )

        # Aggregation weights
        self.branch_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        """
        Args:
            x: input [batch, backcast_length, num_series]
        Returns:
            forecast: coherent forecast [batch, forecast_length, num_series]
            components: dict of interpretable components
        """
        batch_size = x.size(0)
        num_series = x.size(2)

        # Flatten input for processing
        x_flat = x.view(batch_size * num_series, self.backcast_length)

        # Process through each branch
        forecasts = {}
        residuals = {}

        for branch_name, branch in self.branches.items():
            res, fc = branch(x_flat)
            residuals[branch_name] = res
            forecasts[branch_name] = fc

        # Weighted combination
        weights = F.softmax(self.branch_weights, dim=0)
        combined_forecast = (
            weights[0] * forecasts['individual'] +
            weights[1] * forecasts['topdown'] +
            weights[2] * forecasts['bottomup']
        )

        # Reshape to [batch, forecast_length, num_series]
        combined_forecast = combined_forecast.view(batch_size, num_series, self.forecast_length)
        combined_forecast = combined_forecast.transpose(1, 2)

        # Ensure coherence using S matrix
        coherent_forecast = self.reconcile(combined_forecast)

        # Return components for interpretability
        components = {
            'individual': forecasts['individual'].view(batch_size, num_series, self.forecast_length).transpose(1, 2),
            'topdown': forecasts['topdown'].view(batch_size, num_series, self.forecast_length).transpose(1, 2),
            'bottomup': forecasts['bottomup'].view(batch_size, num_series, self.forecast_length).transpose(1, 2),
            'weights': weights
        }

        return coherent_forecast, components

    def reconcile(self, forecast):
        """
        Reconcile forecasts to ensure coherence
        Args:
            forecast: [batch, forecast_length, num_series]
        Returns:
            coherent_forecast: [batch, forecast_length, num_series]
        """
        batch_size = forecast.size(0)
        forecast_length = forecast.size(1)

        # Extract bottom level forecasts
        num_bottom = self.S.size(1)
        bottom_forecast = forecast[:, :, -num_bottom:]

        # Reconstruct hierarchy using S matrix
        coherent = torch.matmul(bottom_forecast, self.S.t())

        return coherent

    def get_interpretable_components(self, x):
        """
        Get interpretable forecast components
        Args:
            x: input [batch, backcast_length, num_series]
        Returns:
            components: dict with trend, seasonality, and individual forecasts
        """
        with torch.no_grad():
            _, components = self.forward(x)
        return components
