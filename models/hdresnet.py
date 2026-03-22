"""
HDResNet: Hierarchical-Decomposition Residual Network
Based on paper: Xiang et al. 2023
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HDResBlock(nn.Module):
    """HDResNet Block with LSTM and Top-Down decomposition"""

    def __init__(self, input_size, hidden_size, level_structure, dropout=0.1):
        super(HDResBlock, self).__init__()
        self.level_structure = level_structure  # [num_level1, num_level2, ...]

        # LSTM for residual forecasting
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # Linear layers for proportion forecasting at each level
        self.proportion_layers = nn.ModuleList()
        for num_children in level_structure:
            if num_children > 0:
                self.proportion_layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_size, num_children),
                        nn.Softmax(dim=-1)
                    )
                )

        # Residual projection (keep in hidden space)
        self.residual_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, parent_forecast=None):
        """
        Args:
            x: input tensor [batch, seq_len, features]
            parent_forecast: parent level forecast for top-down
        Returns:
            residual_forecast: updated forecast
            proportions: list of proportion forecasts
        """
        batch_size = x.size(0)

        # LSTM for residual forecasting
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Take last timestep

        # Generate proportions for each level
        proportions = []
        for prop_layer in self.proportion_layers:
            prop = prop_layer(lstm_out)
            proportions.append(prop)

        # Residual forecast
        residual = self.residual_proj(lstm_out)

        return residual, proportions


class HDResNet(nn.Module):
    """
    Hierarchical-Decomposition Residual Network
    End-to-end top-down hierarchical forecasting
    """

    def __init__(self, input_size, hidden_size, num_blocks, level_structure,
                 hierarchy_matrix, output_horizon=1, dropout=0.1):
        """
        Args:
            input_size: number of input features
            hidden_size: hidden dimension
            num_blocks: number of residual blocks
            level_structure: list of children counts at each level
            hierarchy_matrix: S matrix for coherence [N x M]
            output_horizon: forecasting horizon
        """
        super(HDResNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.level_structure = level_structure
        self.output_horizon = output_horizon

        # Register S matrix as buffer
        self.register_buffer('S', torch.FloatTensor(hierarchy_matrix))

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Stacked HDResBlocks
        self.blocks = nn.ModuleList([
            HDResBlock(hidden_size, hidden_size, level_structure, dropout)
            for _ in range(num_blocks)
        ])

        # Output layer for final prediction
        self.output_layer = nn.Linear(hidden_size, input_size * output_horizon)

    def top_down_update(self, parent_forecast, proportions, level_structure):
        """
        Top-down method to update children forecasts
        Args:
            parent_forecast: forecast of parent nodes
            proportions: proportion for disaggregation
            level_structure: structure of hierarchy
        Returns:
            children_forecast: updated children forecasts
        """
        children_forecasts = []
        start_idx = 0

        for i, num_children in enumerate(level_structure):
            if num_children > 0:
                parent = parent_forecast[:, start_idx:start_idx+1]
                prop = proportions[i]
                children = parent * prop
                children_forecasts.append(children)
                start_idx += 1

        if children_forecasts:
            return torch.cat(children_forecasts, dim=1)
        return parent_forecast

    def forward(self, x):
        """
        Args:
            x: input tensor [batch, seq_len, features]
        Returns:
            forecast: coherent hierarchical forecast [batch, features]
        """
        batch_size = x.size(0)

        # Initial projection
        h = self.input_proj(x[:, -1, :])  # Use last timestep
        h = h.unsqueeze(1)  # [batch, 1, hidden]

        # Residual blocks
        for block in self.blocks:
            residual, proportions = block(h)
            h = h + residual.unsqueeze(1)

        # Final output
        output = self.output_layer(h.squeeze(1))
        output = output.view(batch_size, self.output_horizon, self.input_size)

        # Ensure strict coherence using S matrix
        if self.output_horizon == 1:
            forecast = output.squeeze(1)
            # Project to bottom level and reconstruct
            bottom_forecast = forecast[:, -self.S.size(1):]
            coherent_forecast = torch.matmul(bottom_forecast, self.S.t())
            return coherent_forecast

        return output

    def weighted_loss(self, pred, target, level_weights):
        """
        Weighted loss based on coefficient of variation
        Args:
            pred: predictions [batch, features]
            target: ground truth [batch, features]
            level_weights: weights for each level
        Returns:
            weighted loss
        """
        mse = F.mse_loss(pred, target, reduction='none')

        # Apply level weights
        if level_weights is not None:
            level_weights = level_weights.to(pred.device)
            weighted_mse = mse * level_weights.unsqueeze(0)
            return weighted_mse.mean()

        return mse.mean()
