
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimplifiedFWI(nn.Module):
    """
    Simplified Physics-Guided FWI Network
    Based on competition analysis - simpler models perform better
    """
    
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        encoder_channels=[32, 64, 128],  # Reduced complexity
        decoder_channels=[128, 64, 32],
        physics_weight=0.05,  # Reduced physics weight
        smoothness_weight=0.01
    ):
        super(SimplifiedFWI, self).__init__()
        
        self.physics_weight = physics_weight
        self.smoothness_weight = smoothness_weight
        
        # Simplified encoder
        self.seismic_encoder = self._build_simplified_encoder(input_channels, encoder_channels)
        
        # Direct spatial processor
        self.spatial_processor = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),
            nn.Conv2d(encoder_channels[-1], encoder_channels[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Simplified decoder
        self.velocity_decoder = self._build_simplified_decoder(encoder_channels[-1], decoder_channels, output_channels)
        
        # Minimal physics constraints
        self.min_velocity = 1500.0
        self.max_velocity = 6000.0
        
    def _build_simplified_encoder(self, input_channels, encoder_channels):
        """Simplified encoder with fewer layers"""
        layers = []
        in_channels = input_channels
        
        for out_channels in encoder_channels:
            layers.extend([
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_simplified_decoder(self, input_channels, decoder_channels, output_channels):
        """Simplified decoder"""
        layers = []
        in_channels = input_channels
        
        for out_channels in decoder_channels:
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        # Final output layer with sigmoid for velocity range
        layers.append(nn.Conv2d(in_channels, output_channels, kernel_size=1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def forward(self, seismic_data):
        """Forward pass"""
        if len(seismic_data.shape) == 4:
            seismic_data = seismic_data.unsqueeze(1)
        
        # Encode
        encoded = self.seismic_encoder(seismic_data)
        
        # Process spatially
        spatial = encoded.squeeze(2)  # Remove time dimension
        processed = self.spatial_processor(spatial)
        
        # Decode to velocity
        velocity_normalized = self.velocity_decoder(processed)
        
        # Scale to velocity range
        velocity_map = velocity_normalized * (self.max_velocity - self.min_velocity) + self.min_velocity
        
        return velocity_map
    
    def compute_physics_loss(self, velocity_map, seismic_data=None):
        """Simplified physics loss - only smoothness"""
        # Simple total variation loss for smoothness
        dx = torch.abs(velocity_map[:, :, :, 1:] - velocity_map[:, :, :, :-1])
        dy = torch.abs(velocity_map[:, :, 1:, :] - velocity_map[:, :, :-1, :])
        
        smoothness_loss = torch.mean(dx) + torch.mean(dy)
        
        return self.smoothness_weight * smoothness_loss

class SimplifiedFWILoss(nn.Module):
    """Simplified loss function"""
    
    def __init__(self, data_weight=1.0, physics_weight=0.05):
        super(SimplifiedFWILoss, self).__init__()
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.mae_loss = nn.L1Loss()  # Focus on MAE since that's competition metric
    
    def forward(self, predicted_velocity, target_velocity, model, seismic_data=None):
        """Compute loss with focus on MAE"""
        # Primary data loss (MAE)
        data_loss = self.mae_loss(predicted_velocity, target_velocity)
        
        # Minimal physics loss
        physics_loss = model.compute_physics_loss(predicted_velocity, seismic_data)
        
        # Combined loss
        total_loss = self.data_weight * data_loss + self.physics_weight * physics_loss
        
        loss_components = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item()
        }
        
        return total_loss, loss_components
