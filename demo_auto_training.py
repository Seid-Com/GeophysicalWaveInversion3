
#!/usr/bin/env python3
"""
Demo script showing automatic training detection
Run this to see how the system detects changes and triggers retraining
"""

import time
import sys
sys.path.append('.')

from utils.auto_training_monitor import AutoTrainingManager
from trainer import FWITrainer
from models.physics_guided_network import PhysicsGuidedFWI

def demo_auto_training():
    """Demonstrate automatic training detection"""
    print("🚀 Auto-Training Detection Demo")
    print("=" * 50)
    
    # Create a simple model and trainer for demo
    model = PhysicsGuidedFWI(
        input_channels=1,
        output_channels=1,
        encoder_channels=[32, 64],
        decoder_channels=[64, 32]
    )
    
    trainer = FWITrainer(
        model=model,
        learning_rate=0.001,
        batch_size=4,
        num_epochs=5
    )
    
    # Initialize auto-training manager
    auto_manager = AutoTrainingManager()
    
    print("🔧 Enabling auto-training...")
    auto_manager.enable_auto_training(trainer, "sample_data/train_samples")
    
    print("\n📋 Instructions:")
    print("1. This script is now monitoring for changes")
    print("2. Try modifying files in sample_data/train_samples/")
    print("3. Add new .npy files or modify existing ones")
    print("4. Watch the console for automatic detection")
    print("5. Press Ctrl+C to stop monitoring")
    
    try:
        # Keep the script running to monitor changes
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping auto-training monitor...")
        auto_manager.disable_auto_training()
        print("✅ Demo completed")

if __name__ == "__main__":
    demo_auto_training()
