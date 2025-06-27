
import time
import os
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AutoTrainingManager:
    """Manages automatic training detection and restart capabilities"""
    
    def __init__(self):
        self.observer = None
        self.trainer = None
        self.data_dir = None
        self.is_monitoring = False
        self.config_file = "auto_training_config.json"
        self.last_training_time = 0
        
    def enable_auto_training(self, trainer, data_directory: str):
        """Enable automatic training monitoring"""
        self.trainer = trainer
        self.data_dir = data_directory
        
        if not os.path.exists(data_directory):
            raise ValueError(f"Data directory {data_directory} does not exist")
            
        # Set up file system monitoring
        event_handler = TrainingFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, data_directory, recursive=True)
        self.observer.start()
        self.is_monitoring = True
        
        print(f"🔄 Auto-training enabled for {data_directory}")
        
    def disable_auto_training(self):
        """Disable automatic training monitoring"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            
        self.is_monitoring = False
        print("🛑 Auto-training disabled")
        
    def trigger_training(self, reason: str = "File change detected"):
        """Trigger automatic training"""
        current_time = time.time()
        
        # Prevent too frequent retraining (minimum 30 seconds between runs)
        if current_time - self.last_training_time < 30:
            return
            
        print(f"🔄 Auto-training triggered: {reason}")
        self.last_training_time = current_time
        
        # In a real implementation, this would trigger training
        # For now, just log the event
        
    def save_current_config(self, config: Dict[str, Any]):
        """Save current training configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    def load_config(self) -> Dict[str, Any]:
        """Load saved training configuration"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
            
    def compare_with_competition(self, current_mae: float) -> Dict[str, Any]:
        """Compare current performance with competition benchmarks"""
        competition_scores = {
            'DumberNet': 213.7,
            'Seismic Wave Inversion': 85.7,  # Best score
            'OpenFWI-GANs': 287.5,
            'PhysicsNet': 156.2,
            'WaveNet-FWI': 198.1
        }
        
        ranking = 1
        for model, score in competition_scores.items():
            if current_mae > score:
                ranking += 1
                
        target_score = 85.7  # Best competition score
        improvement_needed = max(0, current_mae - target_score)
        
        return {
            'current_mae': current_mae,
            'ranking': ranking,
            'total_models': len(competition_scores) + 1,
            'target_score': target_score,
            'improvement_needed': improvement_needed,
            'target_met': current_mae <= target_score,
            'competition_scores': competition_scores
        }

class TrainingFileHandler(FileSystemEventHandler):
    """Handles file system events for auto-training"""
    
    def __init__(self, manager: AutoTrainingManager):
        self.manager = manager
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.npy'):
            self.manager.trigger_training(f"Modified: {os.path.basename(event.src_path)}")
            
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.npy'):
            self.manager.trigger_training(f"Created: {os.path.basename(event.src_path)}")

class AutoTrainingMonitor:
    """Monitor training performance and provide recommendations"""
    
    def __init__(self):
        self.metrics_history = []
        
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics for monitoring"""
        self.metrics_history.append({
            'epoch': epoch,
            'timestamp': time.time(),
            **metrics
        })
        
    def get_training_recommendations(self) -> List[str]:
        """Get recommendations based on training performance"""
        recommendations = []
        
        if len(self.metrics_history) > 10:
            recent_losses = [m['val_loss'] for m in self.metrics_history[-10:]]
            
            # Check for stagnation
            if max(recent_losses) - min(recent_losses) < 0.001:
                recommendations.append("Consider reducing learning rate - loss appears to have plateaued")
                
            # Check for instability
            if any(abs(recent_losses[i] - recent_losses[i-1]) > 0.1 for i in range(1, len(recent_losses))):
                recommendations.append("Training appears unstable - consider gradient clipping or lower learning rate")
                
        return recommendations
        
    def compare_with_competition(self, current_mae: float) -> Dict[str, Any]:
        """Compare with competition benchmarks"""
        return auto_training_manager.compare_with_competition(current_mae)

# Create global instance
auto_training_manager = AutoTrainingManager()
