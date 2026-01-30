"""
Utility functions for YOLO Plant Detection System
Professional utilities for data processing, system monitoring, and model management
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import psutil
import GPUtil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('plant_detection.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources during training"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA available: {self.gpu_available}, GPU count: {self.gpu_count}")
        else:
            logger.warning("CUDA not available, using CPU")
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'cuda_available': self.gpu_available,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                gpu_info = []
                for i, gpu in enumerate(gpus):
                    gpu_info.append({
                        'id': i,
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_free_mb': gpu.memoryFree,
                        'temperature': gpu.temperature,
                        'load': gpu.load * 100 if gpu.load else 0
                    })
                info['gpus'] = gpu_info
            except Exception as e:
                logger.warning(f"Could not get GPU info: {e}")
        
        return info
    
    def log_system_status(self):
        """Log current system status"""
        info = self.get_system_info()
        logger.info(f"System Status: {json.dumps(info, indent=2)}")
        return info


class ConfigManager:
    """Manage configuration files and parameters"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def save_config(self, config: Dict, path: str = None):
        """Save configuration to YAML file"""
        if path is None:
            path = self.config_path
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key: str, value):
        """Update configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value


class DatasetAnalyzer:
    """Analyze dataset structure and statistics"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.classes = []
        self.class_counts = {}
        self.total_images = 0
    
    def analyze_dataset(self) -> Dict:
        """Analyze dataset and return statistics"""
        logger.info(f"Analyzing dataset at {self.dataset_path}")
        
        splits = ['train', 'val', 'test']
        analysis = {}
        
        for split in splits:
            split_path = self.dataset_path / split
            if split_path.exists():
                analysis[split] = self._analyze_split(split_path)
            else:
                logger.warning(f"Split {split} not found at {split_path}")
        
        # Get all unique classes
        all_classes = set()
        for split_data in analysis.values():
            all_classes.update(split_data['classes'])
        
        self.classes = sorted(list(all_classes))
        analysis['all_classes'] = self.classes
        analysis['num_classes'] = len(self.classes)
        
        logger.info(f"Dataset analysis complete. Found {len(self.classes)} classes")
        return analysis
    
    def _analyze_split(self, split_path: Path) -> Dict:
        """Analyze a specific dataset split"""
        classes = []
        class_counts = {}
        total_images = 0
        
        for class_dir in split_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                classes.append(class_name)
                
                # Count images in class directory
                image_count = len([f for f in class_dir.iterdir() 
                                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                class_counts[class_name] = image_count
                total_images += image_count
        
        return {
            'classes': classes,
            'class_counts': class_counts,
            'total_images': total_images,
            'num_classes': len(classes)
        }
    
    def create_class_mapping(self) -> Dict[str, int]:
        """Create class name to index mapping"""
        return {class_name: idx for idx, class_name in enumerate(self.classes)}
    
    def save_analysis(self, output_path: str = "dataset_analysis.json"):
        """Save dataset analysis to JSON file"""
        analysis = self.analyze_dataset()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset analysis saved to {output_path}")
        return analysis


class ModelUtils:
    """Utility functions for model management"""
    
    @staticmethod
    def get_device() -> torch.device:
        """Get the best available device (CUDA if available, else CPU)"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU")
        
        return device
    
    @staticmethod
    def get_optimal_batch_size(model_size: str = "n", device: torch.device = None) -> int:
        """Get optimal batch size based on model size and available memory"""
        if device is None:
            device = ModelUtils.get_device()
        
        if device.type == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Batch size recommendations based on GPU memory and model size
            if gpu_memory >= 12:  # RTX 4070 Super has 12GB
                batch_sizes = {"n": 64, "s": 32, "m": 16, "l": 8, "x": 4}
            elif gpu_memory >= 8:
                batch_sizes = {"n": 32, "s": 16, "m": 8, "l": 4, "x": 2}
            else:
                batch_sizes = {"n": 16, "s": 8, "m": 4, "l": 2, "x": 1}
            
            return batch_sizes.get(model_size, 16)
        else:
            # CPU batch sizes
            return {"n": 8, "s": 4, "m": 2, "l": 1, "x": 1}.get(model_size, 4)
    
    @staticmethod
    def count_parameters(model) -> int:
        """Count total parameters in model"""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def count_trainable_parameters(model) -> int:
        """Count trainable parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class VisualizationUtils:
    """Utility functions for visualization and plotting"""
    
    @staticmethod
    def plot_training_history(history: Dict, save_path: str = "training_history.png"):
        """Plot training history metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Plot loss
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss')
            axes[0, 0].plot(history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot accuracy
        if 'train_acc' in history and 'val_acc' in history:
            axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
            axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot learning rate
        if 'lr' in history:
            axes[1, 0].plot(history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        # Plot system resources (if available)
        if 'gpu_memory' in history:
            axes[1, 1].plot(history['gpu_memory'])
            axes[1, 1].set_title('GPU Memory Usage')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Memory (MB)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training history plot saved to {save_path}")
    
    @staticmethod
    def plot_confusion_matrix(y_true: List, y_pred: List, class_names: List[str], 
                            save_path: str = "confusion_matrix.png"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")
    
    @staticmethod
    def plot_class_distribution(class_counts: Dict, save_path: str = "class_distribution.png"):
        """Plot class distribution"""
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(classes)), counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Images')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Class distribution plot saved to {save_path}")


class DataUtils:
    """Utility functions for data processing"""
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """Validate if image file is readable and not corrupted"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_image_info(image_path: str) -> Dict:
        """Get image information"""
        try:
            with Image.open(image_path) as img:
                return {
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'filename': os.path.basename(image_path)
                }
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            return {}
    
    @staticmethod
    def resize_image(image_path: str, target_size: Tuple[int, int], 
                    output_path: str = None) -> str:
        """Resize image to target size"""
        try:
            with Image.open(image_path) as img:
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                
                if output_path is None:
                    output_path = image_path
                
                img_resized.save(output_path, quality=95)
                return output_path
        except Exception as e:
            logger.error(f"Error resizing image {image_path}: {e}")
            return image_path


def setup_experiment(experiment_name: str = None) -> str:
    """Setup experiment directory and logging"""
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    experiment_dir = Path("runs") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup experiment-specific logging
    log_file = experiment_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    logger.info(f"Experiment setup complete: {experiment_dir}")
    return str(experiment_dir)


def cleanup_old_runs(keep_last: int = 5):
    """Clean up old experiment runs, keeping only the most recent ones"""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return
    
    experiments = [d for d in runs_dir.iterdir() if d.is_dir()]
    experiments.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for exp_dir in experiments[keep_last:]:
        try:
            import shutil
            shutil.rmtree(exp_dir)
            logger.info(f"Cleaned up old experiment: {exp_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up {exp_dir}: {e}")


if __name__ == "__main__":
    # Test utilities
    monitor = SystemMonitor()
    monitor.log_system_status()
    
    analyzer = DatasetAnalyzer("Plants_Datadet")
    analysis = analyzer.analyze_dataset()
    print(f"Dataset analysis: {analysis}") 