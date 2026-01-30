"""
Professional YOLO Trainer for Plant Detection
Advanced training pipeline with overfitting prevention and industry-standard practices
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device

from .utils import (
    SystemMonitor, ConfigManager, DatasetAnalyzer, ModelUtils, 
    VisualizationUtils, setup_experiment, cleanup_old_runs
)

logger = logging.getLogger(__name__)


class YOLOTrainer:
    """
    Professional YOLO Trainer with advanced features:
    - Overfitting prevention
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Comprehensive logging
    - Performance optimization
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.system_monitor = SystemMonitor()
        self.device = ModelUtils.get_device()
        
        # Training state
        self.model = None
        self.trainer = None
        self.experiment_dir = None
        self.best_model_path = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': [],
            'gpu_memory': []
        }
        
        # Setup experiment
        self.setup_experiment()
        
        # Analyze dataset
        self.dataset_analyzer = DatasetAnalyzer(self.config['dataset']['path'])
        self.dataset_analysis = self.dataset_analyzer.analyze_dataset()
        
        # Update config with dataset info
        self.config_manager.update('dataset.nc', self.dataset_analysis['num_classes'])
        self.config_manager.update('dataset.names', self.dataset_analysis['all_classes'])
        
        logger.info(f"YOLO Trainer initialized with {self.dataset_analysis['num_classes']} classes")
    
    def setup_experiment(self):
        """Setup experiment directory and logging"""
        experiment_name = f"plant_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = setup_experiment(experiment_name)
        
        # Save configuration
        config_save_path = Path(self.experiment_dir) / "config.yaml"
        self.config_manager.save_config(self.config, str(config_save_path))
        
        # Save system info
        system_info = self.system_monitor.get_system_info()
        system_info_path = Path(self.experiment_dir) / "system_info.json"
        with open(system_info_path, 'w') as f:
            json.dump(system_info, f, indent=2)
        
        logger.info(f"Experiment setup complete: {self.experiment_dir}")
    
    def prepare_dataset(self) -> Dict:
        """Prepare dataset for YOLO training"""
        logger.info("Preparing dataset for YOLO training...")
        
        # Get absolute paths
        dataset_base_path = Path(self.config['dataset']['path']).resolve()
        
        # Create YOLO dataset configuration
        dataset_config = {
            'path': str(dataset_base_path),
            'train': str(dataset_base_path / self.config['dataset']['train']),
            'val': str(dataset_base_path / self.config['dataset']['val']),
            'test': str(dataset_base_path / self.config['dataset']['test']),
            'nc': self.dataset_analysis['num_classes'],
            'names': self.dataset_analysis['all_classes']
        }
        
        # Save dataset config
        dataset_config_path = Path(self.experiment_dir) / "dataset.yaml"
        with open(dataset_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, indent=2)
        
        # Create dataset analysis plots
        if 'train' in self.dataset_analysis:
            VisualizationUtils.plot_class_distribution(
                self.dataset_analysis['train']['class_counts'],
                str(Path(self.experiment_dir) / "train_class_distribution.png")
            )
        
        logger.info("Dataset preparation complete")
        logger.info(f"Dataset config saved to: {dataset_config_path}")
        logger.info(f"Dataset paths: train={dataset_config['train']}, val={dataset_config['val']}")
        
        return dataset_config
    
    def create_model(self) -> YOLO:
        """Create and configure YOLO model"""
        logger.info("Creating YOLO model...")
        
        # Get model architecture
        model_arch = self.config['model']['architecture']
        
        # Create model
        if model_arch.endswith('.pt'):
            # Load pretrained model
            self.model = YOLO(model_arch)
            logger.info(f"Loaded pretrained model: {model_arch}")
        else:
            # Create new model
            self.model = YOLO(model_arch)
            logger.info(f"Created new model: {model_arch}")
        
        # Configure model for classification
        if hasattr(self.model, 'task'):
            self.model.task = 'classify'
        
        # Log model information
        total_params = ModelUtils.count_parameters(self.model.model)
        trainable_params = ModelUtils.count_trainable_parameters(self.model.model)
        
        logger.info(f"Model created successfully:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Model size: {model_arch}")
        
        return self.model
    
    def configure_training(self) -> Dict:
        """Configure training parameters with overfitting prevention"""
        logger.info("Configuring training parameters...")
        
        # Get optimal batch size
        optimal_batch_size = ModelUtils.get_optimal_batch_size(
            self.config['model']['architecture'].split('-')[1][0],  # Extract model size
            self.device
        )
        
        # Update batch size if needed
        if optimal_batch_size != self.config['training']['batch_size']:
            logger.info(f"Updating batch size from {self.config['training']['batch_size']} to {optimal_batch_size}")
            self.config_manager.update('training.batch_size', optimal_batch_size)
        
        # Get dataset paths - use absolute paths
        dataset_base_path = Path(self.config['dataset']['path']).resolve()
        train_path = str(dataset_base_path / self.config['dataset']['train'])
        val_path = str(dataset_base_path / self.config['dataset']['val'])
        test_path = str(dataset_base_path / self.config['dataset']['test'])
        
        # Verify paths exist
        if not Path(train_path).exists():
            raise FileNotFoundError(f"Training path does not exist: {train_path}")
        if not Path(val_path).exists():
            raise FileNotFoundError(f"Validation path does not exist: {val_path}")
        if not Path(test_path).exists():
            raise FileNotFoundError(f"Test path does not exist: {test_path}")
        
        # Training configuration - pass train path directly
        train_config = {
            'data': train_path,  # Pass train path directly
            'epochs': self.config['training']['epochs'],
            'batch': self.config['training']['batch_size'],
            'imgsz': self.config['training']['imgsz'],
            'device': self.device,
            'workers': self.config['training']['workers'],
            'project': str(Path(self.experiment_dir) / "training"),
            'name': 'yolo_plant_detection',
            'exist_ok': True,
            'pretrained': self.config['model']['pretrained'],
            'optimizer': self.config['training']['optimizer'],
            'lr0': self.config['training']['lr0'],
            'lrf': self.config['training']['lrf'],
            'momentum': self.config['training']['momentum'],
            'weight_decay': self.config['training']['weight_decay'],
            'warmup_epochs': self.config['training']['warmup_epochs'],
            'warmup_momentum': self.config['training']['warmup_momentum'],
            'warmup_bias_lr': self.config['training']['warmup_bias_lr'],
            'box': self.config['training']['box'],
            'cls': self.config['training']['cls'],
            'dfl': self.config['training']['dfl'],
            'hsv_h': self.config['training']['hsv_h'],
            'hsv_s': self.config['training']['hsv_s'],
            'hsv_v': self.config['training']['hsv_v'],
            'degrees': self.config['training']['degrees'],
            'translate': self.config['training']['translate'],
            'scale': self.config['training']['scale'],
            'shear': self.config['training']['shear'],
            'perspective': self.config['training']['perspective'],
            'flipud': self.config['training']['flipud'],
            'fliplr': self.config['training']['fliplr'],
            'mosaic': self.config['training']['mosaic'],
            'mixup': self.config['training']['mixup'],
            'copy_paste': self.config['training']['copy_paste'],
            'val': True,  # Enable validation (boolean)
            'save_period': self.config['training']['save_period'],
            'patience': self.config['training']['patience'],
            'save_json': self.config['validation']['save_json'],
            'save_hybrid': self.config['validation']['save_hybrid'],
            'conf': self.config['validation']['conf'],
            'iou': self.config['validation']['iou'],
            'max_det': self.config['validation']['max_det'],
            'half': self.config['validation']['half'],
            'dnn': self.config['validation']['dnn'],
            'verbose': self.config['logging']['verbose'],
            'seed': self.config['system']['seed'],
            'deterministic': self.config['system']['deterministic'],
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'amp': self.config['performance']['amp'],
            'fraction': self.config['performance']['fraction'],
            'cache': self.config['performance']['cache'],
            'overlap_mask': self.config['performance']['overlap_mask'],
            'mask_ratio': self.config['performance']['mask_ratio'],
            'dropout': self.config['performance']['dropout'],
        }
        
        logger.info("Training configuration complete")
        logger.info(f"📊 Epochs: {self.config['training']['epochs']}")
        logger.info(f"📊 Patience: {self.config['training']['patience']}")
        logger.info(f"📊 Batch size: {self.config['training']['batch_size']}")
        logger.info(f"📊 Image size: {self.config['training']['imgsz']}x{self.config['training']['imgsz']}")
        logger.info(f"📊 Training data: {train_path}")
        logger.info(f"📊 Validation data: {val_path}")
        logger.info(f"📊 Test data: {test_path}")
        
        return train_config
    
    def train(self) -> Dict:
        """Train the YOLO model with comprehensive logging and monitoring"""
        logger.info("🚀 Starting YOLO model training...")
        
        try:
            # Configure training
            train_config = self.configure_training()
            
            # Initialize model
            model = YOLO(self.config['model']['architecture'])
            logger.info(f"✅ Model initialized: {self.config['model']['architecture']}")
            
            # Start training
            logger.info("🔥 Training started...")
            results = model.train(**train_config)
            
            # Save training results and generate performance graphs
            self.save_training_results(results)
            
            logger.info("✅ Training completed successfully!")
            return {
                'status': 'success',
                'model_path': str(Path(train_config['project']) / train_config['name'] / 'weights' / 'best.pt'),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return {'status': 'error', 'error': str(e)}
    
    def save_training_results(self, results):
        """Save training results and generate performance graphs"""
        logger.info("📊 Saving training results and generating performance graphs...")
        
        try:
            # Create results directory
            results_dir = Path(self.experiment_dir) / "results"
            results_dir.mkdir(exist_ok=True)
            
            # Save training metrics
            if hasattr(results, 'results_dict'):
                metrics_file = results_dir / "training_metrics.json"
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(results.results_dict, f, indent=2, default=str)
                logger.info(f"📈 Training metrics saved to: {metrics_file}")
            
            # Generate and save performance graphs
            self.generate_performance_graphs(results_dir)
            
        except Exception as e:
            logger.error(f"❌ Failed to save training results: {str(e)}")
    
    def generate_performance_graphs(self, results_dir: Path):
        """Generate and save performance graphs without displaying them"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Plant Detection Model Training Performance', fontsize=16, fontweight='bold')
            
            # Get training history from results
            if hasattr(self, 'training_history'):
                history = self.training_history
            else:
                # Try to load from results
                history = getattr(self, 'results', {}).get('results_dict', {})
            
            # Plot 1: Loss curves
            if 'train/box_loss' in history and 'val/box_loss' in history:
                axes[0, 0].plot(history['train/box_loss'], label='Train Box Loss', linewidth=2)
                axes[0, 0].plot(history['val/box_loss'], label='Val Box Loss', linewidth=2)
                axes[0, 0].set_title('Box Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Classification loss
            if 'train/cls_loss' in history and 'val/cls_loss' in history:
                axes[0, 1].plot(history['train/cls_loss'], label='Train Cls Loss', linewidth=2)
                axes[0, 1].plot(history['val/cls_loss'], label='Val Cls Loss', linewidth=2)
                axes[0, 1].set_title('Classification Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: mAP metrics
            if 'metrics/mAP50(B)' in history and 'metrics/mAP50-95(B)' in history:
                axes[1, 0].plot(history['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2)
                axes[1, 0].plot(history['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2)
                axes[1, 0].set_title('Mean Average Precision')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('mAP')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Learning rate
            if 'train/lr0' in history:
                axes[1, 1].plot(history['train/lr0'], label='Learning Rate', linewidth=2, color='orange')
                axes[1, 1].set_title('Learning Rate Schedule')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_yscale('log')
            
            # Adjust layout and save
            plt.tight_layout()
            performance_plot_path = results_dir / "training_performance.png"
            plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"📊 Performance graphs saved to: {performance_plot_path}")
            
            # Save individual plots
            self.save_individual_plots(history, results_dir)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate performance graphs: {str(e)}")
    
    def save_individual_plots(self, history: Dict, results_dir: Path):
        """Save individual performance plots"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Loss comparison plot
            if 'train/box_loss' in history and 'val/box_loss' in history:
                plt.figure(figsize=(10, 6))
                plt.plot(history['train/box_loss'], label='Train Box Loss', linewidth=2)
                plt.plot(history['val/box_loss'], label='Val Box Loss', linewidth=2)
                plt.title('Box Loss Comparison', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(results_dir / "box_loss.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # mAP plot
            if 'metrics/mAP50(B)' in history:
                plt.figure(figsize=(10, 6))
                plt.plot(history['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2, color='green')
                if 'metrics/mAP50-95(B)' in history:
                    plt.plot(history['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2, color='blue')
                plt.title('Mean Average Precision', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch')
                plt.ylabel('mAP')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(results_dir / "map_metrics.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info("📈 Individual performance plots saved")
            
        except Exception as e:
            logger.error(f"❌ Failed to save individual plots: {str(e)}")
    
    def validate(self, model_path: str = None) -> Dict:
        """Validate trained model"""
        logger.info("Validating model...")
        
        if model_path is None:
            model_path = self.best_model_path
        
        if not model_path or not os.path.exists(model_path):
            logger.error(f"Model path not found: {model_path}")
            return {'success': False, 'error': 'Model path not found'}
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Get test dataset path
            dataset_base_path = Path(self.config['dataset']['path']).resolve()
            test_path = str(dataset_base_path / self.config['dataset']['test'])
            
            results = model.val(
                data=test_path,  # Use test path directly
                split='test',
                imgsz=self.config['training']['imgsz'],
                batch=self.config['training']['batch_size'],
                device=self.device,
                workers=self.config['training']['workers'],
                verbose=self.config['logging']['verbose'],
                save_json=True,
                save_hybrid=False,
                conf=self.config['validation']['conf'],
                iou=self.config['validation']['iou'],
                max_det=self.config['validation']['max_det'],
                half=self.config['validation']['half'],
                dnn=self.config['validation']['dnn']
            )
            
            # Save validation results
            val_results_path = Path(self.experiment_dir) / "validation_results.json"
            with open(val_results_path, 'w') as f:
                json.dump(results.results_dict, f, indent=2)
            
            logger.info("Validation completed successfully")
            return {
                'success': True,
                'results': results.results_dict,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def export_model(self, model_path: str = None, formats: List[str] = None) -> Dict:
        """Export model to various formats"""
        logger.info("Exporting model...")
        
        if model_path is None:
            model_path = self.best_model_path
        
        if formats is None:
            formats = ['torchscript', 'onnx', 'engine']
        
        if not model_path or not os.path.exists(model_path):
            logger.error(f"Model path not found: {model_path}")
            return {'success': False, 'error': 'Model path not found'}
        
        try:
            model = YOLO(model_path)
            export_results = {}
            
            for format in formats:
                try:
                    exported_path = model.export(format=format, imgsz=self.config['training']['imgsz'])
                    export_results[format] = str(exported_path)
                    logger.info(f"Model exported to {format}: {exported_path}")
                except Exception as e:
                    logger.warning(f"Failed to export to {format}: {e}")
                    export_results[format] = None
            
            # Save export results
            export_results_path = Path(self.experiment_dir) / "export_results.json"
            with open(export_results_path, 'w') as f:
                json.dump(export_results, f, indent=2)
            
            logger.info("Model export completed")
            return {
                'success': True,
                'export_results': export_results
            }
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        summary = {
            'experiment_dir': self.experiment_dir,
            'dataset_info': {
                'num_classes': self.dataset_analysis['num_classes'],
                'total_images': sum(
                    split_data['total_images'] 
                    for split_data in self.dataset_analysis.values() 
                    if isinstance(split_data, dict) and 'total_images' in split_data
                ),
                'classes': self.dataset_analysis['all_classes']
            },
            'model_info': {
                'architecture': self.config['model']['architecture'],
                'device': str(self.device),
                'batch_size': self.config['training']['batch_size'],
                'image_size': self.config['training']['imgsz']
            },
            'training_info': {
                'epochs': self.config['training']['epochs'],
                'optimizer': self.config['training']['optimizer'],
                'learning_rate': self.config['training']['lr0'],
                'patience': self.config['training']['patience']
            },
            'system_info': self.system_monitor.get_system_info(),
            'best_model_path': self.best_model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary


def main():
    """Main training function"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Clean up old runs
    cleanup_old_runs(keep_last=3)
    
    # Initialize trainer
    trainer = YOLOTrainer()
    
    # Log system status
    trainer.system_monitor.log_system_status()
    
    # Start training
    training_result = trainer.train()
    
    if training_result['status'] == 'success':
        logger.info("Training completed successfully!")
        
        # Validate model
        validation_result = trainer.validate()
        if validation_result['success']:
            logger.info("Validation completed successfully!")
        
        # Export model
        export_result = trainer.export_model()
        if export_result['success']:
            logger.info("Model export completed successfully!")
        
        # Print summary
        summary = trainer.get_training_summary()
        logger.info("Training Summary:")
        logger.info(json.dumps(summary, indent=2))
        
    else:
        logger.error(f"Training failed: {training_result['error']}")


if __name__ == "__main__":
    main() 