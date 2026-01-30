"""
Professional YOLO Inference Pipeline for Plant Detection
Advanced inference with optimization, batch processing, and comprehensive results
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time
import json
from datetime import datetime
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device

from .utils import (
    SystemMonitor, ConfigManager, ModelUtils, VisualizationUtils
)

logger = logging.getLogger(__name__)


class PlantDetector:
    """
    Professional Plant Detection Inference Pipeline
    Features:
    - Batch processing
    - Real-time inference
    - Confidence filtering
    - Result visualization
    - Performance optimization
    - Comprehensive logging
    """
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.system_monitor = SystemMonitor()
        self.device = ModelUtils.get_device()
        
        # Load model
        self.model_path = model_path
        self.model = self.load_model()
        
        # Inference state
        self.class_names = []
        self.confidence_threshold = self.config['inference']['conf']
        self.iou_threshold = self.config['inference']['iou']
        self.max_detections = self.config['inference']['max_det']
        
        # Performance tracking
        self.inference_times = []
        self.total_images_processed = 0
        
        logger.info(f"Plant Detector initialized with model: {model_path}")
    
    def load_model(self) -> YOLO:
        """Load and configure YOLO model for inference"""
        logger.info(f"Loading model from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Load model
            model = YOLO(self.model_path)
            
            # Move to device
            model.to(self.device)
            
            # Get class names
            if hasattr(model, 'names'):
                self.class_names = model.names
            else:
                # Try to load from config
                self.class_names = self.config['dataset']['names']
            
            logger.info(f"Model loaded successfully with {len(self.class_names)} classes")
            logger.info(f"Using device: {self.device}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for inference"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
            else:
                image = image_path
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image_rgb
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def predict_single(self, image_path: str, save_result: bool = False, 
                      output_dir: str = None) -> Dict:
        """Perform inference on a single image"""
        start_time = time.time()
        
        try:
            # Preprocess image
            image = self.preprocess_image(image_path)
            
            # Perform inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            # Process results
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_images_processed += 1
            
            # Extract predictions
            predictions = self.extract_predictions(results, image_path)
            
            # Save result if requested
            if save_result:
                self.save_prediction_result(image, predictions, image_path, output_dir)
            
            return {
                'success': True,
                'image_path': image_path,
                'predictions': predictions,
                'inference_time': inference_time,
                'image_shape': image.shape
            }
            
        except Exception as e:
            logger.error(f"Error during inference on {image_path}: {e}")
            return {
                'success': False,
                'image_path': image_path,
                'error': str(e),
                'inference_time': time.time() - start_time
            }
    
    def predict_batch(self, image_paths: List[str], batch_size: int = 8,
                     save_results: bool = False, output_dir: str = None) -> Dict:
        """Perform batch inference on multiple images"""
        logger.info(f"Starting batch inference on {len(image_paths)} images")
        
        start_time = time.time()
        results = []
        successful_predictions = 0
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
            
            try:
                # Load batch images
                batch_images = []
                valid_paths = []
                
                for img_path in batch_paths:
                    try:
                        image = self.preprocess_image(img_path)
                        batch_images.append(image)
                        valid_paths.append(img_path)
                    except Exception as e:
                        logger.warning(f"Skipping invalid image {img_path}: {e}")
                        continue
                
                if not batch_images:
                    continue
                
                # Perform batch inference
                batch_start_time = time.time()
                batch_results = self.model(
                    batch_images,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_detections,
                    verbose=False
                )
                batch_inference_time = time.time() - batch_start_time
                
                # Process batch results
                for idx, (result, img_path) in enumerate(zip(batch_results, valid_paths)):
                    predictions = self.extract_predictions(result, img_path)
                    
                    result_dict = {
                        'success': True,
                        'image_path': img_path,
                        'predictions': predictions,
                        'inference_time': batch_inference_time / len(valid_paths),
                        'image_shape': batch_images[idx].shape
                    }
                    
                    results.append(result_dict)
                    successful_predictions += 1
                    
                    # Save individual result if requested
                    if save_results:
                        self.save_prediction_result(
                            batch_images[idx], predictions, img_path, output_dir
                        )
                
                self.inference_times.extend([batch_inference_time / len(valid_paths)] * len(valid_paths))
                self.total_images_processed += len(valid_paths)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add failed results
                for img_path in batch_paths:
                    results.append({
                        'success': False,
                        'image_path': img_path,
                        'error': str(e),
                        'inference_time': 0
                    })
        
        total_time = time.time() - start_time
        
        return {
            'success': True,
            'total_images': len(image_paths),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(image_paths) - successful_predictions,
            'total_time': total_time,
            'average_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'results': results
        }
    
    def extract_predictions(self, result, image_path: str) -> List[Dict]:
        """Extract predictions from YOLO result"""
        predictions = []
        
        try:
            # Handle different result formats
            if hasattr(result, 'boxes') and result.boxes is not None:
                # Detection results
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes[i]
                        prediction = {
                            'class_id': int(box.cls[i].item()) if len(box.cls) > i else 0,
                            'class_name': self.class_names[int(box.cls[i].item())] if len(box.cls) > i and int(box.cls[i].item()) < len(self.class_names) else 'unknown',
                            'confidence': float(box.conf[i].item()) if len(box.conf) > i else 0.0,
                            'bbox': box.xyxy[i].cpu().numpy().tolist() if len(box.xyxy) > i else [0, 0, 0, 0]
                        }
                        predictions.append(prediction)
            
            elif hasattr(result, 'probs') and result.probs is not None:
                # Classification results
                probs = result.probs
                if probs is not None:
                    # Get top predictions
                    top_k = min(5, len(self.class_names))
                    top_indices = torch.topk(probs, top_k).indices
                    top_probs = torch.topk(probs, top_k).values
                    
                    for i in range(top_k):
                        class_id = int(top_indices[i].item())
                        prediction = {
                            'class_id': class_id,
                            'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown',
                            'confidence': float(top_probs[i].item()),
                            'bbox': None  # No bounding box for classification
                        }
                        predictions.append(prediction)
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error extracting predictions: {e}")
        
        return predictions
    
    def save_prediction_result(self, image: np.ndarray, predictions: List[Dict], 
                             image_path: str, output_dir: str = None):
        """Save prediction result with visualization"""
        try:
            if output_dir is None:
                output_dir = "inference_results"
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original image
            ax1.imshow(image)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Predictions visualization
            ax2.imshow(image)
            ax2.set_title('Predictions')
            ax2.axis('off')
            
            # Draw bounding boxes and labels
            for pred in predictions:
                if pred['bbox'] is not None:
                    x1, y1, x2, y2 = pred['bbox']
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, color='red', linewidth=2)
                    ax2.add_patch(rect)
                    
                    label = f"{pred['class_name']}: {pred['confidence']:.2f}"
                    ax2.text(x1, y1-10, label, color='red', fontsize=8, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Save visualization
            image_name = Path(image_path).stem
            viz_path = output_path / f"{image_name}_predictions.png"
            plt.tight_layout()
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save prediction data
            pred_data = {
                'image_path': image_path,
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            json_path = output_path / f"{image_name}_predictions.json"
            with open(json_path, 'w') as f:
                json.dump(pred_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving prediction result: {e}")
    
    def predict_directory(self, input_dir: str, output_dir: str = None, 
                         batch_size: int = 8) -> Dict:
        """Perform inference on all images in a directory"""
        logger.info(f"Processing directory: {input_dir}")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(input_dir).glob(f"*{ext}"))
            image_paths.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            logger.warning(f"No image files found in {input_dir}")
            return {'success': False, 'error': 'No image files found'}
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Perform batch inference
        results = self.predict_batch(image_paths, batch_size, True, output_dir)
        
        # Generate summary report
        summary = self.generate_summary_report(results, input_dir)
        
        return {
            **results,
            'summary': summary
        }
    
    def generate_summary_report(self, results: Dict, input_dir: str) -> Dict:
        """Generate comprehensive summary report"""
        logger.info("Generating summary report...")
        
        # Collect statistics
        class_counts = defaultdict(int)
        confidence_scores = []
        successful_results = [r for r in results['results'] if r['success']]
        
        for result in successful_results:
            if result['predictions']:
                top_prediction = result['predictions'][0]
                class_counts[top_prediction['class_name']] += 1
                confidence_scores.append(top_prediction['confidence'])
        
        # Calculate statistics
        total_images = len(results['results'])
        successful_images = len(successful_results)
        failed_images = total_images - successful_images
        
        summary = {
            'input_directory': input_dir,
            'total_images_processed': total_images,
            'successful_predictions': successful_images,
            'failed_predictions': failed_images,
            'success_rate': successful_images / total_images if total_images > 0 else 0,
            'average_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'total_processing_time': results['total_time'],
            'class_distribution': dict(class_counts),
            'confidence_statistics': {
                'mean': np.mean(confidence_scores) if confidence_scores else 0,
                'std': np.std(confidence_scores) if confidence_scores else 0,
                'min': np.min(confidence_scores) if confidence_scores else 0,
                'max': np.max(confidence_scores) if confidence_scores else 0
            },
            'performance_metrics': {
                'images_per_second': successful_images / results['total_time'] if results['total_time'] > 0 else 0,
                'average_inference_time_ms': np.mean(self.inference_times) * 1000 if self.inference_times else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary report
        if 'output_dir' in results:
            summary_path = Path(results['output_dir']) / "inference_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Create visualization
            self.create_summary_visualization(summary, results['output_dir'])
        
        return summary
    
    def create_summary_visualization(self, summary: Dict, output_dir: str):
        """Create visualization for summary report"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Inference Summary Report', fontsize=16)
            
            # Class distribution
            if summary['class_distribution']:
                classes = list(summary['class_distribution'].keys())
                counts = list(summary['class_distribution'].values())
                
                axes[0, 0].bar(range(len(classes)), counts)
                axes[0, 0].set_title('Class Distribution')
                axes[0, 0].set_xlabel('Class')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Confidence distribution
            if 'confidence_scores' in summary:
                axes[0, 1].hist(summary['confidence_scores'], bins=20, alpha=0.7)
                axes[0, 1].set_title('Confidence Distribution')
                axes[0, 1].set_xlabel('Confidence')
                axes[0, 1].set_ylabel('Frequency')
            
            # Performance metrics
            metrics = summary['performance_metrics']
            metric_names = ['Images/sec', 'Avg Time (ms)']
            metric_values = [metrics['images_per_second'], metrics['average_inference_time_ms']]
            
            axes[1, 0].bar(metric_names, metric_values)
            axes[1, 0].set_title('Performance Metrics')
            axes[1, 0].set_ylabel('Value')
            
            # Success rate
            success_rate = summary['success_rate'] * 100
            axes[1, 1].pie([success_rate, 100-success_rate], 
                          labels=['Success', 'Failed'], 
                          autopct='%1.1f%%',
                          colors=['green', 'red'])
            axes[1, 1].set_title('Success Rate')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = Path(output_dir) / "inference_summary.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Summary visualization saved to {viz_path}")
            
        except Exception as e:
            logger.error(f"Error creating summary visualization: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {'error': 'No inference data available'}
        
        return {
            'total_images_processed': self.total_images_processed,
            'average_inference_time': np.mean(self.inference_times),
            'median_inference_time': np.median(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'total_processing_time': np.sum(self.inference_times),
            'images_per_second': self.total_images_processed / np.sum(self.inference_times) if np.sum(self.inference_times) > 0 else 0
        }


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plant Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize detector
        detector = PlantDetector(args.model, args.config)
        
        # Update confidence threshold if provided
        if args.conf != 0.25:
            detector.confidence_threshold = args.conf
        
        # Check if input is file or directory
        if os.path.isfile(args.input):
            # Single image inference
            result = detector.predict_single(args.input, True, args.output)
            if result['success']:
                logger.info(f"Prediction completed for {args.input}")
                logger.info(f"Top prediction: {result['predictions'][0] if result['predictions'] else 'No predictions'}")
            else:
                logger.error(f"Prediction failed: {result['error']}")
        
        elif os.path.isdir(args.input):
            # Directory inference
            result = detector.predict_directory(args.input, args.output, args.batch_size)
            if result['success']:
                logger.info(f"Batch inference completed successfully!")
                logger.info(f"Processed {result['total_images']} images")
                logger.info(f"Success rate: {result['success_rate']:.2%}")
                logger.info(f"Average inference time: {result['average_inference_time']:.3f}s")
            else:
                logger.error(f"Batch inference failed: {result['error']}")
        
        else:
            logger.error(f"Input path does not exist: {args.input}")
        
        # Print performance stats
        stats = detector.get_performance_stats()
        logger.info("Performance Statistics:")
        logger.info(json.dumps(stats, indent=2))
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")


if __name__ == "__main__":
    main() 