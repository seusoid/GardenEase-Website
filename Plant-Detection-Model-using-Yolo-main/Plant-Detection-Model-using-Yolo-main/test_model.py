#!/usr/bin/env python3
"""
Model Testing Script for YOLO Plant Detection
Comprehensive evaluation on test dataset with detailed performance metrics
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ultralytics import YOLO
from src.utils import ConfigManager, SystemMonitor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_testing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ModelTester:
    """Comprehensive model testing and evaluation"""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        self.model_path = Path(model_path)
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.test_path = Path(self.config['dataset']['path']) / self.config['dataset']['test']
        
        # Load model
        logger.info(f"Loading model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        logger.info("✅ Model loaded successfully!")
        
        # Get class names
        self.class_names = self._get_class_names()
        logger.info(f"📊 Found {len(self.class_names)} classes")
        
        # Results storage
        self.predictions = []
        self.true_labels = []
        self.class_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': []})
        
    def _get_class_names(self) -> List[str]:
        """Get class names from test dataset"""
        if self.test_path.exists():
            classes = [d.name for d in self.test_path.iterdir() if d.is_dir()]
            return sorted(classes)
        else:
            logger.error(f"Test path does not exist: {self.test_path}")
            return []
    
    def test_dataset(self) -> Dict:
        """Test the model on the entire test dataset"""
        logger.info("🚀 Starting model evaluation on test dataset...")
        
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test path not found: {self.test_path}")
        
        start_time = time.time()
        total_images = 0
        processed_images = 0
        
        # Process each class
        for class_dir in sorted(self.test_path.iterdir()):
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            logger.info(f"📁 Testing class: {class_name}")
            
            # Get all images in this class
            image_files = [f for f in class_dir.iterdir() 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
            
            total_images += len(image_files)
            
            for img_path in image_files:
                try:
                    # Make prediction
                    results = self.model.predict(str(img_path), verbose=False)
                    
                    if results and len(results) > 0:
                        result = results[0]
                        
                        # Get predicted class
                        if hasattr(result, 'probs') and result.probs is not None:
                            predicted_class_idx = result.probs.top1
                            predicted_class = self.class_names[predicted_class_idx]
                            confidence = float(result.probs.top1conf)
                        else:
                            # Fallback for older YOLO versions
                            predicted_class = result.names[result.probs.top1] if hasattr(result, 'names') else "unknown"
                            confidence = 0.0
                        
                        # Store results
                        is_correct = predicted_class == class_name
                        self.predictions.append(predicted_class)
                        self.true_labels.append(class_name)
                        
                        # Update class statistics
                        self.class_results[class_name]['total'] += 1
                        if is_correct:
                            self.class_results[class_name]['correct'] += 1
                        
                        self.class_results[class_name]['predictions'].append({
                            'image': img_path.name,
                            'predicted': predicted_class,
                            'confidence': confidence,
                            'correct': is_correct
                        })
                        
                        processed_images += 1
                        
                        # Progress update
                        if processed_images % 100 == 0:
                            logger.info(f"📊 Processed {processed_images} images...")
                            
                except Exception as e:
                    logger.warning(f"❌ Error processing {img_path}: {e}")
                    continue
        
        testing_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(testing_time, total_images, processed_images)
        
        logger.info("✅ Model evaluation completed!")
        return metrics
    
    def _calculate_metrics(self, testing_time: float, total_images: int, processed_images: int) -> Dict:
        """Calculate comprehensive performance metrics"""
        logger.info("📊 Calculating performance metrics...")
        
        # Overall accuracy
        overall_accuracy = accuracy_score(self.true_labels, self.predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, self.predictions, average=None, labels=self.class_names
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.true_labels, self.predictions, labels=self.class_names)
        
        # Per-class accuracy
        class_accuracy = {}
        for class_name in self.class_names:
            if class_name in self.class_results:
                class_accuracy[class_name] = (
                    self.class_results[class_name]['correct'] / 
                    self.class_results[class_name]['total']
                )
        
        # Top-5 accuracy (if available)
        top5_accuracy = self._calculate_top5_accuracy()
        
        metrics = {
            'overall_accuracy': overall_accuracy,
            'top5_accuracy': top5_accuracy,
            'testing_time_seconds': testing_time,
            'total_images': total_images,
            'processed_images': processed_images,
            'images_per_second': processed_images / testing_time if testing_time > 0 else 0,
            'class_accuracy': class_accuracy,
            'precision': dict(zip(self.class_names, precision)),
            'recall': dict(zip(self.class_names, recall)),
            'f1_score': dict(zip(self.class_names, f1)),
            'support': dict(zip(self.class_names, support)),
            'confusion_matrix': cm.tolist(),
            'class_names': self.class_names,
            'detailed_results': dict(self.class_results)
        }
        
        return metrics
    
    def _calculate_top5_accuracy(self) -> float:
        """Calculate top-5 accuracy (placeholder for now)"""
        # This would require storing top-5 predictions, simplified for now
        return 0.0
    
    def save_results(self, metrics: Dict, output_dir: str = "test_results"):
        """Save comprehensive test results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"💾 Saving results to: {output_path}")
        
        # Save metrics as JSON
        metrics_file = output_path / "test_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save detailed results
        detailed_file = output_path / "detailed_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(metrics['detailed_results'], f, indent=2, default=str)
        
        # Generate and save visualizations
        self._generate_visualizations(metrics, output_path)
        
        # Generate summary report
        self._generate_summary_report(metrics, output_path)
        
        logger.info(f"✅ Results saved to: {output_path}")
    
    def _generate_visualizations(self, metrics: Dict, output_path: Path):
        """Generate performance visualizations"""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Confusion Matrix
            fig, ax = plt.subplots(figsize=(20, 16))
            cm = np.array(metrics['confusion_matrix'])
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=metrics['class_names'], 
                       yticklabels=metrics['class_names'],
                       ax=ax)
            ax.set_title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
            ax.set_xlabel('Predicted Class', fontsize=12)
            ax.set_ylabel('True Class', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_path / "confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Per-class Accuracy
            fig, ax = plt.subplots(figsize=(20, 10))
            class_acc = list(metrics['class_accuracy'].values())
            class_names = list(metrics['class_accuracy'].keys())
            
            bars = ax.bar(range(len(class_acc)), class_acc, color='skyblue', alpha=0.7)
            ax.set_title('Per-Class Accuracy', fontsize=16, fontweight='bold')
            ax.set_xlabel('Classes', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, class_acc):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_path / "per_class_accuracy.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Performance Summary
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
            
            # Overall metrics
            metrics_summary = [
                metrics['overall_accuracy'],
                np.mean(list(metrics['precision'].values())),
                np.mean(list(metrics['recall'].values())),
                np.mean(list(metrics['f1_score'].values()))
            ]
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
            
            bars = ax1.bar(metric_names, metrics_summary, color=colors, alpha=0.7)
            ax1.set_title('Overall Performance Metrics')
            ax1.set_ylim(0, 1)
            for bar, value in zip(bars, metrics_summary):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # Processing speed
            ax2.bar(['Images/sec'], [metrics['images_per_second']], color='orange', alpha=0.7)
            ax2.set_title('Processing Speed')
            ax2.text(0, metrics['images_per_second'] + 0.1, 
                    f"{metrics['images_per_second']:.1f}", ha='center', va='bottom')
            
            # Class distribution
            class_counts = [metrics['support'][name] for name in metrics['class_names']]
            ax3.pie(class_counts, labels=metrics['class_names'], autopct='%1.1f%%', startangle=90)
            ax3.set_title('Test Dataset Class Distribution')
            
            # Accuracy distribution
            accuracies = list(metrics['class_accuracy'].values())
            ax4.hist(accuracies, bins=20, color='lightblue', alpha=0.7, edgecolor='black')
            ax4.set_title('Accuracy Distribution Across Classes')
            ax4.set_xlabel('Accuracy')
            ax4.set_ylabel('Number of Classes')
            ax4.axvline(np.mean(accuracies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(accuracies):.3f}')
            ax4.legend()
            
            plt.tight_layout()
            plt.savefig(output_path / "performance_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("📊 Visualizations generated successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error generating visualizations: {e}")
    
    def _generate_summary_report(self, metrics: Dict, output_path: Path):
        """Generate a comprehensive summary report"""
        report_file = output_path / "test_summary_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("YOLO PLANT DETECTION MODEL TEST RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Test Dataset: {self.test_path}\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)\n")
            f.write(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)\n")
            f.write(f"Total Images: {metrics['total_images']}\n")
            f.write(f"Processed Images: {metrics['processed_images']}\n")
            f.write(f"Testing Time: {metrics['testing_time_seconds']:.2f} seconds\n")
            f.write(f"Processing Speed: {metrics['images_per_second']:.2f} images/second\n\n")
            
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            for class_name in metrics['class_names']:
                acc = metrics['class_accuracy'][class_name]
                prec = metrics['precision'][class_name]
                rec = metrics['recall'][class_name]
                f1 = metrics['f1_score'][class_name]
                support = metrics['support'][class_name]
                
                f.write(f"{class_name}:\n")
                f.write(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
                f.write(f"  Precision: {prec:.4f}\n")
                f.write(f"  Recall: {rec:.4f}\n")
                f.write(f"  F1-Score: {f1:.4f}\n")
                f.write(f"  Support: {support}\n\n")
            
            f.write("BEST PERFORMING CLASSES (Top 10)\n")
            f.write("-" * 40 + "\n")
            sorted_classes = sorted(metrics['class_accuracy'].items(), 
                                  key=lambda x: x[1], reverse=True)
            for i, (class_name, accuracy) in enumerate(sorted_classes[:10], 1):
                f.write(f"{i:2d}. {class_name}: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            
            f.write("\nWORST PERFORMING CLASSES (Bottom 10)\n")
            f.write("-" * 40 + "\n")
            for i, (class_name, accuracy) in enumerate(sorted_classes[-10:], 1):
                f.write(f"{i:2d}. {class_name}: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        
        logger.info(f"📄 Summary report saved to: {report_file}")


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='YOLO Plant Detection Model Testing')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='test_results',
                       help='Output directory for results')
    parser.add_argument('--test-path', type=str, default=None,
                       help='Custom test dataset path')
    
    args = parser.parse_args()
    
    try:
        # Initialize system monitor
        system_monitor = SystemMonitor()
        system_info = system_monitor.log_system_status()
        
        # Initialize model tester
        tester = ModelTester(args.model, args.config)
        
        # Override test path if provided
        if args.test_path:
            tester.test_path = Path(args.test_path)
            logger.info(f"Using custom test path: {tester.test_path}")
        
        # Run testing
        logger.info("🚀 Starting model evaluation...")
        metrics = tester.test_dataset()
        
        # Save results
        tester.save_results(metrics, args.output)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("🎉 TESTING SUMMARY")
        logger.info("="*60)
        logger.info(f"📊 Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
        logger.info(f"📈 Total Images: {metrics['total_images']}")
        logger.info(f"⏱️  Testing Time: {metrics['testing_time_seconds']:.2f} seconds")
        logger.info(f"🚀 Processing Speed: {metrics['images_per_second']:.2f} images/second")
        logger.info(f"📂 Results saved to: {args.output}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 