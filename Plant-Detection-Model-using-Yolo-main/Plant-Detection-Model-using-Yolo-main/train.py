#!/usr/bin/env python3
"""
Main Training Script for YOLO Plant Detection
Professional training pipeline with comprehensive monitoring and optimization
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.trainer import YOLOTrainer
from src.utils import SystemMonitor, cleanup_old_runs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='YOLO Plant Detection Training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--cleanup', action='store_true', 
                       help='Clean up old experiment runs')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation after training')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing model')
    parser.add_argument('--export-only', action='store_true',
                       help='Only export existing model')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to existing model for validation/export')
    
    args = parser.parse_args()
    
    try:
        # Clean up old runs if requested
        if args.cleanup:
            cleanup_old_runs(keep_last=3)
            logger.info("Cleaned up old experiment runs")
        
        # Initialize system monitor
        system_monitor = SystemMonitor()
        system_info = system_monitor.log_system_status()
        
        # Initialize trainer
        trainer = YOLOTrainer(args.config)
        
        if args.validate_only:
            # Only validate existing model
            if args.model_path:
                validation_result = trainer.validate(args.model_path)
                if validation_result['success']:
                    logger.info("Validation completed successfully!")
                    logger.info(f"Validation results: {json.dumps(validation_result['results'], indent=2)}")
                else:
                    logger.error(f"Validation failed: {validation_result['error']}")
            else:
                logger.error("Model path required for validation-only mode")
            return
        
        if args.export_only:
            # Only export existing model
            if args.model_path:
                export_result = trainer.export_model(args.model_path)
                if export_result['success']:
                    logger.info("Model export completed successfully!")
                    logger.info(f"Export results: {json.dumps(export_result['export_results'], indent=2)}")
                else:
                    logger.error(f"Model export failed: {export_result['error']}")
            else:
                logger.error("Model path required for export-only mode")
            return
        
        # Start training
        logger.info("🚀 Starting plant detection model training...")
        training_result = trainer.train()
        
        if training_result['status'] == 'success':
            logger.info("✅ Training completed successfully!")
            logger.info(f"📁 Best model saved to: {training_result['model_path']}")
            
            # Optional: Run validation
            if args.validate:
                logger.info("🔍 Running validation on best model...")
                validation_result = trainer.validate(training_result['model_path'])
                if validation_result['status'] == 'success':
                    logger.info("✅ Validation completed successfully!")
                else:
                    logger.error(f"❌ Validation failed: {validation_result['error']}")
            
            # Show summary
            logger.info("\n" + "="*60)
            logger.info("🎉 TRAINING SUMMARY")
            logger.info("="*60)
            logger.info(f"📊 Model: {trainer.config['model']['architecture']}")
            logger.info(f"📁 Dataset: {trainer.config['dataset']['path']}")
            logger.info(f"🔢 Classes: {trainer.dataset_analysis['num_classes']}")
            logger.info(f"📈 Best model: {training_result['model_path']}")
            logger.info(f"📂 Results: {trainer.experiment_dir}")
            logger.info("="*60)
            
        else:
            logger.error(f"❌ Training failed: {training_result['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 