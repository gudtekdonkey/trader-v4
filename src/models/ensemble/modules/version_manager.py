"""
Model version management module for ensemble predictor.
Handles model versioning and compatibility checks.
"""

from typing import Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Model version constants
MODEL_VERSION = "2.0.0"
MIN_COMPATIBLE_VERSION = "1.0.0"


class ModelVersionManager:
    """Manage model versions and compatibility"""
    
    @staticmethod
    def check_compatibility(saved_version: str, current_version: str = MODEL_VERSION) -> bool:
        """Check if saved model version is compatible"""
        try:
            saved_parts = [int(x) for x in saved_version.split('.')]
            current_parts = [int(x) for x in current_version.split('.')]
            min_parts = [int(x) for x in MIN_COMPATIBLE_VERSION.split('.')]
            
            # Check if saved version is at least minimum compatible
            for i in range(len(min_parts)):
                if i >= len(saved_parts):
                    return False
                if saved_parts[i] < min_parts[i]:
                    return False
                elif saved_parts[i] > min_parts[i]:
                    break
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking version compatibility: {e}")
            return False
    
    @staticmethod
    def migrate_model(old_state: Dict, old_version: str, new_version: str) -> Dict:
        """Migrate model from old version to new version"""
        try:
            # Implement version-specific migrations
            if old_version.startswith("1.") and new_version.startswith("2."):
                # Example migration from v1 to v2
                logger.info(f"Migrating model from {old_version} to {new_version}")
                
                # Add new fields with defaults
                if 'uncertainty_weighting' not in old_state:
                    old_state['uncertainty_weighting'] = True
                if 'temperature' not in old_state:
                    old_state['temperature'] = 1.0
                    
            return old_state
            
        except Exception as e:
            logger.error(f"Error migrating model: {e}")
            return old_state
    
    @staticmethod
    def get_version_info() -> Dict[str, str]:
        """Get current version information"""
        return {
            'current_version': MODEL_VERSION,
            'min_compatible_version': MIN_COMPATIBLE_VERSION
        }
