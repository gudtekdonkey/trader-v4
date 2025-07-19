#!/usr/bin/env python3
"""
Project Creator - Hyperliquid Trading Bot project structure generator
Creates the complete project directory structure and initial files for
the trading bot system with all necessary components.

File: create_project.py
Modified: 2025-07-15
"""

import os
import sys
from pathlib import Path

# Complete file contents
FILES = {
    "src/__init__.py": '"""Advanced Cryptocurrency Trading Bot for Hyperliquid DEX"""\n__version__ = "1.0.0"\n__author__ = "AI Trading Systems"',
    
    "src/models/__init__.py": '''from .lstm_attention import AttentionLSTM
from .temporal_fusion_transformer import TFTModel
from .ensemble import EnsemblePredictor
from .regime_detector import MarketRegimeDetector

__all__ = ['AttentionLSTM', 'TFTModel', 'EnsemblePredictor', 'MarketRegimeDetector']''',

    # ... (All file contents would be included here - truncated for brevity)
}

def create_project():
    """Create all project files and directories"""
    print("🚀 Creating Hyperliquid Trading Bot project...")
    
    # Create all files
    for filepath, content in FILES.items():
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Write file content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Created: {filepath}")
    
    # Create empty directories
    empty_dirs = [
        "logs",
        "data/exports",
        "monitoring",
        "notebooks"
    ]
    
    for dir_path in empty_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    # Make scripts executable
    for script in ["setup.sh", "create_project.py"]:
        if os.path.exists(script):
            os.chmod(script, 0o755)
    
    print("\n✨ Project created successfully!")
    print("\nNext steps:")
    print("1. Run ./setup.sh to install dependencies")
    print("2. Configure your settings in configs/config.yaml")
    print("3. Add your Hyperliquid private key to .env")
    print("4. Run python src/main.py to start trading")

if __name__ == "__main__":
    create_project()
