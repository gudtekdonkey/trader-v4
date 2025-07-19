"""
Import tests for all packages in the trading bot system.
This ensures all modules can be imported successfully and have proper __init__.py files.
"""

import sys
import os
import importlib
import traceback
from typing import List, Tuple, Dict
from pathlib import Path


class ImportTester:
    """Test all imports in the trading bot system"""
    
    def __init__(self, src_path: str = None):
        """Initialize the import tester
        
        Args:
            src_path: Path to the src directory
        """
        if src_path is None:
            # Assume we're in src/tests, go up one level
            self.src_path = Path(__file__).parent.parent.absolute()
        else:
            self.src_path = Path(src_path).absolute()
            
        # Add src to Python path if not already there
        src_str = str(self.src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
            
        self.results: Dict[str, Tuple[bool, str]] = {}
        
    def test_module_imports(self) -> Dict[str, Tuple[bool, str]]:
        """Test importing all main modules and their submodules
        
        Returns:
            Dictionary of module_name -> (success, error_message)
        """
        # Define all modules to test
        modules_to_test = [
            # Main module
            'main',
            'main.modules',
            
            # Data modules
            'data',
            'data.collector',
            'data.collector.modules',
            'data.feature_engineer',
            'data.feature_engineer.modules',
            
            # Exchange modules
            'exchange',
            'exchange.hyperliquid_client',
            'exchange.hyperliquid_client.modules',
            
            # Model modules
            'models',
            'models.ensemble',
            'models.ensemble.modules',
            'models.lstm_attention',
            'models.lstm_attention.modules',
            'models.temporal_fusion_transformer',
            'models.temporal_fusion_transformer.modules',
            'models.reinforcement_learning',
            'models.reinforcement_learning.modules',
            
            # Trading modules
            'trading',
            'trading.risk_manager',
            'trading.risk_manager.modules',
            'trading.position_sizer',
            'trading.position_sizer.modules',
            'trading.adaptive_strategy_manager',
            'trading.adaptive_strategy_manager.modules',
            'trading.regime_detector',
            'trading.regime_detector.modules',
            'trading.order_executor',
            'trading.order_executor.modules',
            'trading.dynamic_hedging',
            'trading.dynamic_hedging.modules',
            'trading.execution',
            'trading.execution.advanced_executor',
            'trading.execution.advanced_executor.modules',
            'trading.strategies',
            'trading.strategies.momentum',
            'trading.strategies.momentum.modules',
            'trading.strategies.mean_reversion',
            'trading.strategies.mean_reversion.modules',
            'trading.strategies.arbitrage',
            'trading.strategies.arbitrage.modules',
            'trading.strategies.market_making',
            'trading.strategies.market_making.modules',
            'trading.portfolio',
            'trading.portfolio.analytics',
            'trading.portfolio.analytics.modules',
            'trading.portfolio.monitor',
            'trading.portfolio.monitor.modules',
            'trading.optimization',
            'trading.optimization.black_litterman',
            'trading.optimization.black_litterman.modules',
            'trading.optimization.hierarchical_risk_parity',
            'trading.optimization.hierarchical_risk_parity.modules',
            
            # Utils modules
            'utils',
            'utils.database',
            'utils.database.modules',
            'utils.tg_notifications',
            'utils.tg_notifications.modules',
            
            # Templates (not a Python package, but check if directory exists)
            # 'templates',
            # 'templates.dashboard',
        ]
        
        # Test each module
        for module_name in modules_to_test:
            self.results[module_name] = self._test_single_import(module_name)
            
        return self.results
    
    def _test_single_import(self, module_name: str) -> Tuple[bool, str]:
        """Test importing a single module
        
        Args:
            module_name: Name of the module to import
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            
            # Check if it has __all__ defined (good practice)
            if hasattr(module, '__all__'):
                return (True, f"Success - __all__ defined with {len(module.__all__)} exports")
            else:
                return (True, "Success - No __all__ defined")
                
        except ImportError as e:
            return (False, f"ImportError: {str(e)}")
        except Exception as e:
            return (False, f"{type(e).__name__}: {str(e)}")
    
    def test_specific_imports(self) -> Dict[str, Tuple[bool, str]]:
        """Test specific imports that should work after modularization"""
        
        specific_imports = [
            # Main imports
            ('from main import HyperliquidTradingBot', 'HyperliquidTradingBot'),
            ('from main.modules import HealthMonitor', 'HealthMonitor'),
            ('from main.modules import ComponentInitializer', 'ComponentInitializer'),
            
            # Data imports
            ('from data import DataCollector', 'DataCollector'),
            ('from data import DataPreprocessor', 'DataPreprocessor'),
            ('from data import FeatureEngineer', 'FeatureEngineer'),
            ('from data.collector import DataCollector', 'DataCollector'),
            ('from data.feature_engineer import FeatureEngineer', 'FeatureEngineer'),
            
            # Model imports - Updated to use EnsembleModel
            ('from models import AttentionLSTM', 'AttentionLSTM'),
            ('from models import TFTModel', 'TFTModel'),
            ('from models import EnsembleModel', 'EnsembleModel'),
            ('from models.ensemble import EnsembleModel', 'EnsembleModel'),
            ('from models.reinforcement_learning import MultiAgentTradingSystem', 'MultiAgentTradingSystem'),
            
            # Trading imports
            ('from trading.risk_manager import RiskManager', 'RiskManager'),
            ('from trading.position_sizer import PositionSizer', 'PositionSizer'),
            ('from trading.strategies.momentum import MomentumStrategy', 'MomentumStrategy'),
            ('from trading.strategies.mean_reversion import MeanReversionStrategy', 'MeanReversionStrategy'),
            ('from trading.strategies.arbitrage import ArbitrageStrategy', 'ArbitrageStrategy'),
            ('from trading.strategies.market_making import MarketMakingStrategy', 'MarketMakingStrategy'),
            
            # Portfolio optimization imports
            ('from trading.optimization import BlackLittermanOptimizer', 'BlackLittermanOptimizer'),
            ('from trading.optimization import HRPOptimizer', 'HRPOptimizer'),
            ('from trading.optimization.black_litterman import BlackLittermanOptimizer', 'BlackLittermanOptimizer'),
            ('from trading.optimization.hierarchical_risk_parity import HRPOptimizer', 'HRPOptimizer'),
            
            # Execution imports
            ('from trading.execution import AdvancedExecutor', 'AdvancedExecutor'),
            ('from trading.execution.advanced_executor import AdvancedExecutor', 'AdvancedExecutor'),
            
            # Exchange imports
            ('from exchange.hyperliquid_client import HyperliquidClient', 'HyperliquidClient'),
            
            # Utils imports
            ('from utils import Config', 'Config'),
            ('from utils import setup_logger', 'setup_logger'),
            ('from utils.database import DatabaseManager', 'DatabaseManager'),
            ('from utils.tg_notifications import TelegramNotifier', 'TelegramNotifier'),
        ]
        
        results = {}
        for import_statement, expected_name in specific_imports:
            try:
                # Create a new namespace for each import
                namespace = {}
                exec(import_statement, namespace)
                
                if expected_name in namespace:
                    results[import_statement] = (True, f"Successfully imported {expected_name}")
                else:
                    results[import_statement] = (False, f"{expected_name} not found in namespace")
                    
            except Exception as e:
                results[import_statement] = (False, f"{type(e).__name__}: {str(e)}")
                
        return results
    
    def generate_report(self) -> str:
        """Generate a formatted report of all import tests"""
        
        report_lines = [
            "=" * 80,
            "IMPORT TEST REPORT",
            "=" * 80,
            f"Source Path: {self.src_path}",
            "",
        ]
        
        # Module imports
        module_results = self.test_module_imports()
        report_lines.extend([
            "MODULE IMPORT TESTS:",
            "-" * 40,
        ])
        
        passed = 0
        failed = 0
        for module_name, (success, message) in sorted(module_results.items()):
            status = "✓ PASS" if success else "✗ FAIL"
            report_lines.append(f"{status} {module_name:<50} {message}")
            if success:
                passed += 1
            else:
                failed += 1
                
        report_lines.extend([
            "",
            f"Module Import Summary: {passed} passed, {failed} failed",
            "",
        ])
        
        # Specific imports
        specific_results = self.test_specific_imports()
        report_lines.extend([
            "SPECIFIC IMPORT TESTS:",
            "-" * 40,
        ])
        
        s_passed = 0
        s_failed = 0
        for import_stmt, (success, message) in sorted(specific_results.items()):
            status = "✓ PASS" if success else "✗ FAIL"
            # Truncate long import statements
            stmt_display = import_stmt if len(import_stmt) <= 60 else import_stmt[:57] + "..."
            report_lines.append(f"{status} {stmt_display:<65} {message}")
            if success:
                s_passed += 1
            else:
                s_failed += 1
                
        report_lines.extend([
            "",
            f"Specific Import Summary: {s_passed} passed, {s_failed} failed",
            "",
            "=" * 80,
            f"TOTAL: {passed + s_passed} passed, {failed + s_failed} failed",
            "=" * 80,
        ])
        
        return "\n".join(report_lines)
    
    def check_init_files(self) -> List[str]:
        """Check for missing __init__.py files in package directories"""
        
        missing_init_files = []
        
        # Walk through all directories in src
        for root, dirs, files in os.walk(self.src_path):
            # Skip __pycache__ and other special directories
            dirs[:] = [d for d in dirs if not d.startswith('__') and not d.startswith('.')]
            
            # Convert to Path for easier manipulation
            root_path = Path(root)
            
            # Skip the src directory itself and tests
            if root_path == self.src_path or 'tests' in root_path.parts:
                continue
                
            # Check if this directory has Python files
            python_files = [f for f in files if f.endswith('.py') and f != '__init__.py']
            
            if python_files:
                # This directory has Python files, check for __init__.py
                init_file = root_path / '__init__.py'
                if not init_file.exists():
                    # Make path relative to src for cleaner output
                    relative_path = root_path.relative_to(self.src_path)
                    missing_init_files.append(str(relative_path))
                    
        return missing_init_files


def main():
    """Run the import tests"""
    
    # Determine src path
    if len(sys.argv) > 1:
        src_path = sys.argv[1]
    else:
        # Try to find src directory
        current_file = Path(__file__).absolute()
        if current_file.parent.name == 'tests':
            src_path = current_file.parent.parent
        else:
            src_path = current_file.parent
            
    print(f"Testing imports from: {src_path}")
    print()
    
    # Create tester
    tester = ImportTester(src_path)
    
    # Check for missing __init__.py files
    print("Checking for missing __init__.py files...")
    missing_inits = tester.check_init_files()
    if missing_inits:
        print(f"WARNING: Found {len(missing_inits)} directories missing __init__.py:")
        for missing in missing_inits:
            print(f"  - {missing}")
        print()
    else:
        print("All package directories have __init__.py files ✓")
        print()
    
    # Generate and print report
    report = tester.generate_report()
    print(report)
    
    # Return exit code based on failures
    total_passed = sum(1 for r in tester.results.values() if r[0])
    total_failed = sum(1 for r in tester.results.values() if not r[0])
    
    if total_failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
