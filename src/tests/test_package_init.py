"""
Tests package initialization files (__init__.py).
"""


def test_imports():
    """Simple import test for pytest"""
    # Test main imports
    try:
        import main
        assert hasattr(main, '__version__')
        print("✓ main module imports successfully")
    except ImportError as e:
        print(f"✗ Failed to import main: {e}")
        
    # Test reinforcement learning imports  
    try:
        import models.reinforcement_learning
        assert hasattr(models.reinforcement_learning, '__version__')
        print("✓ reinforcement_learning module imports successfully")
    except ImportError as e:
        print(f"✗ Failed to import reinforcement_learning: {e}")
        
    # Test specific imports
    try:
        from main.modules import HealthMonitor
        print("✓ Can import HealthMonitor from main.modules")
    except ImportError as e:
        print(f"✗ Failed to import HealthMonitor: {e}")
        
    try:
        from models.reinforcement_learning import MultiAgentTradingSystem
        print("✓ Can import MultiAgentTradingSystem")
    except ImportError as e:
        print(f"✗ Failed to import MultiAgentTradingSystem: {e}")


if __name__ == "__main__":
    test_imports()
