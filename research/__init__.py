# Automatically configure PyTorch to use CPU when research module is imported
try:
    import sys
    from pathlib import Path
    # Add project root to path if not already there
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import pytorch_config
except ImportError:
    # PyTorch may not be installed, which is fine for non-ML components
    pass