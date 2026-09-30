import sys
from pathlib import Path

# Add project root to sys.path so pytest can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
