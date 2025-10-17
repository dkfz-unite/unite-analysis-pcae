import sys
from pathlib import Path


# Add src directory to Python path so imports work
src_path = (
    Path(__file__).parent.parent / "src"
)  # Go up one level from tests/ to project root, then to src/
sys.path.insert(0, str(src_path))
