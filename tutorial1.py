"""
Tutorial 1: Multi-Document Analysis Pipeline

This is a thin wrapper that launches the multi_document tutorial.

Usage: streamlit run tutorial1.py
"""

import sys
from pathlib import Path

# Add tutorials directory to path
tutorials_path = Path(__file__).parent / "tutorials" / "multi_document"
sys.path.insert(0, str(tutorials_path))

# Import and run the main tutorial
if __name__ == "__main__":
    from main import main
    main()