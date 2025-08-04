#!/usr/bin/env python3
"""
Dachi Tutorials Launcher

Usage: python run.py <tutorial_name>

Examples:
    python run.py multi_document
"""

import sys
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <tutorial_name>")
        print("\nAvailable tutorials:")
        tutorials_dir = Path("tutorials")
        if tutorials_dir.exists():
            for tutorial in tutorials_dir.iterdir():
                if tutorial.is_dir() and (tutorial / "main.py").exists():
                    print(f"  - {tutorial.name}")
        sys.exit(1)
    
    tutorial_name = sys.argv[1]
    tutorial_dir = Path("tutorials") / tutorial_name
    
    if not tutorial_dir.exists():
        print(f"Tutorial '{tutorial_name}' not found in tutorials/")
        sys.exit(1)
    
    main_file = tutorial_dir / "main.py"
    if not main_file.exists():
        print(f"No main.py found in tutorials/{tutorial_name}/")
        sys.exit(1)
    
    print(f"Launching {tutorial_name} tutorial...")
    
    # Run streamlit with the tutorial's main.py
    cmd = ["streamlit", "run", str(main_file)] + sys.argv[2:]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()