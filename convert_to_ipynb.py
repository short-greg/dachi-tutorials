#!/usr/bin/env python3
"""
convert_to_ipynb.py - Convert structured Python files to Jupyter notebooks

This script converts specially formatted Python files into Jupyter notebook (.ipynb) format.
The input files must follow a specific structure using cell markers and type indicators.

USAGE:
    python convert_to_ipynb.py <filename1>.py [<filename2>.py ...]

EXAMPLES:
    python convert_to_ipynb.py tutorial1.py
    python convert_to_ipynb.py tutorial1.py tutorial2.py tutorial3.py
    
OUTPUT:
    Creates corresponding .ipynb files:
    tutorial1.py -> tutorial1.ipynb
    tutorial2.py -> tutorial2.ipynb

REQUIRED INPUT FILE FORMAT:
============================

Input Python files must follow this exact structure:

1. CELL SEPARATORS:
   - Each cell must start with: # %%
   - This marks the beginning of a new notebook cell

2. CELL TYPE INDICATORS:
   - Immediately after # %%, add one of these type indicators:
   
   a) For Markdown cells:
      # %%
      # Markdown
      \"\"\"
      Your markdown content here
      Can span multiple lines
      Supports all markdown syntax
      \"\"\"
   
   b) For Code cells:
      # %%
      # Python
      your_python_code_here()
      print("Hello World")
      # More code...

3. CONTENT RULES:
   - Markdown content MUST be enclosed in triple quotes (\"\"\")
   - Python code should be valid Python syntax
   - Empty lines are preserved in both markdown and code cells
   - Comments in Python cells are preserved as code

EXAMPLE INPUT FILE:
==================

# %%
# Markdown
\"\"\"
# My Tutorial

This is an introduction to the tutorial.

## Section 1
Let's start with some basics.
\"\"\"

# %%
# Python
import numpy as np
import matplotlib.pyplot as plt

# This is a code comment
x = np.linspace(0, 10, 100)
y = np.sin(x)

# %%
# Markdown
\"\"\"
Now let's plot the results:
\"\"\"

# %%
# Python
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Sine Wave")
plt.show()

FEATURES:
=========

- Preserves all content exactly as written
- Maintains proper indentation and formatting
- Handles multiple files in a single command
- Creates standard Jupyter notebook format (.ipynb)
- Validates input file format before conversion
- Provides detailed error messages for format issues
- Supports Unicode content
- Preserves empty lines and whitespace

ERROR HANDLING:
===============

The script will report specific errors if:
- File format doesn't match required structure
- Missing cell type indicators
- Malformed markdown triple quotes
- Invalid cell separators
- File read/write permissions issues

TECHNICAL DETAILS:
==================

- Output format: Jupyter Notebook v4.5+ compatible
- Encoding: UTF-8
- Cell execution count: Reset to null (not executed)
- Metadata: Minimal required metadata only
- Python version: Compatible with Python 3.6+

DEPENDENCIES:
=============

- json (standard library)
- sys (standard library)
- os (standard library)
- re (standard library)
- argparse (standard library)

No external dependencies required.

AUTHOR: Generated for Dachi Framework Tutorial System
VERSION: 1.0
"""

import json
import sys
import os
import re
import argparse
from typing import List, Dict, Any, Tuple, Optional


class NotebookConverter:
    """
    Converts specially formatted Python files to Jupyter notebooks.
    
    This class handles the parsing of structured Python files and conversion
    to the standard Jupyter notebook format.
    """
    
    def __init__(self):
        """Initialize the converter with default notebook structure."""
        self.notebook_template = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }
    
    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Parse a structured Python file into notebook cells.
        
        Args:
            filepath: Path to the input Python file
            
        Returns:
            List of cell dictionaries in Jupyter format
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Error reading file {filepath}: {e}")
        
        return self._parse_content(content, filepath)
    
    def _parse_content(self, content: str, filepath: str) -> List[Dict[str, Any]]:
        """
        Parse file content into notebook cells.
        
        Args:
            content: The file content as a string
            filepath: Original filepath for error reporting
            
        Returns:
            List of cell dictionaries
        """
        cells = []
        
        # Split content by cell markers
        cell_pattern = r'^# %%\s*$'
        parts = re.split(cell_pattern, content, flags=re.MULTILINE)
        
        # Remove empty first part if file starts with cell marker
        if parts and not parts[0].strip():
            parts = parts[1:]
        
        if not parts:
            raise ValueError(f"No cells found in {filepath}. File must contain '# %%' cell markers.")
        
        for i, part in enumerate(parts):
            if not part.strip():
                continue
                
            try:
                cell = self._parse_cell(part.strip(), i + 1, filepath)
                if cell:
                    cells.append(cell)
            except Exception as e:
                raise ValueError(f"Error parsing cell {i + 1} in {filepath}: {e}")
        
        if not cells:
            raise ValueError(f"No valid cells found in {filepath}")
        
        return cells
    
    def _parse_cell(self, cell_content: str, cell_number: int, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Parse individual cell content.
        
        Args:
            cell_content: Raw cell content
            cell_number: Cell number for error reporting
            filepath: Filepath for error reporting
            
        Returns:
            Cell dictionary or None if empty
        """
        lines = cell_content.split('\n')
        
        if not lines:
            return None
        
        # Check for type indicator
        type_indicator = lines[0].strip()
        
        if type_indicator == '# Markdown':
            return self._create_markdown_cell(lines[1:], cell_number, filepath)
        elif type_indicator == '# Python':
            return self._create_code_cell(lines[1:], cell_number, filepath)
        else:
            raise ValueError(
                f"Invalid cell type indicator in cell {cell_number}: '{type_indicator}'. "
                f"Must be '# Markdown' or '# Python'"
            )
    
    def _create_markdown_cell(self, lines: List[str], cell_number: int, filepath: str) -> Dict[str, Any]:
        """
        Create a markdown cell from content lines.
        
        Args:
            lines: Content lines (excluding type indicator)
            cell_number: Cell number for error reporting
            filepath: Filepath for error reporting
            
        Returns:
            Markdown cell dictionary
        """
        # Join lines and look for triple quotes
        content = '\n'.join(lines)
        
        # Find triple-quoted content
        quote_pattern = r'^\s*"""\s*\n(.*?)\n\s*"""\s*$'
        match = re.search(quote_pattern, content, re.DOTALL)
        
        if not match:
            raise ValueError(
                f"Markdown cell {cell_number} in {filepath} must have content enclosed in triple quotes (\"\"\"). "
                f"Format:\n# Markdown\n\"\"\"\nYour content here\n\"\"\""
            )
        
        markdown_content = match.group(1)
        
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": markdown_content.split('\n')
        }
    
    def _create_code_cell(self, lines: List[str], cell_number: int, filepath: str) -> Dict[str, Any]:
        """
        Create a code cell from content lines.
        
        Args:
            lines: Content lines (excluding type indicator)
            cell_number: Cell number for error reporting
            filepath: Filepath for error reporting
            
        Returns:
            Code cell dictionary
        """
        # Remove leading/trailing empty lines but preserve internal ones
        while lines and not lines[0].strip():
            lines = lines[1:]
        while lines and not lines[-1].strip():
            lines = lines[:-1]
        
        if not lines:
            # Empty code cell
            source_lines = []
        else:
            source_lines = lines
        
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines
        }
    
    def convert_file(self, input_filepath: str, output_filepath: Optional[str] = None) -> str:
        """
        Convert a single Python file to Jupyter notebook.
        
        Args:
            input_filepath: Path to input Python file
            output_filepath: Path for output notebook (optional)
            
        Returns:
            Path to created notebook file
        """
        if output_filepath is None:
            # Generate output filename
            base_name = os.path.splitext(input_filepath)[0]
            output_filepath = f"{base_name}.ipynb"
        
        print(f"Converting {input_filepath} -> {output_filepath}")
        
        try:
            # Parse the input file
            cells = self.parse_file(input_filepath)
            
            # Create notebook structure
            notebook = self.notebook_template.copy()
            notebook["cells"] = cells
            
            # Write notebook file
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Successfully created {output_filepath} with {len(cells)} cells")
            return output_filepath
            
        except Exception as e:
            print(f"✗ Error converting {input_filepath}: {e}")
            raise
    
    def convert_multiple(self, input_filepaths: List[str]) -> List[str]:
        """
        Convert multiple Python files to notebooks.
        
        Args:
            input_filepaths: List of input file paths
            
        Returns:
            List of created notebook file paths
        """
        output_paths = []
        errors = []
        
        for filepath in input_filepaths:
            try:
                output_path = self.convert_file(filepath)
                output_paths.append(output_path)
            except Exception as e:
                errors.append(f"{filepath}: {e}")
        
        if errors:
            print(f"\n{len(errors)} conversion errors occurred:")
            for error in errors:
                print(f"  - {error}")
        
        print(f"\nConversion complete: {len(output_paths)}/{len(input_filepaths)} files successful")
        return output_paths


def validate_python_file(filepath: str) -> bool:
    """
    Validate that a file has .py extension and exists.
    
    Args:
        filepath: Path to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not filepath.endswith('.py'):
        print(f"Warning: {filepath} does not have .py extension")
        return False
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False
    
    return True


def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert structured Python files to Jupyter notebooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python convert_to_ipynb.py tutorial1.py
  python convert_to_ipynb.py tutorial1.py tutorial2.py tutorial3.py

INPUT FILE FORMAT:
  Files must use # %% cell markers with # Markdown or # Python type indicators.
  Markdown content must be enclosed in triple quotes.
  
  See script docstring for detailed format requirements.
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='Python files to convert to notebooks'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate file format without converting'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed conversion information'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    valid_files = []
    for filepath in args.files:
        if validate_python_file(filepath):
            valid_files.append(filepath)
    
    if not valid_files:
        print("No valid input files found.")
        sys.exit(1)
    
    # Create converter
    converter = NotebookConverter()
    
    if args.validate_only:
        print("Validating file formats...")
        errors = 0
        for filepath in valid_files:
            try:
                cells = converter.parse_file(filepath)
                print(f"✓ {filepath}: Valid format ({len(cells)} cells)")
            except Exception as e:
                print(f"✗ {filepath}: {e}")
                errors += 1
        
        if errors:
            print(f"\n{errors} validation errors found.")
            sys.exit(1)
        else:
            print(f"\nAll {len(valid_files)} files have valid format.")
            sys.exit(0)
    
    # Convert files
    try:
        output_paths = converter.convert_multiple(valid_files)
        
        if args.verbose:
            print(f"\nCreated notebooks:")
            for path in output_paths:
                print(f"  - {path}")
        
        if len(output_paths) == len(valid_files):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nConversion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()