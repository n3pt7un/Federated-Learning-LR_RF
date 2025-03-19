#!/usr/bin/env python
"""
Dependency Installation Script for Visualization
-----------------------------------------------
This script installs the necessary dependencies for the visualization script.
"""

import subprocess
import sys
import os

def check_installed(package):
    """Check if package is already installed."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install required dependencies for visualization."""
    print("Checking and installing required dependencies...")
    
    # Required packages
    packages = [
        'pandas',
        'matplotlib',
        'seaborn',
        'numpy'
    ]
    
    # Check and install missing packages
    for package in packages:
        if not check_installed(package.split('==')[0]):
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {e}")
                print("Please install it manually using:")
                print(f"  pip install {package}")
        else:
            print(f"{package} is already installed.")
    
    print("\nAll dependencies installed successfully!")
    print("You can now run the visualization script:")
    print("  python visualize_results.py")

if __name__ == "__main__":
    install_dependencies() 