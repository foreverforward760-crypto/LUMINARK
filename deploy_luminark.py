#!/usr/bin/env python3
"""
LUMINARK AI Framework - One-Click Deployment Script
Automates setup, dependency installation, and system initialization
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

class LuminarkDeployer:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.requirements = self.project_root / "requirements.txt"
        
    def check_python(self):
        """Check Python version"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8+ required")
            return False
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    
    def create_virtual_environment(self):
        """Create virtual environment"""
        if self.venv_path.exists():
            print("✅ Virtual environment already exists")
            return True
        
        print("Creating virtual environment...")
        result = subprocess.run([
            sys.executable, "-m", "venv", str(self.venv_path)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Virtual environment created")
            return True
        else:
            print(f"❌ Failed to create virtual environment: {result.stderr}")
            return False
    
    def install_dependencies(self):
        """Install all dependencies"""
        if not self.requirements.exists():
            print("⚠️ requirements.txt not found, creating basic one...")
            self.create_requirements()
        
        # Determine pip path
        pip_path = self.venv_path / "bin" / "pip"
        if sys.platform == "win32":
            pip_path = self.venv_path / "Scripts" / "pip.exe"
        
        print("Installing dependencies...")
        result = subprocess.run([
            str(pip_path), "install", "-r", str(self.requirements)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
    
    def create_requirements(self):
        """Create requirements.txt if missing"""
        requirements_content = """# LUMINARK AI Framework Dependencies

# Core ML/AI
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Web Framework
flask>=2.3.0
flask-socketio>=5.3.0
python-socketio>=5.9.0

# Visualization
matplotlib>=3.7.0
plotly>=5.15.0

# Utilities
pydantic>=2.0.0
python-dotenv>=1.0.0

# Development
pytest>=7.4.0
black>=23.7.0
"""
        self.requirements.write_text(requirements_content)
        print("✅ Created requirements.txt")
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            "web_dashboard/templates",
            "web_dashboard/static/css",
            "web_dashboard/static/js",
            "luminark/biofeedback",
            "logs",
            "data",
            "visualizations",
            "models"
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_name}")
        
        return True
    
    def generate_config(self):
        """Generate configuration file"""
        config_content = """# LUMINARK AI Framework Configuration

[server]
host = "0.0.0.0"
port = 5000
debug = true
secret_key = "luminark_secret_2024"

[sar_framework]
default_stage = 4
enable_inversion_detection = true
auto_transition = true

[biofeedback]
update_interval = 1.0
hrv_threshold_low = 30
hrv_threshold_high = 100
enable_stress_detection = true

[visualization]
update_interval = 5
max_data_points = 1000
enable_real_time = true

[logging]
level = "INFO"
file = "logs/luminark.log"
max_size_mb = 10
backup_count = 5
"""
        
        config_path = self.project_root / "config.ini"
        config_path.write_text(config_content)
        print("✅ Configuration file generated")
        
        return True
    
    def run_tests(self):
        """Run basic tests"""
        print("Running basic tests...")
        
        # Test imports
        test_code = """
import sys
try:
    import torch
    import numpy as np
    import flask
    print("✅ All core imports successful")
except ImportError as e:
    print(f"❌ Import test failed: {e}")
    sys.exit(1)
"""
        
        python_path = self.venv_path / "bin" / "python"
        if sys.platform == "win32":
            python_path = self.venv_path / "Scripts" / "python.exe"
        
        result = subprocess.run([
            str(python_path), "-c", test_code
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(f"❌ Import test failed: {result.stderr}")
            return False
    
    def start_system(self):
        """Start the LUMINARK system"""
        print("\n" + "="*50)
        print("🚀 STARTING LUMINARK AI FRAMEWORK")
        print("="*50)
        
        python_path = self.venv_path / "bin" / "python"
        if sys.platform == "win32":
            python_path = self.venv_path / "Scripts" / "python.exe"
        
        # Start the web dashboard
        app_path = self.project_root / "web_dashboard" / "app.py"
        
        if not app_path.exists():
            print("⚠️ Web dashboard not found. Run deployment first.")
            return False
        
        print(f"\nStarting web dashboard at: http://localhost:5000")
        print("Press Ctrl+C to stop\n")
        
        try:
            subprocess.run([
                str(python_path), str(app_path)
            ])
        except KeyboardInterrupt:
            print("\n🛑 System stopped by user")
        
        return True
    
    def deploy(self):
        """Complete deployment process"""
        print("="*50)
        print("LUMINARK AI FRAMEWORK DEPLOYMENT")
        print("="*50)
        
        steps = [
            ("Checking Python version", self.check_python),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Setting up directories", self.setup_directories),
            ("Generating configuration", self.generate_config),
            ("Running tests", self.run_tests)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if not step_func():
                print(f"❌ Deployment failed at: {step_name}")
                return False
        
        print("\n" + "="*50)
        print("✅ DEPLOYMENT COMPLETE!")
        print("="*50)
        
        # Ask to start system
        response = input("\nStart the system now? (y/n): ").strip().lower()
        if response == 'y':
            self.start_system()
        
        return True

if __name__ == "__main__":
    deployer = LuminarkDeployer()
    deployer.deploy()
