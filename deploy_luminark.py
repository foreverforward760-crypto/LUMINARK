#!/usr/bin/env python3
"""
LUMINARK AI Framework - Local Development Setup

Creates a virtual environment, installs the declared requirements, and
creates local directories/configuration. This is not a production deployment
or a security/validation certification.
"""

import os
import sys
import subprocess
from pathlib import Path
import secrets


class LuminarkDeployer:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.requirements = self.project_root / "requirements.txt"

    def check_python(self):
        """Check Python version."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("Python 3.8+ required")
            return False
        print(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True

    def create_virtual_environment(self):
        """Create a virtual environment."""
        if self.venv_path.exists():
            print("Virtual environment already exists")
            return True

        print("Creating virtual environment...")
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(self.venv_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Virtual environment created")
            return True

        print(f"Failed to create virtual environment: {result.stderr}")
        return False

    def install_dependencies(self):
        """Install dependencies from the repository's requirements file."""
        if not self.requirements.exists():
            print("requirements.txt not found; refusing to invent a dependency set.")
            return False

        pip_path = self.venv_path / "bin" / "pip"
        if sys.platform == "win32":
            pip_path = self.venv_path / "Scripts" / "pip.exe"

        if not pip_path.exists():
            print(f"pip executable not found: {pip_path}")
            return False

        print("Installing dependencies...")
        result = subprocess.run(
            [str(pip_path), "install", "-r", str(self.requirements)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Dependencies installed")
            return True

        print(f"Failed to install dependencies: {result.stderr}")
        return False

    def setup_directories(self):
        """Create local runtime directories."""
        directories = [
            "web_dashboard/templates",
            "web_dashboard/static/css",
            "web_dashboard/static/js",
            "luminark/biofeedback",
            "logs",
            "data",
            "visualizations",
            "models",
        ]
        for dir_name in directories:
            (self.project_root / dir_name).mkdir(parents=True, exist_ok=True)
        return True

    def generate_config(self):
        """Generate a local-only configuration with a fresh secret."""
        config_content = f'''# LUMINARK local development configuration

[server]
host = "127.0.0.1"
port = 5000
debug = false
secret_key = "{secrets.token_urlsafe(32)}"

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
'''
        config_path = self.project_root / "config.ini"
        config_path.write_text(config_content, encoding="utf-8")
        print("Local development configuration generated")
        print("Keep config.ini private; do not commit generated secrets.")
        return True

    def run_tests(self):
        """Run a minimal import check; this is not a product validation test."""
        print("Running basic import checks...")
        test_code = """
import sys
try:
    import flask
    import flask_socketio
    import requests
except ImportError as exc:
    print(f"Import check failed: {exc}")
    sys.exit(1)
print("Core imports successful")
"""
        python_path = self.venv_path / "bin" / "python"
        if sys.platform == "win32":
            python_path = self.venv_path / "Scripts" / "python.exe"
        result = subprocess.run(
            [str(python_path), "-c", test_code],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(result.stdout.strip())
            return True

        print(f"Import check failed: {result.stderr}")
        return False

    def start_system(self):
        """Start the local dashboard if its application file exists."""
        python_path = self.venv_path / "bin" / "python"
        if sys.platform == "win32":
            python_path = self.venv_path / "Scripts" / "python.exe"

        app_path = self.project_root / "web_dashboard" / "app.py"
        if not app_path.exists():
            print("Web dashboard not found.")
            return False

        env = os.environ.copy()
        env.setdefault("LUMINARK_HOST", "127.0.0.1")
        env.setdefault("LUMINARK_DEBUG", "false")
        print(f"Starting local dashboard at http://{env['LUMINARK_HOST']}:5000")
        print("Press Ctrl+C to stop.")
        try:
            subprocess.run([str(python_path), str(app_path)], env=env, check=False)
        except KeyboardInterrupt:
            print("Dashboard stopped.")
        return True

    def deploy(self):
        """Set up the local development environment."""
        steps = [
            ("Checking Python version", self.check_python),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing declared dependencies", self.install_dependencies),
            ("Setting up local directories", self.setup_directories),
            ("Generating local configuration", self.generate_config),
            ("Running import checks", self.run_tests),
        ]
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if not step_func():
                print(f"Setup stopped at: {step_name}")
                return False

        print("\nLocal development setup completed.")
        response = input("Start the local dashboard now? (y/n): ").strip().lower()
        if response == "y":
            self.start_system()
        return True


if __name__ == "__main__":
    LuminarkDeployer().deploy()
