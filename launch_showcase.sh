#!/bin/bash
# LUMINARK Showcase Dashboard Launcher
# Quick launch script for the complete demo

echo "🌟 LUMINARK Showcase Dashboard Launcher"
echo "========================================"
echo ""
echo "This will start the interactive demonstration dashboard."
echo "Open http://localhost:5001 in your browser after it starts."
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Flask not found. Installing..."
    pip install flask
fi

# Check if running in virtual environment (recommended)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "💡 Tip: Consider using a virtual environment"
    echo "   python -m venv venv && source venv/bin/activate"
    echo ""
fi

# Launch dashboard
echo "🚀 Launching dashboard..."
python3 examples/luminark_showcase_dashboard.py
