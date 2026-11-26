#!/usr/bin/env python3
"""
LUMINARK AI Framework - Web Dashboard

Experimental dashboard for viewing and interacting with the SAP stage model.
This module is a prototype, not a validated safety, clinical, or predictive system.
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import sys
import os
import secrets
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from luminark_omega.core.sar_framework import SARFramework
except ImportError:
    print("SAP Framework not found; stage endpoints will be unavailable.")
    SARFramework = None

app = Flask(__name__)
# Set LUMINARK_SECRET_KEY in the environment for a stable key across restarts.
# The generated fallback is suitable only for local development and invalidates
# signed sessions whenever the process restarts.
app.config["SECRET_KEY"] = os.getenv("LUMINARK_SECRET_KEY") or secrets.token_urlsafe(32)

# Same-origin access is the default. Set a comma-separated allowlist only when
# a separately hosted frontend is intentionally configured to access this server.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("LUMINARK_CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
socketio = SocketIO(app, cors_allowed_origins=allowed_origins or None)

# Initialize SAP Framework
sar = SARFramework() if SARFramework else None

# Global state
current_state = {
    "stage": 4,
    "stage_name": "CRUCIBLE OF EQUILIBRIUM",
    "physical_state": "stable",
    "conscious_state": "unstable",
    "inversion_level": 0,
    "resonance": 0.5,
    "timestamp": datetime.now().isoformat(),
}


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("dashboard.html")


@app.route("/api/status")
def get_status():
    """Get current system status."""
    return jsonify(current_state)


@app.route("/api/stages")
def get_stages():
    """Get all SAP stages."""
    if sar:
        stages = []
        for level in range(10):
            stage = sar.get_stage(level)
            stages.append(
                {
                    "level": stage.level,
                    "name": stage.name,
                    "description": stage.description,
                    "energy_signature": stage.energy_signature,
                    "physical_state": stage.physical_state,
                    "conscious_state": stage.conscious_state,
                    "is_inverted": stage.is_inverted,
                }
            )
        return jsonify({"stages": stages})
    return jsonify({"error": "SAP Framework not available"}), 503


@app.route("/api/detect_inversion", methods=["POST"])
def detect_inversion():
    """Detect inversion based on physical and conscious states."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    physical_stable = data.get("physical_stable", True)
    conscious_stable = data.get("conscious_stable", True)
    if not isinstance(physical_stable, bool) or not isinstance(conscious_stable, bool):
        return jsonify({"error": "physical_stable and conscious_stable must be booleans"}), 400

    if not sar:
        return jsonify({"error": "SAP Framework not available"}), 503

    result = sar.detect_inversion(physical_stable, conscious_stable)
    current_state.update(result)
    current_state["timestamp"] = datetime.now().isoformat()
    socketio.emit("state_update", current_state)
    return jsonify(result)


@app.route("/api/transition", methods=["POST"])
def transition_stage():
    """Transition to a new stage using the framework's configured rule."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    resonance = data.get("resonance", 0.5)
    if isinstance(resonance, bool) or not isinstance(resonance, (int, float)):
        return jsonify({"error": "resonance must be a number between 0 and 1"}), 400
    if not 0 <= resonance <= 1:
        return jsonify({"error": "resonance must be between 0 and 1"}), 400

    if not sar:
        return jsonify({"error": "SAP Framework not available"}), 503

    new_stage = sar.assess_transition(current_state["stage"], float(resonance))
    stage_obj = sar.get_stage(new_stage)
    current_state.update(
        {
            "stage": new_stage,
            "stage_name": stage_obj.name,
            "physical_state": stage_obj.physical_state,
            "conscious_state": stage_obj.conscious_state,
            "resonance": float(resonance),
            "timestamp": datetime.now().isoformat(),
        }
    )
    socketio.emit("state_update", current_state)
    return jsonify(current_state)


@socketio.on("connect")
def handle_connect():
    """Send current state to a newly connected client."""
    emit("state_update", current_state)


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    return None


@socketio.on("request_update")
def handle_update_request():
    """Send the current state when requested by a connected client."""
    emit("state_update", current_state)


if __name__ == "__main__":
    host = os.getenv("LUMINARK_HOST", "127.0.0.1")
    port = int(os.getenv("LUMINARK_PORT", "5000"))
    debug = os.getenv("LUMINARK_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    print("=" * 60)
    print("LUMINARK experimental web dashboard")
    print(f"Dashboard: http://{host}:{port}")
    print("=" * 60)
    socketio.run(app, host=host, port=port, debug=debug)
