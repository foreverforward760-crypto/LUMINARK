#!/usr/bin/env python3
"""
LUMINARK AI Framework - Web Dashboard
Real-time monitoring and visualization of SAP framework stages
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from luminark_omega.core.sar_framework import SARFramework
except ImportError:
    print("⚠️ SAP Framework not found, using mock")
    SARFramework = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'luminark_secret_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize SAP Framework
sar = SARFramework() if SARFramework else None

# Global state
current_state = {
    'stage': 4,
    'stage_name': 'Foundation',
    'physical_state': 'stable',
    'conscious_state': 'unstable',
    'inversion_level': 0,
    'resonance': 0.5,
    'timestamp': datetime.now().isoformat()
}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify(current_state)

@app.route('/api/stages')
def get_stages():
    """Get all SAP stages"""
    if sar:
        stages = []
        for level in range(10):
            stage = sar.get_stage(level)
            stages.append({
                'level': stage.level,
                'name': stage.name,
                'description': stage.description,
                'energy_signature': stage.energy_signature,
                'physical_state': stage.physical_state,
                'conscious_state': stage.conscious_state,
                'is_inverted': stage.is_inverted
            })
        return jsonify({'stages': stages})
    return jsonify({'stages': []})

@app.route('/api/detect_inversion', methods=['POST'])
def detect_inversion():
    """Detect inversion based on physical and conscious states"""
    data = request.json
    physical_stable = data.get('physical_stable', True)
    conscious_stable = data.get('conscious_stable', True)
    
    if sar:
        result = sar.detect_inversion(physical_stable, conscious_stable)
        current_state.update(result)
        current_state['timestamp'] = datetime.now().isoformat()
        
        # Broadcast update to all connected clients
        socketio.emit('state_update', current_state)
        
        return jsonify(result)
    
    return jsonify({'error': 'SAR Framework not available'})

@app.route('/api/transition', methods=['POST'])
def transition_stage():
    """Transition to a new stage"""
    data = request.json
    resonance = data.get('resonance', 0.5)
    
    if sar:
        new_stage = sar.assess_transition(current_state['stage'], resonance)
        stage_obj = sar.get_stage(new_stage)
        
        current_state.update({
            'stage': new_stage,
            'stage_name': stage_obj.name,
            'physical_state': stage_obj.physical_state,
            'conscious_state': stage_obj.conscious_state,
            'resonance': resonance,
            'timestamp': datetime.now().isoformat()
        })
        
        # Broadcast update
        socketio.emit('state_update', current_state)
        
        return jsonify(current_state)
    
    return jsonify({'error': 'SAR Framework not available'})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('state_update', current_state)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_update')
def handle_update_request():
    """Handle update request from client"""
    emit('state_update', current_state)

if __name__ == '__main__':
    print("="*60)
    print("🌟 LUMINARK AI FRAMEWORK - Web Dashboard")
    print("="*60)
    print(f"📊 Dashboard: http://localhost:5000")
    print(f"🔌 WebSocket: ws://localhost:5000")
    print("="*60)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
