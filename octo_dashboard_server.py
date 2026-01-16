#!/usr/bin/env python3
"""
LUMINARK Dashboard Server - Web Interface for Real-time Monitoring
"""
from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
import random
import time
from datetime import datetime
import os
import threading
import json


app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
CORS(app)

# Global state for demo data
demo_state = {
    'running': True,
    'start_time': time.time(),
    'iterations': 0,
    'metrics_history': [],
    'max_history': 100,
}


def generate_metrics():
    """Generate simulated metrics"""
    return {
        'timestamp': datetime.now().isoformat(),
        'accuracy': random.uniform(0.85, 0.99),
        'loss': random.uniform(0.01, 0.15),
        'throughput': random.uniform(100, 500),
        'memory_usage': random.uniform(1024, 4096),
        'cpu_usage': random.uniform(20, 80),
        'iteration': demo_state['iterations'],
    }


def background_worker():
    """Background thread to generate metrics"""
    while demo_state['running']:
        demo_state['iterations'] += 1
        metrics = generate_metrics()

        # Add to history
        demo_state['metrics_history'].append(metrics)

        # Trim history to max size
        if len(demo_state['metrics_history']) > demo_state['max_history']:
            demo_state['metrics_history'] = demo_state['metrics_history'][-demo_state['max_history']:]

        time.sleep(2)  # Update every 2 seconds


@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('index.html')


@app.route('/api/metrics')
def get_metrics():
    """API endpoint to get current metrics"""
    if demo_state['metrics_history']:
        current = demo_state['metrics_history'][-1]
    else:
        current = generate_metrics()

    return jsonify({
        'current': current,
        'uptime': time.time() - demo_state['start_time'],
        'total_iterations': demo_state['iterations'],
    })


@app.route('/api/history')
def get_history():
    """API endpoint to get metrics history"""
    return jsonify({
        'history': demo_state['metrics_history'],
        'count': len(demo_state['metrics_history']),
    })


@app.route('/api/status')
def get_status():
    """API endpoint to get system status"""
    uptime = time.time() - demo_state['start_time']

    return jsonify({
        'status': 'running',
        'uptime': uptime,
        'uptime_formatted': f"{int(uptime // 60)}m {int(uptime % 60)}s",
        'iterations': demo_state['iterations'],
        'start_time': datetime.fromtimestamp(demo_state['start_time']).isoformat(),
    })


@app.route('/api/reset')
def reset():
    """API endpoint to reset metrics"""
    demo_state['iterations'] = 0
    demo_state['metrics_history'] = []
    demo_state['start_time'] = time.time()

    return jsonify({
        'status': 'reset',
        'message': 'Metrics have been reset',
    })


def main():
    """Main entry point"""
    # Create directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    print("=" * 70)
    print("LUMINARK Dashboard Server")
    print("=" * 70)
    print(f"Starting server at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Dashboard will be available at:")
    print("  → http://localhost:8000")
    print("  → http://127.0.0.1:8000")
    print()
    print("API Endpoints:")
    print("  → GET /api/metrics   - Current metrics")
    print("  → GET /api/history   - Metrics history")
    print("  → GET /api/status    - System status")
    print("  → GET /api/reset     - Reset metrics")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()

    # Start background worker
    worker_thread = threading.Thread(target=background_worker, daemon=True)
    worker_thread.start()

    # Start Flask server
    try:
        app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        demo_state['running'] = False
    finally:
        print("Server stopped.")


if __name__ == '__main__':
    main()
