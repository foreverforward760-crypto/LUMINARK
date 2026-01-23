#!/usr/bin/env python3
"""
Real-Time Visualization Dashboard for Quantum Pattern Predictor
Shows training progress, predictions, quantum confidence, and awareness stages
"""
import sys
sys.path.insert(0, '/home/user/LUMINARK')

import numpy as np
import json
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import threading
import time

# Import our predictor
from quantum_pattern_predictor import QuantumAwarePredictor, generate_sample_data

app = Flask(__name__)
CORS(app)

# Global predictor instance
predictor = None
training_data = None
is_training = False
current_status = {
    'training_complete': False,
    'training_progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'predictions': [],
    'training_history': [],
    'meta_insights': {}
}


def train_in_background():
    """Train the predictor in a background thread"""
    global predictor, training_data, is_training, current_status

    is_training = True
    current_status['training_complete'] = False

    # Generate data
    print("Generating training data...")
    training_data = generate_sample_data(num_points=500, pattern='sine_trend')

    # Create predictor
    print("Initializing predictor...")
    predictor = QuantumAwarePredictor(
        sequence_length=20,
        hidden_dim=64,
        learning_rate=0.001
    )

    # Train with progress tracking
    epochs = 50
    current_status['total_epochs'] = epochs
    X, y = predictor.prepare_sequences(training_data)

    print(f"Training on {len(X)} samples for {epochs} epochs...")

    for epoch in range(epochs):
        current_status['current_epoch'] = epoch + 1
        current_status['training_progress'] = int((epoch + 1) / epochs * 100)

        # Train one epoch
        metrics = predictor.train_epoch(X, y)
        loss = metrics['loss']

        # Update scheduler
        predictor.scheduler.step(loss)

        # Defense analysis
        grad_norm = predictor._calculate_grad_norm()
        defense_state = predictor.defense.analyze_training_state({
            'loss': loss,
            'accuracy': 1.0 - min(loss, 1.0),
            'grad_norm': grad_norm,
            'epoch': epoch
        })

        # Record
        predictor.meta_learner.record_training_result(
            config={'lr': predictor.optimizer.lr, 'epoch': epoch},
            performance={'loss': loss}
        )

        # Save history
        predictor.training_history.append({
            'epoch': epoch,
            'loss': float(loss),
            'lr': float(predictor.optimizer.lr),
            'awareness_stage': int(defense_state['stage'].value),
            'risk_level': defense_state['risk_level']
        })

        # Update status
        current_status['training_history'] = predictor.training_history

        # Save best
        if loss < predictor.best_loss:
            predictor.best_loss = loss

    # Training complete
    current_status['training_complete'] = True
    is_training = False

    # Make sample predictions
    print("Making predictions...")
    test_sequences = [
        training_data[-40:-20],
        training_data[-60:-40],
        training_data[-80:-60],
    ]

    predictions = []
    for seq in test_sequences:
        result = predictor.predict_with_confidence(seq)
        predictions.append({
            'prediction': float(result['prediction']),
            'confidence': float(result['confidence'] * 100),
            'quantum_uncertainty': float(result['quantum_uncertainty']),
            'awareness_stage': int(result['awareness_stage_value']),
            'awareness_name': result['awareness_stage'].name,
            'risk_level': result['risk_level'],
            'should_trust': result['should_trust'],
            'description': result['defense_description']
        })

    current_status['predictions'] = predictions

    # Get meta insights
    insights = predictor.get_meta_insights()
    current_status['meta_insights'] = {
        'total_experiments': insights['total_experiments'],
        'best_lr': float(insights['best_lr']),
        'insights': insights['insights']
    }

    print("Dashboard ready!")


@app.route('/')
def index():
    """Serve the dashboard"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Pattern Predictor Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }

        .card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .card h2 {
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            padding-bottom: 10px;
        }

        .metric {
            margin: 15px 0;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }

        .metric-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 2em;
            font-weight: bold;
        }

        .progress-bar {
            width: 100%;
            height: 30px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }

        .prediction-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border-left: 4px solid #4facfe;
        }

        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin: 5px 0;
        }

        .status-nominal { background: #4ade80; color: #000; }
        .status-elevated { background: #fbbf24; color: #000; }
        .status-high { background: #f97316; color: #fff; }
        .status-critical { background: #ef4444; color: #fff; }

        .loading {
            text-align: center;
            padding: 40px;
            font-size: 1.2em;
        }

        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 4px solid white;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .chart {
            width: 100%;
            height: 200px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            position: relative;
        }

        .chart-bar {
            display: inline-block;
            width: 3%;
            background: linear-gradient(180deg, #4facfe 0%, #00f2fe 100%);
            margin: 0 1px;
            border-radius: 2px;
            vertical-align: bottom;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔮 Quantum Pattern Predictor Dashboard</h1>
        <p>Real-Time Self-Aware AI Monitoring</p>
    </div>

    <div id="loading" class="loading">
        <div class="spinner"></div>
        <p>Training predictor... This may take a minute.</p>
    </div>

    <div id="dashboard" class="dashboard" style="display: none;">
        <!-- Training Progress -->
        <div class="card">
            <h2>📊 Training Progress</h2>
            <div class="metric">
                <div class="metric-label">Status</div>
                <div class="metric-value" id="training-status">Training...</div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill" style="width: 0%">0%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Epoch</div>
                <div class="metric-value" id="current-epoch">0 / 0</div>
            </div>
            <div class="metric">
                <div class="metric-label">Best Loss</div>
                <div class="metric-value" id="best-loss">-</div>
            </div>
        </div>

        <!-- Predictions -->
        <div class="card">
            <h2>🎯 Predictions with Quantum Confidence</h2>
            <div id="predictions-container"></div>
        </div>

        <!-- Awareness Stage -->
        <div class="card">
            <h2>🛡️ 10-Stage Awareness Monitor</h2>
            <div class="metric">
                <div class="metric-label">Current Stage</div>
                <div class="metric-value" id="awareness-stage">-</div>
            </div>
            <div class="metric">
                <div class="metric-label">Risk Level</div>
                <div id="risk-badge"></div>
            </div>
        </div>

        <!-- Meta-Learning Insights -->
        <div class="card">
            <h2>🧠 Meta-Learning Insights</h2>
            <div class="metric">
                <div class="metric-label">Experiments Tracked</div>
                <div class="metric-value" id="experiments">-</div>
            </div>
            <div class="metric">
                <div class="metric-label">Best Learning Rate</div>
                <div class="metric-value" id="best-lr">-</div>
            </div>
        </div>

        <!-- Training History Chart -->
        <div class="card" style="grid-column: span 2;">
            <h2>📈 Training Loss History</h2>
            <div class="chart" id="loss-chart"></div>
        </div>
    </div>

    <script>
        let updateInterval;

        function updateDashboard() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    // Update training progress
                    document.getElementById('progress-fill').style.width = data.training_progress + '%';
                    document.getElementById('progress-fill').textContent = data.training_progress + '%';
                    document.getElementById('current-epoch').textContent =
                        data.current_epoch + ' / ' + data.total_epochs;

                    if (data.training_complete) {
                        document.getElementById('training-status').textContent = '✓ Complete';
                        document.getElementById('loading').style.display = 'none';
                        document.getElementById('dashboard').style.display = 'grid';

                        // Show best loss
                        if (data.training_history.length > 0) {
                            const losses = data.training_history.map(h => h.loss);
                            const bestLoss = Math.min(...losses);
                            document.getElementById('best-loss').textContent = bestLoss.toFixed(4);

                            // Draw loss chart
                            drawLossChart(losses);
                        }

                        // Show predictions
                        if (data.predictions.length > 0) {
                            showPredictions(data.predictions);
                        }

                        // Show meta insights
                        if (data.meta_insights.total_experiments) {
                            document.getElementById('experiments').textContent =
                                data.meta_insights.total_experiments;
                            document.getElementById('best-lr').textContent =
                                data.meta_insights.best_lr.toExponential(2);
                        }
                    }
                })
                .catch(error => console.error('Error:', error));
        }

        function showPredictions(predictions) {
            const container = document.getElementById('predictions-container');
            container.innerHTML = '';

            predictions.forEach((pred, i) => {
                const card = document.createElement('div');
                card.className = 'prediction-card';
                card.innerHTML = `
                    <strong>Prediction #${i + 1}</strong><br>
                    <strong>Value:</strong> ${pred.prediction.toFixed(4)}<br>
                    <strong>Confidence:</strong> ${pred.confidence.toFixed(1)}%<br>
                    <strong>Quantum Uncertainty:</strong> ${pred.quantum_uncertainty.toFixed(4)}<br>
                    <strong>Stage:</strong> ${pred.awareness_stage} - ${pred.awareness_name}<br>
                    <span class="status-badge status-${pred.risk_level}">${pred.risk_level.toUpperCase()}</span>
                    ${pred.should_trust ? '✓ TRUST' : '⚠️ CAUTION'}
                `;
                container.appendChild(card);
            });

            // Update awareness from last prediction
            const lastPred = predictions[predictions.length - 1];
            document.getElementById('awareness-stage').textContent =
                `Stage ${lastPred.awareness_stage}`;

            const riskBadge = document.getElementById('risk-badge');
            riskBadge.className = `status-badge status-${lastPred.risk_level}`;
            riskBadge.textContent = lastPred.risk_level.toUpperCase();
        }

        function drawLossChart(losses) {
            const chart = document.getElementById('loss-chart');
            chart.innerHTML = '';

            const maxLoss = Math.max(...losses);
            const minLoss = Math.min(...losses);
            const range = maxLoss - minLoss;

            // Sample every few epochs if too many
            const step = Math.ceil(losses.length / 50);

            for (let i = 0; i < losses.length; i += step) {
                const loss = losses[i];
                const height = ((loss - minLoss) / range) * 180 + 10;

                const bar = document.createElement('div');
                bar.className = 'chart-bar';
                bar.style.height = height + 'px';
                chart.appendChild(bar);
            }
        }

        // Start updating
        updateInterval = setInterval(updateDashboard, 1000);
        updateDashboard();
    </script>
</body>
</html>
    '''


@app.route('/status')
def get_status():
    """Get current training status"""
    return jsonify(current_status)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'training': is_training})


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════╗
║  QUANTUM PATTERN PREDICTOR - LIVE DASHBOARD              ║
╚══════════════════════════════════════════════════════════╝

Starting dashboard server...

1. Training will begin automatically
2. Open your browser to: http://localhost:8080
3. Watch real-time training progress
4. See quantum confidence predictions
5. Monitor 10-stage awareness system

Dashboard will be ready in ~30 seconds...
""")

    # Start training in background
    training_thread = threading.Thread(target=train_in_background, daemon=True)
    training_thread.start()

    # Start Flask server
    app.run(host='0.0.0.0', port=8080, debug=False)
