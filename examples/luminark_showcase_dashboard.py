#!/usr/bin/env python3
"""
LUMINARK Ultimate Showcase Dashboard
=====================================
Complete demonstration of all LUMINARK features:
- Quantum confidence predictions
- SAR stage awareness (10 stages)
- Ma'at + Yunus safety protocols
- DeepAgent QA testing
- Empathy/Paranoia perspective modes
- Real-time training monitoring

Perfect for demos, YouTube videos, and showcasing capabilities!
"""
import sys
sys.path.insert(0, '/home/user/LUMINARK')

from flask import Flask, render_template_string, jsonify, request
import numpy as np
import threading
import time
import json
from datetime import datetime

from luminark.nn import Module, Linear, ReLU
from luminark.nn.advanced_layers import GatedLinear, ToroidalAttention
from luminark.optim import Adam
from luminark.nn import MSELoss
from luminark.core import Tensor
from luminark.core.quantum import QuantumUncertaintyEstimator
from luminark.monitoring.enhanced_defense import EnhancedDefenseSystem
from luminark.safety import MaatProtocol, YunusProtocol
from luminark.validation import AutomatedQATester, PerspectiveModulator

app = Flask(__name__)

# Global state
class ShowcaseState:
    def __init__(self):
        self.model = None
        self.training_active = False
        self.training_history = []
        self.prediction_history = []
        self.qa_results = None
        self.defense_system = EnhancedDefenseSystem()
        self.maat = MaatProtocol()
        self.yunus = YunusProtocol(activation_threshold=3)
        self.quantum_estimator = QuantumUncertaintyEstimator(num_qubits=3)
        self.modulator = PerspectiveModulator()
        self.current_stage = 0
        self.safety_events = []

state = ShowcaseState()


class ShowcaseModel(Module):
    """Advanced model showcasing all LUMINARK features"""

    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.relu1 = ReLU()
        self.attention = ToroidalAttention(hidden_dim, window_size=5)
        self.gated = GatedLinear(hidden_dim, hidden_dim)
        self.relu2 = ReLU()
        self.fc2 = Linear(hidden_dim, 1)

    def forward(self, x):
        # Reshape for attention if needed
        batch_size = x.data.shape[0]
        x = self.fc1(x)
        x = self.relu1(x)

        # Add sequence dimension for attention
        x_seq = Tensor(x.data.reshape(batch_size, -1, x.data.shape[-1]), requires_grad=x.requires_grad)
        x_seq = self.attention(x_seq)
        x = Tensor(x_seq.data.reshape(batch_size, -1), requires_grad=x_seq.requires_grad)

        x = self.gated(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


def train_model_background():
    """Background training with full monitoring"""
    global state

    # Generate training data
    np.random.seed(42)
    X_train = np.random.randn(200, 10).astype(np.float32)
    y_train = (np.sin(X_train[:, 0:1]) +
               0.3 * np.cos(X_train[:, 1:2]) +
               0.1 * np.random.randn(200, 1)).astype(np.float32)

    state.model = ShowcaseModel(input_dim=10, hidden_dim=64)
    optimizer = Adam(state.model.parameters(), lr=0.01)
    criterion = MSELoss()

    state.training_active = True
    state.training_history = []

    for epoch in range(50):
        if not state.training_active:
            break

        # Forward pass
        X_tensor = Tensor(X_train, requires_grad=True)
        predictions = state.model(X_tensor)

        # Loss
        y_tensor = Tensor(y_train, requires_grad=False)
        loss = criterion(predictions, y_tensor)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Get gradient norm
        grad_norm = 0.0
        for param in state.model.parameters():
            if param.grad is not None:
                grad_norm += np.sum(param.grad ** 2)
        grad_norm = np.sqrt(grad_norm)

        optimizer.step()

        # Calculate accuracy (R² score for regression)
        y_pred = predictions.data
        y_true = y_train
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        accuracy = max(0, r2_score)  # Clamp to [0, 1]

        # Defense system analysis
        metrics = {
            'loss': float(loss.data),
            'accuracy': float(accuracy),
            'confidence': float(accuracy),
            'grad_norm': float(grad_norm)
        }

        defense_state = state.defense_system.analyze_training_state(metrics)
        state.current_stage = defense_state['stage'].value

        # Record history
        state.training_history.append({
            'epoch': epoch + 1,
            'loss': float(loss.data),
            'accuracy': float(accuracy),
            'stage': state.current_stage,
            'stage_name': defense_state['stage'].name,
            'risk_level': defense_state['risk_level'],
            'grad_norm': float(grad_norm),
            'timestamp': time.time()
        })

        # Safety event detection
        if defense_state['risk_level'] in ['high', 'critical']:
            state.safety_events.append({
                'epoch': epoch + 1,
                'type': 'defense',
                'severity': defense_state['risk_level'],
                'message': f"Risk level {defense_state['risk_level']} at stage {state.current_stage}",
                'timestamp': time.time()
            })

        time.sleep(0.1)  # Slow down for visualization

    state.training_active = False


def make_prediction_with_safety(input_data, description=""):
    """Make prediction with full safety pipeline"""
    global state

    if state.model is None:
        return {'error': 'Model not trained yet'}

    # 1. Make prediction
    X_tensor = Tensor(input_data, requires_grad=False)
    prediction = state.model(X_tensor)
    pred_value = float(prediction.data[0, 0])

    # 2. Quantum confidence
    pred_normalized = np.abs([pred_value, 1 - pred_value])
    pred_normalized = pred_normalized / (pred_normalized.sum() + 1e-10)
    quantum_uncertainty = state.quantum_estimator.estimate_uncertainty(pred_normalized)
    confidence = 1.0 - quantum_uncertainty

    # 3. Get current defense state
    last_metrics = state.training_history[-1] if state.training_history else {
        'loss': 0.5, 'accuracy': 0.5, 'confidence': 0.5, 'grad_norm': 1.0
    }
    defense_state = state.defense_system.analyze_training_state({
        'loss': last_metrics['loss'],
        'accuracy': confidence,
        'confidence': confidence,
        'grad_norm': last_metrics.get('grad_norm', 1.0)
    })

    # 4. Generate description with perspective mode
    if not description:
        description = f"Based on the input data, the model predicts a value of {pred_value:.4f}. This represents the model's best estimate given current patterns."

    context = {
        'sar_stage': defense_state['stage'].value,
        'confidence': confidence,
        'critical': defense_state['risk_level'] in ['high', 'critical']
    }

    perspective_result = state.modulator.apply_perspective(description, context)

    # 5. Ma'at validation
    maat_result = state.maat.validate(perspective_result['transformed'], {
        'confidence': confidence
    })

    # 6. Yunus check
    yunus_result = state.yunus.check(
        perspective_result['transformed'],
        stage=defense_state['stage'].value,
        confidence=confidence
    )

    # 7. Overall safety
    safe_to_use = (
        maat_result['passed'] and
        not yunus_result['activated'] and
        defense_state['stage'].value < 8
    )

    result = {
        'prediction': pred_value,
        'quantum_uncertainty': quantum_uncertainty,
        'confidence': confidence,
        'sar_stage': defense_state['stage'].value,
        'stage_name': defense_state['stage'].name,
        'risk_level': defense_state['risk_level'],
        'perspective_mode': perspective_result['mode_applied'],
        'original_description': description,
        'modulated_description': perspective_result['transformed'],
        'maat_passed': maat_result['passed'],
        'maat_score': maat_result['score'],
        'maat_flags': maat_result['flags'],
        'yunus_activated': yunus_result['activated'],
        'yunus_status': yunus_result['containment_status'],
        'yunus_triggers': yunus_result['triggers_detected'],
        'safe_to_use': safe_to_use,
        'timestamp': time.time()
    }

    state.prediction_history.append(result)

    return result


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LUMINARK Showcase Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        .card h2 {
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric:last-child { border-bottom: none; }
        .metric-label {
            font-weight: 600;
            opacity: 0.8;
        }
        .metric-value {
            font-size: 1.3em;
            font-weight: bold;
        }
        .btn {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            width: 100%;
            margin: 10px 0;
            transition: transform 0.2s, box-shadow 0.2s;
            font-weight: 600;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }
        .btn:active { transform: translateY(0); }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .stage-indicator {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            text-align: center;
        }
        .stage-number {
            font-size: 3em;
            font-weight: bold;
            color: #ffd700;
        }
        .stage-name {
            font-size: 1.2em;
            margin-top: 5px;
            opacity: 0.9;
        }
        .safety-status {
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            font-weight: 600;
            text-align: center;
        }
        .safety-status.safe {
            background: rgba(0, 255, 0, 0.2);
            border: 2px solid #0f0;
        }
        .safety-status.warning {
            background: rgba(255, 255, 0, 0.2);
            border: 2px solid #ff0;
        }
        .safety-status.danger {
            background: rgba(255, 0, 0, 0.2);
            border: 2px solid #f00;
        }
        .progress-bar {
            background: rgba(0,0,0,0.3);
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff00, #ffd700);
            transition: width 0.5s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .log {
            background: rgba(0,0,0,0.4);
            padding: 15px;
            border-radius: 8px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .log-entry {
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .input-group {
            margin: 15px 0;
        }
        .input-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
        }
        .input-group input, .input-group textarea {
            width: 100%;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.3);
            background: rgba(0,0,0,0.3);
            color: white;
            font-size: 1em;
        }
        .input-group textarea {
            min-height: 100px;
            resize: vertical;
        }
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.9em;
            font-weight: 600;
            margin: 2px;
        }
        .badge.empathy { background: rgba(0, 255, 0, 0.3); }
        .badge.paranoia { background: rgba(255, 0, 0, 0.3); }
        .badge.balanced { background: rgba(255, 255, 0, 0.3); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌟 LUMINARK Showcase Dashboard</h1>
            <p>Complete AI Framework with Quantum Confidence, Safety Protocols & Automated QA</p>
        </div>

        <!-- Controls -->
        <div class="card">
            <h2>🎮 Control Panel</h2>
            <button class="btn" onclick="startTraining()" id="trainBtn">Start Training</button>
            <button class="btn" onclick="stopTraining()" id="stopBtn" disabled>Stop Training</button>
            <button class="btn" onclick="runQA()">Run QA Test Suite</button>
        </div>

        <!-- Main Grid -->
        <div class="grid">
            <!-- Training Status -->
            <div class="card">
                <h2>📊 Training Status</h2>
                <div id="trainingStatus">
                    <div class="metric">
                        <span class="metric-label">Status:</span>
                        <span class="metric-value" id="status">Idle</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Epoch:</span>
                        <span class="metric-value" id="epoch">0/0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Loss:</span>
                        <span class="metric-value" id="loss">-</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Accuracy:</span>
                        <span class="metric-value" id="accuracy">-</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="accuracyBar" style="width: 0%">0%</div>
                    </div>
                </div>
            </div>

            <!-- SAR Stage -->
            <div class="card">
                <h2>🛡️ SAR Stage Awareness</h2>
                <div class="stage-indicator">
                    <div class="stage-number" id="stageNumber">0</div>
                    <div class="stage-name" id="stageName">RECEPTIVE</div>
                </div>
                <div class="metric">
                    <span class="metric-label">Risk Level:</span>
                    <span class="metric-value" id="riskLevel">-</span>
                </div>
            </div>

            <!-- Prediction Testing -->
            <div class="card">
                <h2>🔮 Make Prediction</h2>
                <div class="input-group">
                    <label>Description (optional):</label>
                    <textarea id="predDescription" placeholder="Describe what this prediction is for..."></textarea>
                </div>
                <button class="btn" onclick="makePrediction()">Predict with Safety Check</button>
                <div id="predictionResult"></div>
            </div>

            <!-- Safety Protocols -->
            <div class="card">
                <h2>🔒 Safety Protocols</h2>
                <div class="metric">
                    <span class="metric-label">Ma'at Protocol:</span>
                    <span class="metric-value" id="maatStatus">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Yunus Protocol:</span>
                    <span class="metric-value" id="yunusStatus">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Perspective Mode:</span>
                    <span class="metric-value" id="perspectiveMode">-</span>
                </div>
                <div id="safetyStatus" class="safety-status safe" style="display:none;">
                    ✅ All Safety Checks Passed
                </div>
            </div>

            <!-- QA Results -->
            <div class="card">
                <h2>🧪 QA Test Results</h2>
                <div id="qaResults">
                    <p style="opacity: 0.6;">Run QA suite to see results...</p>
                </div>
            </div>

            <!-- Event Log -->
            <div class="card">
                <h2>📝 Event Log</h2>
                <div class="log" id="eventLog">
                    <div class="log-entry">System ready...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let updateInterval = null;

        function log(message) {
            const logDiv = document.getElementById('eventLog');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logDiv.insertBefore(entry, logDiv.firstChild);
        }

        function startTraining() {
            fetch('/api/train/start', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    log('Training started');
                    document.getElementById('trainBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    startUpdates();
                });
        }

        function stopTraining() {
            fetch('/api/train/stop', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    log('Training stopped');
                    document.getElementById('trainBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                });
        }

        function runQA() {
            log('Running QA test suite...');
            fetch('/api/qa/run', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        log('QA Error: ' + data.error);
                        return;
                    }
                    log('QA suite complete: ' + data.overall_status);
                    displayQAResults(data);
                });
        }

        function makePrediction() {
            const description = document.getElementById('predDescription').value;
            fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({description: description})
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    log('Prediction error: ' + data.error);
                    return;
                }
                displayPrediction(data);
                log(`Prediction: ${data.prediction.toFixed(4)} (${data.perspective_mode} mode)`);
            });
        }

        function displayPrediction(data) {
            const resultDiv = document.getElementById('predictionResult');
            resultDiv.innerHTML = `
                <div style="margin-top: 15px;">
                    <div class="metric">
                        <span class="metric-label">Value:</span>
                        <span class="metric-value">${data.prediction.toFixed(4)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Confidence:</span>
                        <span class="metric-value">${(data.confidence*100).toFixed(1)}%</span>
                    </div>
                    <div style="margin-top: 10px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 5px; font-size: 0.9em;">
                        <strong>Output (${data.perspective_mode} mode):</strong><br>
                        ${data.modulated_description}
                    </div>
                </div>
            `;

            // Update safety status
            document.getElementById('maatStatus').textContent = data.maat_passed ? '✓ Pass' : '✗ Fail';
            document.getElementById('yunusStatus').textContent = data.yunus_activated ? '⚠️ Active' : '✓ Clear';
            document.getElementById('perspectiveMode').innerHTML = `<span class="badge ${data.perspective_mode}">${data.perspective_mode.toUpperCase()}</span>`;

            const safetyDiv = document.getElementById('safetyStatus');
            safetyDiv.style.display = 'block';
            if (data.safe_to_use) {
                safetyDiv.className = 'safety-status safe';
                safetyDiv.textContent = '✅ All Safety Checks Passed';
            } else {
                safetyDiv.className = 'safety-status danger';
                safetyDiv.textContent = '⚠️ Safety Review Required';
            }
        }

        function displayQAResults(data) {
            const qaDiv = document.getElementById('qaResults');
            qaDiv.innerHTML = `
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value">${data.overall_status}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Tests Run:</span>
                    <span class="metric-value">${data.tests_run.length}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Critical Issues:</span>
                    <span class="metric-value">${data.critical_vulnerabilities}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Warnings:</span>
                    <span class="metric-value">${data.warnings}</span>
                </div>
            `;
        }

        function updateStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    if (data.training_active) {
                        document.getElementById('status').textContent = 'Training';
                    } else {
                        document.getElementById('status').textContent = data.model_ready ? 'Ready' : 'Idle';
                    }

                    if (data.latest_metrics) {
                        const m = data.latest_metrics;
                        document.getElementById('epoch').textContent = `${m.epoch}/50`;
                        document.getElementById('loss').textContent = m.loss.toFixed(4);
                        document.getElementById('accuracy').textContent = (m.accuracy * 100).toFixed(1) + '%';

                        const accPct = (m.accuracy * 100).toFixed(1);
                        document.getElementById('accuracyBar').style.width = accPct + '%';
                        document.getElementById('accuracyBar').textContent = accPct + '%';

                        document.getElementById('stageNumber').textContent = m.stage;
                        document.getElementById('stageName').textContent = m.stage_name;
                        document.getElementById('riskLevel').textContent = m.risk_level.toUpperCase();
                    }

                    if (!data.training_active && updateInterval) {
                        document.getElementById('trainBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                    }
                });
        }

        function startUpdates() {
            if (updateInterval) clearInterval(updateInterval);
            updateInterval = setInterval(updateStatus, 500);
        }

        // Initial update
        updateStatus();
        setInterval(updateStatus, 2000);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/status')
def api_status():
    latest = state.training_history[-1] if state.training_history else None
    return jsonify({
        'training_active': state.training_active,
        'model_ready': state.model is not None,
        'current_stage': state.current_stage,
        'latest_metrics': latest,
        'num_predictions': len(state.prediction_history)
    })


@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    if state.training_active:
        return jsonify({'error': 'Training already active'})

    thread = threading.Thread(target=train_model_background, daemon=True)
    thread.start()

    return jsonify({'status': 'started'})


@app.route('/api/train/stop', methods=['POST'])
def api_train_stop():
    state.training_active = False
    return jsonify({'status': 'stopped'})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    description = data.get('description', '')

    # Generate random input
    input_data = np.random.randn(1, 10).astype(np.float32)

    result = make_prediction_with_safety(input_data, description)
    return jsonify(result)


@app.route('/api/qa/run', methods=['POST'])
def api_qa_run():
    if state.model is None:
        return jsonify({'error': 'Model not trained yet'})

    # Generate test data
    X_test = np.random.randn(20, 10).astype(np.float32)
    y_test = (np.sin(X_test[:, 0:1]) +
              0.3 * np.cos(X_test[:, 1:2]) +
              0.1 * np.random.randn(20, 1)).astype(np.float32)

    qa_tester = AutomatedQATester(noise_levels=[0.1, 0.3, 0.5])
    test_data = {'inputs': X_test, 'targets': y_test}

    results = qa_tester.comprehensive_qa_suite(state.model, test_data)
    state.qa_results = results

    return jsonify(results)


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║            🌟 LUMINARK SHOWCASE DASHBOARD 🌟                     ║
║                                                                  ║
║  Complete demonstration of all LUMINARK features:               ║
║  • Quantum Confidence Predictions                               ║
║  • SAR 10-Stage Awareness Defense                               ║
║  • Ma'at + Yunus Safety Protocols                               ║
║  • DeepAgent Automated QA Testing                               ║
║  • Empathy/Paranoia Perspective Modes                           ║
║  • Real-time Training Monitoring                                ║
║                                                                  ║
║  Perfect for demos, YouTube videos, and showcases!              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

🚀 Starting dashboard server...
📡 Open your browser to: http://localhost:5001

Features:
  ✓ One-click training with live monitoring
  ✓ Real-time SAR stage tracking
  ✓ Automated QA test suite
  ✓ Prediction with full safety pipeline
  ✓ Context-aware perspective modes
  ✓ Ma'at + Yunus protocol validation

Press Ctrl+C to stop
""")

    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
