// LUMINARK AI Framework - Dashboard JavaScript
// Real-time WebSocket communication and visualization

// Initialize WebSocket connection
const socket = io();

// State management
let currentState = {
    stage: 4,
    stage_name: 'Foundation',
    physical_state: 'stable',
    conscious_state: 'unstable',
    inversion_level: 0,
    resonance: 0.5
};

// Connect to WebSocket
socket.on('connect', () => {
    console.log('✅ Connected to LUMINARK server');
    addLogEntry('Connected to server');
    socket.emit('request_update');
});

socket.on('disconnect', () => {
    console.log('❌ Disconnected from server');
    addLogEntry('Disconnected from server');
});

// Listen for state updates
socket.on('state_update', (data) => {
    console.log('📊 State update received:', data);
    currentState = data;
    updateDashboard(data);
    addLogEntry(`Stage transition: ${data.stage_name} (Level ${data.stage})`);
});

// Update dashboard with new data
function updateDashboard(data) {
    // Update stats
    document.getElementById('current-stage').textContent = data.stage;
    document.getElementById('stage-name').textContent = data.stage_name;
    document.getElementById('inversion-level').textContent = data.inversion_level;
    document.getElementById('resonance').textContent = Math.round(data.resonance * 100) + '%';

    // Update state indicators
    updateStateIndicator('physical-state', data.physical_state);
    updateStateIndicator('conscious-state', data.conscious_state);

    // Update timestamp
    const now = new Date();
    document.getElementById('last-update').textContent = now.toLocaleTimeString();

    // Update visualizations
    updateStagesChart();
    updateInversionChart();
}

// Update state indicator
function updateStateIndicator(elementId, state) {
    const element = document.getElementById(elementId);
    const badge = element.querySelector('.state-badge');

    if (state === 'stable') {
        badge.className = 'state-badge stable';
        badge.textContent = 'Stable';
    } else {
        badge.className = 'state-badge unstable';
        badge.textContent = 'Unstable';
    }
}

// Add log entry
function addLogEntry(message) {
    const logContent = document.getElementById('activity-log');
    const entry = document.createElement('div');
    entry.className = 'log-entry';

    const time = document.createElement('span');
    time.className = 'log-time';
    time.textContent = new Date().toLocaleTimeString();

    const msg = document.createElement('span');
    msg.className = 'log-message';
    msg.textContent = message;

    entry.appendChild(time);
    entry.appendChild(msg);

    logContent.insertBefore(entry, logContent.firstChild);

    // Keep only last 50 entries
    while (logContent.children.length > 50) {
        logContent.removeChild(logContent.lastChild);
    }
}

// Update SAR Stages Chart
function updateStagesChart() {
    fetch('/api/stages')
        .then(response => response.json())
        .then(data => {
            const stages = data.stages || [];

            const trace = {
                x: stages.map(s => s.level),
                y: stages.map(s => s.energy_signature),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Energy Signature',
                line: {
                    color: '#00c9ff',
                    width: 3
                },
                marker: {
                    size: 12,
                    color: stages.map(s => s.is_inverted ? '#ff6b6b' : '#92fe9d'),
                    line: {
                        color: '#ffffff',
                        width: 2
                    }
                },
                text: stages.map(s => `${s.name}<br>Physical: ${s.physical_state}<br>Conscious: ${s.conscious_state}`),
                hoverinfo: 'text+x+y'
            };

            const layout = {
                title: 'SAP Framework Stages (Stanfield\'s Axiom of Perpetuity)',
                xaxis: {
                    title: 'Stage Level',
                    gridcolor: 'rgba(255, 255, 255, 0.1)',
                    color: '#ffffff'
                },
                yaxis: {
                    title: 'Energy Signature',
                    gridcolor: 'rgba(255, 255, 255, 0.1)',
                    color: '#ffffff'
                },
                paper_bgcolor: 'rgba(0, 0, 0, 0)',
                plot_bgcolor: 'rgba(0, 0, 0, 0.3)',
                font: {
                    color: '#ffffff'
                },
                shapes: [{
                    type: 'line',
                    x0: currentState.stage,
                    y0: 0,
                    x1: currentState.stage,
                    y1: 1,
                    line: {
                        color: '#92fe9d',
                        width: 2,
                        dash: 'dash'
                    }
                }]
            };

            Plotly.newPlot('stages-chart', [trace], layout, { responsive: true });
        });
}

// Update Inversion Chart
function updateInversionChart() {
    const stages = [
        { name: 'Plenara', level: 0, inverted: false },
        { name: 'Spark', level: 1, inverted: true },
        { name: 'Polarity', level: 2, inverted: true },
        { name: 'Motion', level: 3, inverted: true },
        { name: 'Foundation', level: 4, inverted: true },
        { name: 'Threshold', level: 5, inverted: true },
        { name: 'Integration', level: 6, inverted: true },
        { name: 'Illusion', level: 7, inverted: true },
        { name: 'Rigidity', level: 8, inverted: true },
        { name: 'Renewal', level: 9, inverted: false }
    ];

    const trace1 = {
        x: stages.map(s => s.level),
        y: stages.map(s => s.level % 2 === 0 ? 1 : 0),
        type: 'bar',
        name: 'Physical Stable',
        marker: {
            color: '#00c9ff'
        }
    };

    const trace2 = {
        x: stages.map(s => s.level),
        y: stages.map(s => s.level % 2 === 0 ? 0 : 1),
        type: 'bar',
        name: 'Conscious Stable',
        marker: {
            color: '#92fe9d'
        }
    };

    const layout = {
        title: 'Inversion Principle Visualization',
        barmode: 'group',
        xaxis: {
            title: 'Stage',
            ticktext: stages.map(s => s.name),
            tickvals: stages.map(s => s.level),
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            color: '#ffffff'
        },
        yaxis: {
            title: 'Stability',
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            color: '#ffffff'
        },
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        plot_bgcolor: 'rgba(0, 0, 0, 0.3)',
        font: {
            color: '#ffffff'
        }
    };

    Plotly.newPlot('inversion-chart', [trace1, trace2], layout, { responsive: true });
}

// Control functions
function detectInversion() {
    const physical = currentState.physical_state === 'stable';
    const conscious = currentState.conscious_state === 'stable';

    fetch('/api/detect_inversion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            physical_stable: physical,
            conscious_stable: conscious
        })
    })
        .then(response => response.json())
        .then(data => {
            console.log('Inversion detection result:', data);
            addLogEntry(`Inversion detected: ${data.description}`);
        })
        .catch(error => {
            console.error('Error detecting inversion:', error);
            addLogEntry('Error detecting inversion');
        });
}

function transitionStage() {
    const resonance = Math.random(); // Random resonance for demo

    fetch('/api/transition', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            resonance: resonance
        })
    })
        .then(response => response.json())
        .then(data => {
            console.log('Stage transition result:', data);
            addLogEntry(`Transitioned to stage ${data.stage}: ${data.stage_name}`);
        })
        .catch(error => {
            console.error('Error transitioning stage:', error);
            addLogEntry('Error transitioning stage');
        });
}

function refreshData() {
    socket.emit('request_update');
    addLogEntry('Data refresh requested');
}

// Initialize dashboard on load
document.addEventListener('DOMContentLoaded', () => {
    console.log('🌟 LUMINARK Dashboard initialized');
    updateStagesChart();
    updateInversionChart();

    // Request initial data
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            updateDashboard(data);
        });

    // Auto-refresh every 5 seconds
    setInterval(() => {
        socket.emit('request_update');
    }, 5000);
});
