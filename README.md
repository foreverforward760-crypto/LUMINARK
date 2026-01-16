# LUMINARK

🌟 **LUMINARK** - AI/ML Real-time Monitoring and Visualization System

A lightweight demo system for visualizing AI/ML metrics in real-time through both CLI and web-based dashboards.

## Features

- 📊 Real-time metrics visualization
- 🖥️ Command-line demo mode with matplotlib charts
- 🌐 Interactive web dashboard with live updates
- 📈 Multiple metric tracking (accuracy, loss, throughput, memory, CPU)
- 🎨 Beautiful gradient UI with responsive design
- 📡 RESTful API for metrics access

## Quick Start

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Option 1: Command-Line Demo

Run a basic 30-second demo that displays real-time metrics in your terminal and generates visualization plots:

```bash
python octo_demo.py --mode basic --duration 30
```

**Options:**
- `--mode`: Choose between `basic` or `advanced` demo modes (default: `basic`)
- `--duration`: Duration of the demo in seconds (default: `30`)

**Example outputs:**
- Terminal displays real-time metrics every second
- Generates `demo_results.png` with accuracy and loss charts
- Shows summary statistics at completion

### Option 2: Web Dashboard

Start the interactive web dashboard server:

```bash
python octo_dashboard_server.py
```

Then open your browser and navigate to:
- **http://localhost:8000**
- **http://127.0.0.1:8000**

**Dashboard features:**
- Live metric updates every 2 seconds
- Interactive charts showing metric trends
- System status and uptime tracking
- Manual refresh and reset controls

## API Endpoints

The dashboard server exposes several REST API endpoints:

- `GET /` - Main dashboard interface
- `GET /api/metrics` - Current metrics snapshot
- `GET /api/history` - Historical metrics data
- `GET /api/status` - System status and uptime
- `GET /api/reset` - Reset all metrics

### Example API Usage

```bash
# Get current metrics
curl http://localhost:8000/api/metrics

# Get system status
curl http://localhost:8000/api/status

# Reset metrics
curl http://localhost:8000/api/reset
```

## Project Structure

```
LUMINARK/
├── octo_demo.py              # CLI demo script
├── octo_dashboard_server.py  # Web dashboard server
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Dashboard HTML template
├── static/                  # Static assets (auto-created)
└── README.md               # This file
```

## Metrics Explained

| Metric | Description | Range |
|--------|-------------|-------|
| **Accuracy** | Model prediction accuracy | 85-99% |
| **Loss** | Training loss value | 0.01-0.15 |
| **Throughput** | Operations per second | 100-500 ops/s |
| **Memory Usage** | RAM consumption | 1024-4096 MB |
| **CPU Usage** | Processor utilization | 20-80% |

## Development

### Requirements
- Python 3.7+
- Modern web browser (for dashboard)
- Terminal with UTF-8 support (for CLI demo)

### Dependencies
- `numpy` - Numerical computations
- `matplotlib` - CLI chart generation
- `flask` - Web server framework
- `flask-cors` - CORS support
- `pillow` - Image processing

## Troubleshooting

**Port already in use:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

**matplotlib display issues:**
The CLI demo saves charts to `demo_results.png` even if display fails in headless environments.

**Dependencies installation fails:**
```bash
# Upgrade pip first
pip install --upgrade pip
pip install -r requirements.txt
```

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

Built with ❤️ for the AI/ML community
