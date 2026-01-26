"""
SAP Assessment - Email Capture Backend
Simple Flask API to capture emails from the assessment tool

Setup:
1. pip install flask flask-cors
2. python email_capture_api.py
3. Update sap_free_assessment.html to POST to this endpoint
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import os

app = Flask(__name__)
CORS(app)  # Allow requests from your HTML file

# Simple file-based storage (upgrade to database later)
EMAILS_FILE = 'captured_emails.json'

def load_emails():
    if os.path.exists(EMAILS_FILE):
        with open(EMAILS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_email(email_data):
    emails = load_emails()
    emails.append(email_data)
    with open(EMAILS_FILE, 'w') as f:
        json.dump(emails, f, indent=2)

@app.route('/api/capture-email', methods=['POST'])
def capture_email():
    """Capture email from assessment tool"""
    try:
        data = request.json
        email = data.get('email')
        stage = data.get('stage')
        scores = data.get('scores', {})
        
        if not email or '@' not in email:
            return jsonify({'success': False, 'error': 'Invalid email'}), 400
        
        # Save to file
        email_data = {
            'email': email,
            'stage': stage,
            'scores': scores,
            'timestamp': datetime.now().isoformat(),
            'source': 'free_assessment'
        }
        
        save_email(email_data)
        
        # In production, also send to Mailchimp/ConvertKit
        # mailchimp_api.add_subscriber(email, {'STAGE': stage})
        
        return jsonify({
            'success': True,
            'message': 'Email captured successfully!'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get email capture statistics"""
    emails = load_emails()
    
    # Calculate stats
    total = len(emails)
    stages = {}
    for entry in emails:
        stage = entry.get('stage', 'unknown')
        stages[stage] = stages.get(stage, 0) + 1
    
    return jsonify({
        'total_emails': total,
        'stage_distribution': stages,
        'latest_capture': emails[-1] if emails else None
    })

@app.route('/')
def home():
    """Simple dashboard"""
    emails = load_emails()
    stats = {
        'total': len(emails),
        'today': len([e for e in emails if e['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))])
    }
    
    return f"""
    <html>
    <head>
        <title>SAP Assessment - Email Dashboard</title>
        <style>
            body {{ font-family: Arial; padding: 40px; background: #f5f5f5; }}
            .stat {{ background: white; padding: 20px; margin: 10px 0; border-radius: 10px; }}
            .stat h2 {{ margin: 0; color: #667eea; }}
            .stat p {{ font-size: 2rem; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>📊 SAP Assessment Dashboard</h1>
        
        <div class="stat">
            <h2>Total Emails Captured</h2>
            <p>{stats['total']}</p>
        </div>
        
        <div class="stat">
            <h2>Today</h2>
            <p>{stats['today']}</p>
        </div>
        
        <div class="stat">
            <h2>Latest Emails</h2>
            <ul>
                {''.join([f"<li>{e['email']} - Stage {e['stage']} ({e['timestamp']})</li>" for e in emails[-10:]])}
            </ul>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("="*70)
    print("📧 SAP EMAIL CAPTURE API")
    print("="*70)
    print("\nStarting server on http://localhost:5000")
    print("\nEndpoints:")
    print("  POST /api/capture-email - Capture email from assessment")
    print("  GET  /api/stats - Get statistics")
    print("  GET  / - View dashboard")
    print("\n" + "="*70)
    
    app.run(debug=True, port=5000)
