"""
LUMINARK Deepchat API Endpoint
Handles chat requests with framework-integrated responses
"""

import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs

class handler(BaseHTTPRequestHandler):
    """Vercel serverless handler for Deepchat"""
    
    def do_POST(self):
        """Handle POST requests from Deepchat"""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body.decode('utf-8'))
            user_message = data.get('messages', [{}])[-1].get('text', '').strip()
            
            # Framework-integrated response
            response_text = self.generate_luminark_response(user_message)
            
            # Send Deepchat-formatted response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                'text': response_text
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                'text': f"Error processing request: {str(e)}"
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def generate_luminark_response(self, message):
        """Generate LUMINARK framework-integrated responses"""
        
        # Framework knowledge base
        framework_responses = {
            'middle path': 'The middle path in the LUMINARK framework means finding balance between extremes—not suppressing, not exploding. Integration happens between polarities, not in fusion or division.',
            'as above so below': 'As above, so below: Your internal state mirrors your external reality. When you change internally (awareness, acceptance, integration), your external circumstances shift. Clean your inner space; your outer space follows.',
            "ma'at": "Ma'at balance is ethical equilibrium. Ask: Am I honoring both myself AND others? Am I authentic AND contained? The middle path manifests as balanced relationships, grounded authenticity.",
            'stage 5': 'THRESHOLD (The Pressure): The stable container (Stage 4) holding unstable content. This is the shortest, most volatile stage. The pressure test: Are you ready? If no—strengthen. If yes—cross. If unsure—pause.',
            'stage 8': "RIGIDITY (The Crystallization): The permanence trap. You achieved understanding and identified with it. That identity becomes the cage. Escape: Radical vulnerability. 'I was right AND incomplete. I'm updating always.'",
            'stages': 'LUMINARK maps 10 consciousness stages: 0-Plenara (void), 1-Radiance (emergence), 2-Duality (division), 3-Expression (outburst), 4-Foundation (rest), 5-Threshold (pressure), 6-Integration (harvest), 7-Illusion (analysis), 8-Rigidity (crystallization), 9-Renewal (transcendence).',
            'help': 'Ask about: middle path, as above so below, Ma\'at balance, Stage 5, Stage 8, the 10 stages, tactical guidance, framework integration, or any aspect of LUMINARK consciousness mapping.',
        }
        
        # Simple keyword matching for responses
        message_lower = message.lower()
        
        for key, response in framework_responses.items():
            if key in message_lower:
                return response
        
        # Default response with framework context
        return (
            "LUMINARK Deep Chat: I'm here to help you understand the framework principles—"
            "middle path balance, as above/so below mechanics, Ma'at ethical testing, and the 10 consciousness stages. "
            "Ask me about stage transitions, framework integration, or tactical applications. "
            "What aspect would you like to explore?"
        )
