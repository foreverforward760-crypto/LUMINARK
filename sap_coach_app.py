"""
SAP COACH - Web Interface
Simple Streamlit app for the coaching bot

To run:
    streamlit run sap_coach_app.py

Revenue Model: $29/month subscription
Target: Life coaches, therapists, consultants
"""

import streamlit as st
import os
from datetime import datetime
from sap_coach_mvp import SAPCoach
import json

# Page config
st.set_page_config(
    page_title="SAP Coach - AI Coaching Assistant",
    page_icon="🌟",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stage-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .coach-message {
        background-color: #f3e5f5;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'coach' not in st.session_state:
    st.session_state.coach = SAPCoach()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = bool(os.getenv("OPENAI_API_KEY"))

# Header
st.markdown('<h1 class="main-header">🌟 SAP Coach</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Developmental Coaching</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key input
    if not st.session_state.api_key_set:
        st.warning("⚠️ OpenAI API key not set")
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        if st.button("Save API Key"):
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.session_state.api_key_set = True
                st.success("✅ API key saved!")
                st.rerun()
    else:
        st.success("✅ API key configured")
    
    st.markdown("---")
    
    # User context
    st.subheader("👤 Your Profile")
    user_name = st.text_input("Name (optional):", key="user_name")
    user_goal = st.text_area("Current goal:", key="user_goal", height=100)
    
    st.markdown("---")
    
    # Conversation stats
    if st.session_state.messages:
        summary = st.session_state.coach.get_conversation_summary()
        st.subheader("📊 Session Stats")
        st.metric("Messages", summary['total_messages'])
        if summary.get('current_stage') is not None:
            st.metric("Current Stage", summary['current_stage'])
    
    st.markdown("---")
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.coach = SAPCoach()
        st.rerun()
    
    st.markdown("---")
    
    # Pricing info
    st.info("""
    **💰 Pricing**
    
    - Free Trial: 10 messages
    - Pro: $29/month
    - Team: $99/month
    
    [Subscribe Now](#)
    """)

# Main chat interface
st.markdown("### 💬 Conversation")

# Display chat history
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        stage_info = msg.get('stage_info', {})
        st.markdown(f'<div class="chat-message coach-message">', unsafe_allow_html=True)
        if stage_info:
            st.markdown(f'<span class="stage-badge" style="background-color: #667eea; color: white;">Stage {stage_info.get("stage", "?")} - {stage_info.get("stage_name", "")}</span>', unsafe_allow_html=True)
        st.markdown(f'<p><strong>Coach:</strong> {msg["content"]}</p>', unsafe_allow_html=True)
        if stage_info.get('insights'):
            st.markdown(f'<p style="font-style: italic; color: #666;">💡 {stage_info["insights"]}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if st.session_state.api_key_set:
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        
        # Build user context
        user_context = {}
        if user_name:
            user_context['name'] = user_name
        if user_goal:
            user_context['goal'] = user_goal
        
        # Get coach response
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.coach.chat(user_input, user_context)
        
        # Add coach response
        st.session_state.messages.append({
            'role': 'coach',
            'content': result['response'],
            'stage_info': {
                'stage': result['stage'],
                'stage_name': result['stage_name'],
                'stage_desc': result['stage_desc'],
                'insights': result['insights']
            }
        })
        
        st.rerun()
else:
    st.warning("⚠️ Please set your OpenAI API key in the sidebar to start chatting")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>SAP Coach uses AI to provide developmental coaching based on Stanfield's Axiom of Perpetuity</p>
    <p>© 2026 SAP Coach | <a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a></p>
</div>
""", unsafe_allow_html=True)
