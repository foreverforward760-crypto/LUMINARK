"""
User Interface Modules
Optional interfaces for LUMINARK - Streamlit, Voice I/O, etc.
"""
try:
    from .streamlit_dashboard import StreamlitDashboard
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from .voice_io import VoiceInterface
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

__all__ = ['StreamlitDashboard', 'VoiceInterface', 'STREAMLIT_AVAILABLE', 'VOICE_AVAILABLE']
