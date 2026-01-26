"""
LUMINARK Dashboard - Beautiful & Simple
Interactive web UI for training, generation, and visualization
Install: pip install streamlit plotly
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional
import time

try:
    from luminark.core import LUMINARK
    from luminark.sensing import MycelialSensorySystem
    LUMINARK_AVAILABLE = True
except ImportError:
    LUMINARK_AVAILABLE = False

try:
    from luminark.interfaces.voice_io import VoiceInterface
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

try:
    from luminark.memory.faiss_memory import FAISSMemory
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ============================================================================
# BEAUTIFUL DESIGN - Custom CSS
# ============================================================================

CUSTOM_CSS = """
<style>
    /* Import beautiful fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');

    /* Dark theme with gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Beautiful headers */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Metrics styling */
    .metric-card {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 15, 30, 0.9);
        backdrop-filter: blur(10px);
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }

    /* Clean inputs */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        padding: 0.75rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        padding: 0.5rem;
        border-radius: 15px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255, 255, 255, 0.6);
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Sacred symbol pulse */
    @keyframes pulse {
        0%, 100% { opacity: 0.6; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
    }

    .sacred-symbol {
        font-size: 4rem;
        text-align: center;
        animation: pulse 3s ease-in-out infinite;
        margin: 2rem 0;
    }
</style>
"""


class StreamlitDashboard:
    """Beautiful LUMINARK Dashboard - Simple & Powerful"""

    def __init__(self):
        self.setup_page()
        self.apply_custom_css()
        self.initialize_session_state()

    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="LUMINARK Dashboard",
            page_icon="✨",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def apply_custom_css(self):
        """Apply beautiful custom CSS"""
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'mycelial' not in st.session_state:
            st.session_state.mycelial = None
        if 'training_history' not in st.session_state:
            st.session_state.training_history = {
                'loss': [],
                'accuracy': [],
                'sar_stage': [],
                'coherence': []
            }
        if 'voice_enabled' not in st.session_state:
            st.session_state.voice_enabled = False

    def render_header(self):
        """Render beautiful header"""
        st.markdown('<div class="sacred-symbol">✨</div>', unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>LUMINARK Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.6); font-size: 1.1rem;'>Quantum-Aware AI Framework with Mycelial Sensing</p>", unsafe_allow_html=True)

    def render_sidebar(self):
        """Simple sidebar controls"""
        st.sidebar.title("🎛️ Controls")

        # Model initialization
        with st.sidebar.expander("⚙️ Model Setup", expanded=True):
            vocab_size = st.number_input("Vocabulary", 100, 50000, 10000)
            hidden_dim = st.number_input("Hidden Dim", 64, 2048, 256)

            if st.button("Initialize Model"):
                self.initialize_model(vocab_size, hidden_dim)

        # Training controls
        if st.session_state.model:
            with st.sidebar.expander("🎓 Training", expanded=True):
                batch_size = st.slider("Batch Size", 1, 128, 32)
                lr = st.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")

                if st.button("Run Training Epoch"):
                    self.run_training_epoch(batch_size, lr)

        # Voice toggle
        if VOICE_AVAILABLE:
            st.sidebar.markdown("---")
            st.session_state.voice_enabled = st.sidebar.checkbox(
                "🎤 Voice I/O",
                value=st.session_state.voice_enabled
            )

    def initialize_model(self, vocab_size: int, hidden_dim: int):
        """Initialize LUMINARK model"""
        with st.spinner("Initializing..."):
            st.session_state.model = LUMINARK(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                output_dim=vocab_size
            )
            st.session_state.mycelial = MycelialSensorySystem()
            st.success("✅ Model initialized!")
            st.balloons()

    def run_training_epoch(self, batch_size: int, lr: float):
        """Simulate training epoch"""
        progress = st.progress(0)
        status = st.empty()

        for step in range(10):
            time.sleep(0.1)

            # Mock metrics
            loss = np.random.rand() * 0.5 + 0.5 * (1 - step/10)
            accuracy = np.random.rand() * 0.3 + 0.7 * (step/10)

            # Get mycelial state
            mycelial_state = st.session_state.mycelial.sense_complete({
                'loss': loss,
                'accuracy': accuracy
            })

            # Update history
            st.session_state.training_history['loss'].append(loss)
            st.session_state.training_history['accuracy'].append(accuracy)
            st.session_state.training_history['sar_stage'].append(
                mycelial_state.sap_state.current_stage
            )
            st.session_state.training_history['coherence'].append(
                mycelial_state.overall_coherence
            )

            progress.progress((step + 1) / 10)
            status.text(f"Step {step+1}/10 - Loss: {loss:.4f}")

        st.success("✅ Training complete!")

    def render_metrics(self):
        """Display key metrics in beautiful cards"""
        if not st.session_state.training_history['loss']:
            return

        history = st.session_state.training_history

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{history['loss'][-1]:.3f}</div>
                <div class="metric-label">Loss</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{history['accuracy'][-1]:.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{history['sar_stage'][-1]}/81</div>
                <div class="metric-label">SAR Stage</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{history['coherence'][-1]:.3f}</div>
                <div class="metric-label">Coherence</div>
            </div>
            """, unsafe_allow_html=True)

    def render_training_chart(self):
        """Beautiful training visualization"""
        history = st.session_state.training_history

        if not history['loss']:
            st.info("No training data yet. Run a training epoch to see charts.")
            return

        steps = list(range(len(history['loss'])))

        # Create dual-axis chart
        fig = go.Figure()

        # Loss trace
        fig.add_trace(go.Scatter(
            x=steps,
            y=history['loss'],
            name='Loss',
            line=dict(color='#f87171', width=3),
            fill='tozeroy',
            fillcolor='rgba(248, 113, 113, 0.1)'
        ))

        # Accuracy trace
        fig.add_trace(go.Scatter(
            x=steps,
            y=history['accuracy'],
            name='Accuracy',
            yaxis='y2',
            line=dict(color='#34d399', width=3),
            fill='tozeroy',
            fillcolor='rgba(52, 211, 153, 0.1)'
        ))

        # Coherence trace
        fig.add_trace(go.Scatter(
            x=steps,
            y=history['coherence'],
            name='Coherence',
            yaxis='y2',
            line=dict(color='#818cf8', width=3, dash='dot')
        ))

        # Layout
        fig.update_layout(
            title=dict(
                text='Training Progress',
                font=dict(size=24, family='Playfair Display')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.02)',
            font=dict(color='white'),
            xaxis=dict(
                title='Step',
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title='Loss',
                gridcolor='rgba(255,255,255,0.1)',
                titlefont=dict(color='#f87171'),
                tickfont=dict(color='#f87171')
            ),
            yaxis2=dict(
                title='Accuracy / Coherence',
                overlaying='y',
                side='right',
                gridcolor='rgba(255,255,255,0.05)',
                titlefont=dict(color='#34d399'),
                tickfont=dict(color='#34d399')
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_mycelial_viz(self):
        """Beautiful mycelial sensing visualization"""
        if not st.session_state.mycelial:
            st.info("Initialize model to see mycelial sensing")
            return

        # Get current state
        state = st.session_state.mycelial.sense_complete({
            'loss': np.random.rand(),
            'accuracy': np.random.rand()
        })

        st.markdown("### 🍄 Mycelial Sensing")

        # Octopus radar chart
        octopus_data = state.octopus_state.get('arm_activations', np.random.rand(8))

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=octopus_data,
            theta=[f'Arm {i+1}' for i in range(8)],
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='#667eea', width=3),
            name='Octopus Sensing'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                bgcolor='rgba(0,0,0,0)',
                angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Emergent properties
        st.markdown("### ✨ Emergent Properties")

        cols = st.columns(3)
        for i, (prop_name, prop_value) in enumerate(state.emergent_properties.items()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 1.5rem;">{prop_value:.3f}</div>
                    <div class="metric-label">{prop_name.replace('_', ' ').title()}</div>
                </div>
                """, unsafe_allow_html=True)

    def render_generation(self):
        """Simple generation interface"""
        st.markdown("### ✨ Generate")

        if not st.session_state.model:
            st.info("Initialize model first")
            return

        col1, col2 = st.columns([3, 1])

        with col1:
            prompt = st.text_input("Prompt:", value="The meaning of life is")

        with col2:
            max_length = st.number_input("Max Length", 10, 200, 50)

        if VOICE_AVAILABLE and st.session_state.voice_enabled:
            if st.button("🎤 Voice Input"):
                voice = VoiceInterface()
                voice_text = voice.listen()
                if voice_text:
                    st.success(f"Heard: {voice_text}")
                    prompt = voice_text

        if st.button("Generate", type="primary"):
            with st.spinner("Generating..."):
                # Mock generation
                words = ["consciousness", "reality", "quantum", "awareness",
                        "light", "harmony", "truth", "beauty", "wisdom"]
                generated = prompt + " " + " ".join(np.random.choice(words, 10))

                st.markdown(f"**Generated:**")
                st.markdown(f"> {generated}")

                if VOICE_AVAILABLE and st.session_state.voice_enabled:
                    voice = VoiceInterface()
                    voice.speak(generated)

    def run(self):
        """Run the beautiful dashboard"""
        self.render_header()
        self.render_sidebar()

        # Main content
        tab1, tab2, tab3 = st.tabs(["📊 Training", "🍄 Mycelial", "✨ Generate"])

        with tab1:
            self.render_metrics()
            st.markdown("<br>", unsafe_allow_html=True)
            self.render_training_chart()

        with tab2:
            self.render_mycelial_viz()

        with tab3:
            self.render_generation()


def main():
    """Main entry point"""
    if not LUMINARK_AVAILABLE:
        st.error("LUMINARK not available. Install the framework first.")
        return

    dashboard = StreamlitDashboard()
    dashboard.run()


if __name__ == '__main__':
    main()
