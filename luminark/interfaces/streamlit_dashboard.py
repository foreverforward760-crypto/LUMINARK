"""
Streamlit Dashboard for LUMINARK
Interactive web UI for training, generation, and visualization
Install: pip install streamlit plotly
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import time
from pathlib import Path

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Plotly not available. Install with: pip install plotly")

# Import LUMINARK components
try:
    from luminark.core import LUMINARK
    from luminark.sensing import MycelialSensorySystem
    LUMINARK_AVAILABLE = True
except ImportError:
    LUMINARK_AVAILABLE = False
    st.error("LUMINARK core not available")

# Optional: Voice I/O
try:
    from luminark.interfaces.voice_io import VoiceInterface
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# Optional: FAISS Memory
try:
    from luminark.memory.faiss_memory import FAISSMemory
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class StreamlitDashboard:
    """
    Interactive Streamlit dashboard for LUMINARK

    Features:
    - Real-time training monitoring
    - Interactive text generation
    - Mycelial sensing visualization
    - Voice I/O integration (optional)
    - FAISS RAG memory (optional)
    - QA testing controls

    Usage:
        streamlit run luminark/interfaces/streamlit_dashboard.py
    """

    def __init__(self):
        self.setup_page()
        self.initialize_session_state()

    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="LUMINARK Dashboard",
            page_icon="✨",
            layout="wide",
            initial_sidebar_state="expanded"
        )

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
                'quantum_confidence': [],
                'mycelial_coherence': []
            }

        if 'voice_enabled' not in st.session_state:
            st.session_state.voice_enabled = False

        if 'faiss_memory' not in st.session_state:
            st.session_state.faiss_memory = None

        if 'generation_history' not in st.session_state:
            st.session_state.generation_history = []

    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.title("🌟 LUMINARK Control")

        # Model initialization
        st.sidebar.header("Model Setup")

        vocab_size = st.sidebar.number_input(
            "Vocabulary Size",
            min_value=100,
            max_value=100000,
            value=10000
        )

        hidden_dim = st.sidebar.number_input(
            "Hidden Dimension",
            min_value=64,
            max_value=2048,
            value=256
        )

        if st.sidebar.button("Initialize Model"):
            self.initialize_model(vocab_size, hidden_dim)

        # Training controls
        st.sidebar.header("Training")

        batch_size = st.sidebar.slider("Batch Size", 1, 128, 32)
        learning_rate = st.sidebar.slider(
            "Learning Rate",
            0.0001,
            0.1,
            0.001,
            format="%.4f"
        )

        if st.sidebar.button("Run Training Epoch"):
            if st.session_state.model:
                self.run_training_epoch(batch_size, learning_rate)
            else:
                st.sidebar.error("Initialize model first")

        # Optional features
        st.sidebar.header("Optional Features")

        if VOICE_AVAILABLE:
            st.session_state.voice_enabled = st.sidebar.checkbox(
                "Enable Voice I/O",
                value=st.session_state.voice_enabled
            )

        if FAISS_AVAILABLE:
            if st.sidebar.button("Initialize FAISS Memory"):
                self.initialize_faiss_memory()

    def initialize_model(self, vocab_size: int, hidden_dim: int):
        """Initialize LUMINARK model and mycelial system"""
        with st.spinner("Initializing LUMINARK model..."):
            # Create model
            st.session_state.model = LUMINARK(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                output_dim=vocab_size
            )

            # Create mycelial sensory system
            st.session_state.mycelial = MycelialSensorySystem()

            st.success(f"✅ Model initialized (vocab={vocab_size}, hidden={hidden_dim})")
            st.balloons()

    def initialize_faiss_memory(self):
        """Initialize FAISS memory for RAG"""
        if st.session_state.model:
            hidden_dim = st.session_state.model.hidden_dim
            st.session_state.faiss_memory = FAISSMemory(
                dimension=hidden_dim,
                index_type='flat'
            )
            st.sidebar.success("✅ FAISS memory initialized")
        else:
            st.sidebar.error("Initialize model first")

    def run_training_epoch(self, batch_size: int, learning_rate: float):
        """Simulate training epoch"""
        model = st.session_state.model
        mycelial = st.session_state.mycelial

        # Mock training data
        progress_bar = st.progress(0)
        status_text = st.empty()

        for step in range(10):
            # Simulate training step
            time.sleep(0.1)

            # Mock metrics
            loss = np.random.rand() * 0.5 + 0.5 * (1 - step/10)
            accuracy = np.random.rand() * 0.3 + 0.7 * (step/10)

            # Get mycelial state
            training_metrics = {
                'loss': loss,
                'accuracy': accuracy,
                'learning_rate': learning_rate
            }

            mycelial_state = mycelial.sense_complete(training_metrics)

            # Update history
            st.session_state.training_history['loss'].append(loss)
            st.session_state.training_history['accuracy'].append(accuracy)
            st.session_state.training_history['sar_stage'].append(
                mycelial_state.sap_state.current_stage
            )
            st.session_state.training_history['quantum_confidence'].append(
                mycelial_state.sap_state.quantum_confidence
            )
            st.session_state.training_history['mycelial_coherence'].append(
                mycelial_state.overall_coherence
            )

            progress_bar.progress((step + 1) / 10)
            status_text.text(f"Step {step+1}/10 - Loss: {loss:.4f}")

        st.success("✅ Training epoch complete!")

    def render_training_metrics(self):
        """Render training metrics visualizations"""
        st.header("📊 Training Metrics")

        if not st.session_state.training_history['loss']:
            st.info("No training data yet. Run a training epoch to see metrics.")
            return

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Loss & Accuracy',
                'SAR Stage Progression',
                'Quantum Confidence',
                'Mycelial Coherence'
            )
        )

        history = st.session_state.training_history
        steps = list(range(len(history['loss'])))

        # Loss & Accuracy
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=history['loss'],
                name='Loss',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=history['accuracy'],
                name='Accuracy',
                line=dict(color='green', width=2),
                yaxis='y2'
            ),
            row=1, col=1
        )

        # SAR Stage
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=history['sar_stage'],
                name='SAR Stage',
                mode='lines+markers',
                line=dict(color='purple', width=2)
            ),
            row=1, col=2
        )

        # Quantum Confidence
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=history['quantum_confidence'],
                name='Quantum Confidence',
                fill='tozeroy',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )

        # Mycelial Coherence
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=history['mycelial_coherence'],
                name='Mycelial Coherence',
                fill='tozeroy',
                line=dict(color='orange', width=2)
            ),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    def render_mycelial_dashboard(self):
        """Render mycelial sensing dashboard"""
        st.header("🍄 Mycelial Sensory System")

        if not st.session_state.mycelial:
            st.info("Initialize model to see mycelial sensing data")
            return

        # Get current state
        mycelial = st.session_state.mycelial

        # Mock current metrics
        current_metrics = {
            'loss': np.random.rand(),
            'accuracy': np.random.rand(),
            'learning_rate': 0.001
        }

        state = mycelial.sense_complete(current_metrics)

        # Display overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Overall Coherence",
                f"{state.overall_coherence:.3f}",
                delta=f"{np.random.rand()*0.1:.3f}"
            )

        with col2:
            st.metric(
                "SAR Stage",
                f"{state.sap_state.current_stage}/81",
                delta=f"{state.sap_state.stage_energy:.2f}"
            )

        with col3:
            st.metric(
                "Quantum Confidence",
                f"{state.sap_state.quantum_confidence:.3f}"
            )

        with col4:
            st.metric(
                "Resonance 369",
                f"{state.resonance_369.control_circuit_strength:.3f}"
            )

        # Detailed sensing data
        st.subheader("Sensory Modalities")

        # Create radar chart for octopus arms
        octopus_data = state.octopus_state.get('arm_activations', np.random.rand(8))

        fig = go.Figure(data=go.Scatterpolar(
            r=octopus_data,
            theta=[f'Arm {i}' for i in range(8)],
            fill='toself',
            name='Octopus Sensory'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="Octopus Distributed Sensing"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Thermal sensing
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Thermal Energy Spectrum")
            thermal_data = state.thermal_state.get('spectrum_values', np.random.rand(8))
            spectrum_names = [
                'Thermal IR', 'Near IR', 'Visible', 'UV',
                'EM', 'Kinetic', 'Potential', 'Quantum'
            ]

            fig = go.Figure(data=[
                go.Bar(x=spectrum_names, y=thermal_data)
            ])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Bio-Sensory Fusion")
            bio_data = state.bio_fusion_state.get('modality_weights', np.random.rand(8))
            modality_names = [
                'Touch', 'Thermal', 'Chemical', 'EM',
                'Acoustic', 'Visual', 'Proprioception', 'Quantum'
            ]

            fig = go.Figure(data=[
                go.Bar(x=modality_names, y=bio_data)
            ])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Emergent properties
        st.subheader("Emergent Properties")

        emergent_cols = st.columns(3)
        emergent_properties = state.emergent_properties

        for i, (prop_name, prop_value) in enumerate(emergent_properties.items()):
            with emergent_cols[i % 3]:
                st.metric(
                    prop_name.replace('_', ' ').title(),
                    f"{prop_value:.3f}"
                )

    def render_generation_interface(self):
        """Render text generation interface"""
        st.header("✨ Interactive Generation")

        if not st.session_state.model:
            st.info("Initialize model to generate text")
            return

        # Input options
        col1, col2 = st.columns([3, 1])

        with col1:
            prompt = st.text_input(
                "Enter prompt:",
                value="The meaning of life is"
            )

        with col2:
            max_length = st.number_input(
                "Max Length",
                min_value=10,
                max_value=500,
                value=50
            )

        # Voice input (if available)
        if VOICE_AVAILABLE and st.session_state.voice_enabled:
            if st.button("🎤 Use Voice Input"):
                with st.spinner("Listening..."):
                    voice = VoiceInterface()
                    voice_text = voice.listen()
                    if voice_text:
                        prompt = voice_text
                        st.success(f"Voice input: {voice_text}")

        # Generation button
        if st.button("Generate", type="primary"):
            with st.spinner("Generating..."):
                # Mock generation (replace with actual model inference)
                generated_text = self.generate_text(prompt, max_length)

                # Display result
                st.markdown("### Generated Text:")
                st.markdown(f"> {generated_text}")

                # Voice output (if available)
                if VOICE_AVAILABLE and st.session_state.voice_enabled:
                    voice = VoiceInterface()
                    voice.speak(generated_text)

                # Save to history
                st.session_state.generation_history.append({
                    'prompt': prompt,
                    'generated': generated_text,
                    'timestamp': time.time()
                })

                # RAG memory (if available)
                if st.session_state.faiss_memory:
                    # Mock embedding
                    embedding = np.random.rand(st.session_state.model.hidden_dim)
                    st.session_state.faiss_memory.add(
                        embedding.reshape(1, -1),
                        texts=[generated_text]
                    )

        # Generation history
        if st.session_state.generation_history:
            st.subheader("Generation History")

            for i, entry in enumerate(reversed(st.session_state.generation_history[-5:])):
                with st.expander(f"Generation {len(st.session_state.generation_history)-i}"):
                    st.write(f"**Prompt:** {entry['prompt']}")
                    st.write(f"**Generated:** {entry['generated']}")

    def generate_text(self, prompt: str, max_length: int) -> str:
        """Generate text (mock implementation)"""
        # Mock generation - replace with actual model inference
        words = ["consciousness", "reality", "quantum", "awareness",
                "light", "harmony", "truth", "beauty", "wisdom"]

        generated = prompt + " "
        for _ in range(max_length // 10):
            generated += np.random.choice(words) + " "

        return generated.strip()

    def render_qa_testing(self):
        """Render QA testing interface"""
        st.header("🧪 QA Testing")

        st.markdown("""
        Test LUMINARK's capabilities across different perspectives:
        - **Awareness Testing**: Consciousness and self-reflection
        - **Reality Grounding**: Physical world understanding
        - **Creative Synthesis**: Novel idea generation
        """)

        test_type = st.selectbox(
            "Test Type",
            ["Awareness Testing", "Reality Grounding", "Creative Synthesis"]
        )

        question = st.text_area(
            "Test Question",
            value="What is the nature of consciousness?"
        )

        if st.button("Run QA Test"):
            with st.spinner("Testing..."):
                # Mock QA response
                response = self.run_qa_test(question, test_type)

                st.markdown("### Response:")
                st.info(response)

                # Show mycelial state during response
                if st.session_state.mycelial:
                    state = st.session_state.mycelial.sense_complete({
                        'test_type': test_type,
                        'question_complexity': len(question)
                    })

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Coherence", f"{state.overall_coherence:.3f}")
                    with col2:
                        st.metric("Confidence", f"{state.sap_state.quantum_confidence:.3f}")
                    with col3:
                        st.metric("SAR Stage", f"{state.sap_state.current_stage}")

    def run_qa_test(self, question: str, test_type: str) -> str:
        """Run QA test (mock implementation)"""
        # Mock response based on test type
        responses = {
            "Awareness Testing": "Consciousness emerges from the interplay of quantum coherence and classical information processing...",
            "Reality Grounding": "Physical reality manifests through quantum field interactions constrained by spacetime geometry...",
            "Creative Synthesis": "Novel insights arise from the fusion of disparate concepts in high-dimensional semantic space..."
        }

        return responses.get(test_type, "Processing question...")

    def render_settings(self):
        """Render settings panel"""
        st.header("⚙️ Settings")

        st.subheader("Model Configuration")

        if st.session_state.model:
            st.write(f"**Vocabulary Size:** {st.session_state.model.vocab_size}")
            st.write(f"**Hidden Dimension:** {st.session_state.model.hidden_dim}")
            st.write(f"**Output Dimension:** {st.session_state.model.output_dim}")

        st.subheader("Optional Features")

        st.write(f"**Voice I/O Available:** {'✅' if VOICE_AVAILABLE else '❌'}")
        st.write(f"**FAISS Memory Available:** {'✅' if FAISS_AVAILABLE else '❌'}")
        st.write(f"**Plotly Available:** {'✅' if PLOTLY_AVAILABLE else '❌'}")

        st.subheader("Data Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Clear Training History"):
                st.session_state.training_history = {
                    'loss': [],
                    'accuracy': [],
                    'sar_stage': [],
                    'quantum_confidence': [],
                    'mycelial_coherence': []
                }
                st.success("Training history cleared")

        with col2:
            if st.button("Clear Generation History"):
                st.session_state.generation_history = []
                st.success("Generation history cleared")

    def run(self):
        """Run the dashboard"""
        # Title
        st.title("✨ LUMINARK Interactive Dashboard")
        st.markdown("**Quantum-Aware AI Framework with Mycelial Sensing**")

        # Sidebar
        self.render_sidebar()

        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Training",
            "🍄 Mycelial Sensing",
            "✨ Generation",
            "🧪 QA Testing",
            "⚙️ Settings"
        ])

        with tab1:
            self.render_training_metrics()

        with tab2:
            self.render_mycelial_dashboard()

        with tab3:
            self.render_generation_interface()

        with tab4:
            self.render_qa_testing()

        with tab5:
            self.render_settings()


def main():
    """Main entry point"""
    if not LUMINARK_AVAILABLE:
        st.error("LUMINARK not available. Install the framework first.")
        return

    dashboard = StreamlitDashboard()
    dashboard.run()


if __name__ == '__main__':
    main()
