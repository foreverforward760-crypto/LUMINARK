import React, { useState } from 'react';
import { stageLibrary } from '@/lib/stageLibrary';

/**
 * LUMINARK SYSTEM INDEX OVERLAY
 * Educational tabs: What / How / Changelog
 */

const SystemIndex = ({ isOpen, onClose, currentStage }) => {
    const [activeTab, setActiveTab] = useState('what');

    if (!isOpen) return null;

    const stageContent = stageLibrary[currentStage] || stageLibrary[0];

    return (
        <div className="system-index-overlay">
            <div className="system-index-container">

                {/* Header */}
                <div className="system-index-header">
                    <h2>📚 SYSTEM INDEX</h2>
                    <button onClick={onClose} className="close-btn">✕</button>
                </div>

                {/* Tabs */}
                <div className="system-index-tabs">
                    <button
                        className={`tab ${activeTab === 'what' ? 'active' : ''}`}
                        onClick={() => setActiveTab('what')}
                    >
                        What
                    </button>
                    <button
                        className={`tab ${activeTab === 'how' ? 'active' : ''}`}
                        onClick={() => setActiveTab('how')}
                    >
                        How
                    </button>
                    <button
                        className={`tab ${activeTab === 'changelog' ? 'active' : ''}`}
                        onClick={() => setActiveTab('changelog')}
                    >
                        Changelog
                    </button>
                </div>

                {/* Content */}
                <div className="system-index-content">

                    {/* WHAT TAB */}
                    {activeTab === 'what' && (
                        <div className="tab-content">
                            <h3>What is LUMINARK?</h3>

                            <section>
                                <h4>🧬 The SAP Framework</h4>
                                <p>
                                    LUMINARK is a consciousness diagnostic tool based on the <strong>SAP
                                        (Systemic Awareness Protocol)</strong> framework—a 10-stage model of
                                    developmental progression for individuals, organizations, and systems.
                                </p>
                                <p>
                                    Unlike static personality tests, SAP recognizes that consciousness
                                    evolves through predictable stages, each with distinct characteristics,
                                    challenges, and opportunities.
                                </p>
                            </section>

                            <section>
                                <h4>🔬 Research Foundation</h4>
                                <ul>
                                    <li><strong>Piaget (1952)</strong> - Cognitive development stages</li>
                                    <li><strong>Kegan (1982)</strong> - Subject-object theory</li>
                                    <li><strong>Cook-Greuter (2004)</strong> - Ego development</li>
                                    <li><strong>Kübler-Ross (1969)</strong> - Crisis transformation</li>
                                    <li><strong>Frankl (1946)</strong> - Meaning-making</li>
                                    <li><strong>Bateson (1972)</strong> - Systems theory</li>
                                </ul>
                            </section>

                            <section>
                                <h4>📊 The 10 Stages</h4>
                                <div className="stages-grid">
                                    {Object.entries(stageLibrary).map(([stage, content]) => (
                                        <div key={stage} className={`stage-card ${parseInt(stage) === currentStage ? 'current' : ''}`}>
                                            <div className="stage-number">{stage}</div>
                                            <div className="stage-name">{content.name}</div>
                                            <div className="stage-title">{content.title}</div>
                                        </div>
                                    ))}
                                </div>
                            </section>

                            <section>
                                <h4>🎯 Current Stage: {currentStage} - {stageContent.name}</h4>
                                <p className="current-stage-desc">{stageContent.title}</p>
                                <blockquote>{stageContent.shortGuidance}</blockquote>
                            </section>
                        </div>
                    )}

                    {/* HOW TAB */}
                    {activeTab === 'how' && (
                        <div className="tab-content">
                            <h3>How LUMINARK Works</h3>

                            <section>
                                <h4>📐 SPAT Vectors</h4>
                                <p>
                                    LUMINARK analyzes consciousness through 5 core dimensions (SPAT + Coherence):
                                </p>

                                <div className="vector-grid">
                                    <div className="vector-card">
                                        <div className="vector-icon">🔢</div>
                                        <h5>Complexity (C)</h5>
                                        <p>Information density, relationships, variables in your system</p>
                                    </div>

                                    <div className="vector-card">
                                        <div className="vector-icon">⚖️</div>
                                        <h5>Stability (S)</h5>
                                        <p>Structural integrity, predictability, resilience</p>
                                    </div>

                                    <div className="vector-card">
                                        <div className="vector-icon">⚡</div>
                                        <h5>Tension (T)</h5>
                                        <p>Internal pressure, drive for change, creative friction</p>
                                    </div>

                                    <div className="vector-card">
                                        <div className="vector-icon">🦎</div>
                                        <h5>Adaptability (A)</h5>
                                        <p>Capacity to pivot, learn, evolve, respond to change</p>
                                    </div>

                                    <div className="vector-card">
                                        <div className="vector-icon">🎯</div>
                                        <h5>Coherence (H)</h5>
                                        <p>Strategic alignment, cultural health, meaning-making</p>
                                    </div>
                                </div>
                            </section>

                            <section>
                                <h4>🧮 Stage Determination Algorithm</h4>
                                <pre className="code-block">
                                    {`function determineStage(vectors) {
  // Calculate distance to each stage's ideal profile
  let minDistance = Infinity;
  let bestStage = 0;
  
  for (let stage = 0; stage <= 9; stage++) {
    const ideal = STAGE_PROFILES[stage];
    const distance = euclideanDistance(vectors, ideal);
    
    if (distance < minDistance) {
      minDistance = distance;
      bestStage = stage;
    }
  }
  
  // Confidence = inverse of distance
  const confidence = (1 - minDistance / 10) * 100;
  
  return { stage: bestStage, confidence };
}`}
                                </pre>
                            </section>

                            <section>
                                <h4>⚛️ Quantum Entropy</h4>
                                <p>
                                    LUMINARK adds controlled quantum noise (0-20%) to prevent identical
                                    readings and simulate the inherent uncertainty in consciousness measurement.
                                </p>
                                <p>
                                    This reflects the reality that consciousness is not static—it fluctuates
                                    based on context, mood, and temporal factors.
                                </p>
                            </section>

                            <section>
                                <h4>🔍 Special Detection Algorithms</h4>
                                <ul>
                                    <li><strong>Permanence Trap (Stage 8):</strong> Detects hubris, rigidity, declining adaptability</li>
                                    <li><strong>Container Rule (Stage 4):</strong> Assesses if structure can hold complexity</li>
                                    <li><strong>Threshold Crisis (Stage 5→7):</strong> Predicts forced crisis from decision paralysis</li>
                                </ul>
                            </section>
                        </div>
                    )}

                    {/* CHANGELOG TAB */}
                    {activeTab === 'changelog' && (
                        <div className="tab-content">
                            <h3>Version History</h3>

                            <div className="changelog-entry">
                                <div className="version-header">
                                    <span className="version-number">v4.0.0</span>
                                    <span className="version-date">2026-01-27</span>
                                </div>
                                <h4>🎙️ Oracle Voice Interface</h4>
                                <ul>
                                    <li>Added Web Speech API integration</li>
                                    <li>ElevenLabs premium voice synthesis</li>
                                    <li>Stage-specific voice profiles (10 unique AI voices)</li>
                                    <li>Speech-to-text input recognition</li>
                                    <li>Ambient soundscapes per stage</li>
                                    <li>Real-time waveform visualization</li>
                                </ul>
                            </div>

                            <div className="changelog-entry">
                                <div className="version-header">
                                    <span className="version-number">v3.5.0</span>
                                    <span className="version-date">2026-01-27</span>
                                </div>
                                <h4>⚛️ Entropy Control & Fractal Address</h4>
                                <ul>
                                    <li>Interactive quantum entropy slider (0-20%)</li>
                                    <li>Real-time fractal address generation</li>
                                    <li>Address legend with component breakdown</li>
                                    <li>Reroll functionality for variability</li>
                                    <li>Quantum particle visualization</li>
                                </ul>
                            </div>

                            <div className="changelog-entry">
                                <div className="version-header">
                                    <span className="version-number">v3.0.0</span>
                                    <span className="version-date">2026-01-27</span>
                                </div>
                                <h4>📚 Research-Backed Content</h4>
                                <ul>
                                    <li>Added 9 academic research citations</li>
                                    <li>18 historical case studies across stages</li>
                                    <li>Validated assessment questions (3 scales)</li>
                                    <li>Evidence-based interventions per stage</li>
                                    <li>Crisis detection patterns with timelines</li>
                                    <li>Emergency resources for Stage 7</li>
                                </ul>
                            </div>

                            <div className="changelog-entry">
                                <div className="version-header">
                                    <span className="version-number">v2.0.0</span>
                                    <span className="version-date">2026-01-26</span>
                                </div>
                                <h4>🎨 Professional UI Polish</h4>
                                <ul>
                                    <li>Quantum processing animation</li>
                                    <li>Premium share functionality</li>
                                    <li>Scanline & noise effects</li>
                                    <li>Multi-vector selection</li>
                                    <li>Temporal focus options</li>
                                </ul>
                            </div>

                            <div className="changelog-entry">
                                <div className="version-header">
                                    <span className="version-number">v1.0.0</span>
                                    <span className="version-date">2026-01-20</span>
                                </div>
                                <h4>🚀 Initial Release</h4>
                                <ul>
                                    <li>10-stage SAP framework implementation</li>
                                    <li>SPAT vector analysis</li>
                                    <li>Stage determination algorithm</li>
                                    <li>Basic UI and results display</li>
                                </ul>
                            </div>
                        </div>
                    )}

                </div>

                {/* Footer */}
                <div className="system-index-footer">
                    <p>LUMINARK SAP Framework © 2026 | Research-backed consciousness diagnostics</p>
                </div>

            </div>
        </div>
    );
};

export default SystemIndex;
