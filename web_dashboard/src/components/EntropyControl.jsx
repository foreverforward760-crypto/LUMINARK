import React, { useState, useEffect } from 'react';

/**
 * LUMINARK ENTROPY CONTROL
 * Interactive quantum noise slider with fractal address display
 */

const EntropyControl = ({
    onEntropyChange,
    currentStage,
    spatVectors,
    onRecalculate
}) => {
    const [entropy, setEntropy] = useState(5); // Default 5% quantum noise
    const [fractalAddress, setFractalAddress] = useState('');
    const [isRecalculating, setIsRecalculating] = useState(false);

    // Generate fractal address
    useEffect(() => {
        if (spatVectors && currentStage !== undefined) {
            const timestamp = Date.now() % 1000;
            const temporal = Math.floor(spatVectors.t * 9);
            const spatial = Math.floor(spatVectors.s * 9);
            const complexity = Math.floor(spatVectors.c * 9);

            const address = `FRAC.${currentStage}.${temporal}.${spatial}.${complexity}.${timestamp}`;
            setFractalAddress(address);
        }
    }, [spatVectors, currentStage, entropy]);

    const handleEntropyChange = (e) => {
        const newEntropy = parseFloat(e.target.value);
        setEntropy(newEntropy);
        if (onEntropyChange) {
            onEntropyChange(newEntropy);
        }
    };

    const handleRecalculate = async () => {
        setIsRecalculating(true);
        if (onRecalculate) {
            await onRecalculate(entropy);
        }
        setTimeout(() => setIsRecalculating(false), 1000);
    };

    const getEntropyDescription = (value) => {
        if (value === 0) return "Deterministic (No Variability)";
        if (value <= 2) return "Minimal Noise (High Precision)";
        if (value <= 5) return "Balanced (Recommended)";
        if (value <= 10) return "High Variability (Exploratory)";
        if (value <= 20) return "Chaotic (Maximum Uncertainty)";
        return "Extreme Chaos";
    };

    const getEntropyColor = (value) => {
        if (value === 0) return "#6b7280"; // Gray
        if (value <= 5) return "#10b981"; // Green
        if (value <= 10) return "#3b82f6"; // Blue
        if (value <= 20) return "#f59e0b"; // Orange
        return "#ef4444"; // Red
    };

    return (
        <div className="entropy-control-container">
            {/* Entropy Slider */}
            <div className="entropy-slider-section">
                <div className="entropy-header">
                    <h3>⚛️ Quantum Entropy</h3>
                    <span className="entropy-value" style={{ color: getEntropyColor(entropy) }}>
                        {entropy}%
                    </span>
                </div>

                <input
                    type="range"
                    min="0"
                    max="20"
                    step="0.5"
                    value={entropy}
                    onChange={handleEntropyChange}
                    className="entropy-slider"
                    style={{
                        background: `linear-gradient(90deg, 
              #10b981 0%, 
              #3b82f6 25%, 
              #f59e0b 50%, 
              #ef4444 100%)`
                    }}
                />

                <div className="entropy-description">
                    {getEntropyDescription(entropy)}
                </div>

                <div className="entropy-info">
                    <p className="info-text">
                        Entropy adds quantum noise to your reading. Higher values create more
                        variability between readings with the same inputs.
                    </p>
                </div>
            </div>

            {/* Fractal Address Display */}
            <div className="fractal-address-section">
                <div className="fractal-header">
                    <h3>🔢 Fractal Address</h3>
                    <button
                        onClick={handleRecalculate}
                        disabled={isRecalculating}
                        className="recalculate-btn"
                    >
                        {isRecalculating ? '⟳ Recalculating...' : '🔄 Reroll'}
                    </button>
                </div>

                <div className="fractal-address-display">
                    <code>{fractalAddress || 'FRAC.0.0.0.0.000'}</code>
                </div>

                <div className="fractal-legend">
                    <h4>Address Legend:</h4>
                    <div className="legend-grid">
                        <div className="legend-item">
                            <span className="legend-label">Stage:</span>
                            <span className="legend-value">{currentStage}</span>
                        </div>
                        <div className="legend-item">
                            <span className="legend-label">Temporal:</span>
                            <span className="legend-value">
                                {spatVectors ? Math.floor(spatVectors.t * 9) : 0}
                            </span>
                        </div>
                        <div className="legend-item">
                            <span className="legend-label">Spatial:</span>
                            <span className="legend-value">
                                {spatVectors ? Math.floor(spatVectors.s * 9) : 0}
                            </span>
                        </div>
                        <div className="legend-item">
                            <span className="legend-label">Complexity:</span>
                            <span className="legend-value">
                                {spatVectors ? Math.floor(spatVectors.c * 9) : 0}
                            </span>
                        </div>
                        <div className="legend-item">
                            <span className="legend-label">Timestamp:</span>
                            <span className="legend-value">{Date.now() % 1000}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Quantum Noise Visualization */}
            <div className="noise-visualization">
                <div className="noise-particles">
                    {[...Array(Math.floor(entropy * 2))].map((_, i) => (
                        <div
                            key={i}
                            className="particle"
                            style={{
                                left: `${Math.random() * 100}%`,
                                animationDelay: `${Math.random() * 2}s`,
                                animationDuration: `${2 + Math.random() * 2}s`
                            }}
                        />
                    ))}
                </div>
            </div>
        </div>
    );
};

export default EntropyControl;
