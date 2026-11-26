"""
LUMINARK WEB APP DATA INTEGRATION
Provides interpretive research references and historical examples for an experimental reflection tool
"""

# This content may be added to a UI as interpretive examples; it is not research validation of SAP.

ENHANCED_STAGE_CONTENT = {
    1: {
        "research_note": "Stage 1 aligns with Piaget's sensorimotor stage - exploring without clear direction.",
        "historical_example": "Steve Jobs founding Apple in his garage (1976) - pure experimentation, no business plan.",
        "intervention": "Daily journaling (15 min) - Expressive writing has been studied in psychology; this specific SAP mapping and expected outcome are not validated.",
        "warning": "Consider whether prolonged overload is affecting focus; this is not a forecast."
    },
    
    2: {
        "research_note": "Kegan's 'Imperial Mind' - clear self vs. other distinction emerges.",
        "historical_example": "Facebook 2005 - 'TheFacebook for college students only' (clear identity vs. MySpace).",
        "intervention": "Define your 'not-this' as clearly as your 'this'. Clarity comes from boundaries.",
        "warning": "Trying to serve everyone = serving no one. Make the split."
    },
    
    3: {
        "research_note": "Peak flow state (Csikszentmihalyi) - skills meeting challenge perfectly.",
        "historical_example": "Muhammad Ali 1964-1967: 'I am the greatest!' - pure self-expression at peak.",
        "intervention": "Ride the wave but prepare for Stage 4. Build stable support structures alongside creative activity; no population percentage is established.",
        "warning": "Celebrity/success can become identity. When wave ends, who are you?"
    },
    
    4: {
        "research_note": "Antonovsky's 'Sense of Coherence' - comprehensibility, manageability, meaningfulness aligned.",
        "historical_example": "Warren Buffett 1980s-2000s - sustainable value investing, no drama, consistent principles.",
        "intervention": "Quarterly life audit (Covey's 7 Habits). Maintain balance, avoid complacency.",
        "warning": "Boredom = readiness for Stage 5. Excessive comfort = adaptability dropping (Stage 8 risk)."
    },
    
    5: {
        "research_note": "Bateson's 'Double Bind' - two conflicting demands, both valid, must choose.",
        "historical_example": "Nelson Mandela 1990 - Choose revenge or reconciliation? Both justified, only one wise.",
        "intervention": "Threshold Decision Protocol: Name it, assess adaptability, decide within 30 days.",
        "warning": "Decision paralysis >6 months = low adaptability = the framework does not predict a crisis."
    },
    
    6: {
        "research_note": "Cook-Greuter's 'Autonomous' stage - multiple perspectives integrated harmoniously.",
        "historical_example": "Apple 2015-2019 - iPhone peak + Services growing + Wearables launching (all working).",
        "intervention": "Enjoy it but don't claim permanence. the framework does not establish a fixed stage duration.",
        "warning": "Confusing Stage 6 with Stage 8 = fatal. Check: Is adaptability still high?"
    },
    
    7: {
        "research_note": "Kübler-Ross grief stages + Frankl's meaning-making in suffering.",
        "historical_example": "J.K. Rowling 1993-95 - Divorced, welfare, suicidal. Wrote Harry Potter in crisis.",
        "intervention": "ACT therapy (12-16 weeks), meaning reconstruction, somatic practices (trauma yoga).",
        "warning": "Suicidal ideation? Call 988 immediately. Crisis can be transformation or death."
    },
    
    8: {
        "research_note": "Hubris syndrome (Owen & Davidson, 2009) - power-induced brain changes reduce empathy.",
        "historical_example": "Lance Armstrong 2005 - 'I'm the most tested athlete ever, I'm clean' (lying while believing it).",
        "intervention": "Monthly permanence trap audit. Beginner's mind practice. Seek critics, not yes-men.",
        "warning": "Believing you're 'different' or rules don't apply = not a validated prediction or timeline."
    },
    
    9: {
        "research_note": "Erikson's 'Ego Integrity' - acceptance of life as lived, no regrets.",
        "historical_example": "David Bowie 2016 - Blackstar album as conscious death preparation. Died 2 days after release.",
        "intervention": "Serve others, mentor, release attachment to outcomes. Prepare for graceful exit.",
        "warning": "Stage 9 is a framework-defined state; prevalence and numeric requirements have not been empirically established."
    }
}


# Exploratory stage-pattern flagging snippet; not a clinical risk assessment.
CRISIS_DETECTION_ALGORITHM = """
// Illustrative SAP pattern flags only. This is not a clinical crisis detector,
// diagnostic tool, or validated forecast. Do not use it to predict timelines,
// assess suicide risk, or replace professional evaluation.

function detectStagePatternFlags(u, stage) {
    const flags = [];

    if (stage === 5 && u.a < 5.0) {
        flags.push("Framework-defined Stage 5 adaptability pattern flagged for reflection.");
    }
    if (stage === 5 && u.t > 8.0 && u.s < 4.0) {
        flags.push("High-tension / low-stability pattern flagged for reflection.");
    }
    if (stage === 8 && u.a < 5.0) {
        flags.push("Framework-defined Stage 8 permanence pattern flagged for reflection.");
    }
    if (stage === 8 && u.s > 8.5 && u.t < 3.0) {
        flags.push("High-stability / low-tension pattern flagged for reflection.");
    }
    if (u.h < 4.0) {
        flags.push("Low-coherence input flagged for reflection.");
    }

    return {
        status: flags.length ? "EXPLORATORY_PATTERN_FLAG" : "NO_PATTERN_FLAG",
        flags,
        notice: "Not a clinical assessment or forecast. If someone may be in immediate danger, contact emergency services or a qualified crisis service."
    };
}
"""


# Enhanced report template with research citations
ENHANCED_REPORT_TEMPLATE = """
<div class="research-section">
    <h3>🔬 Research Foundation</h3>
    <p class="research-note">{research_note}</p>
    
    <h3>📚 Historical Example</h3>
    <p class="historical-example">{historical_example}</p>
    
    <h3>💊 Recommended Practice</h3>
    <p class="intervention">{intervention}</p>
    
    <h3>⚠️ Warning Signs</h3>
    <p class="warning">{warning}</p>
</div>
"""


# Export for use in index.html
if __name__ == "__main__":
    import json
    
    # Save enhanced content
    with open('enhanced_stage_content.json', 'w') as f:
        json.dump(ENHANCED_STAGE_CONTENT, f, indent=2)
    
    print("✅ Enhanced stage content saved to enhanced_stage_content.json")
    print("\nTo integrate into index.html:")
    print("1. Load this JSON file")
    print("2. Add research sections to each stage display")
    print("3. Implement crisis detection algorithm")
    print("4. Add emergency resources for Stage 7")
