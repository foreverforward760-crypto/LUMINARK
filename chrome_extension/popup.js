// SAP Stage Detection Logic
const stageInfo = {
    0: { name: "Stage 0: Plenara", desc: "Primordial, receptive state", guidance: "Allow, don't force" },
    1: { name: "Stage 1: Spark", desc: "Initial ignition", guidance: "Nurture small steps" },
    2: { name: "Stage 2: Polarity", desc: "Binary thinking", guidance: "Choices aren't permanent" },
    3: { name: "Stage 3: Motion", desc: "Action and momentum", guidance: "Keep moving, check direction" },
    4: { name: "Stage 4: Foundation", desc: "Structure and stability", guidance: "Use this platform wisely" },
    5: { name: "Stage 5: Threshold", desc: "Critical decision point", guidance: "Take your time" },
    6: { name: "Stage 6: Integration", desc: "Nuanced thinking", guidance: "Embrace complexity" },
    7: { name: "Stage 7: Illusion", desc: "Testing reality", guidance: "Healthy skepticism" },
    8: { name: "Stage 8: Rigidity", desc: "⚠️ TRAP RISK", guidance: "Stay flexible!" },
    9: { name: "Stage 9: Renewal", desc: "Transcendence", guidance: "Let go, create space" }
};

function analyzeText(text) {
    // Keyword-based stage detection
    const keywords = {
        0: ['lost', 'confused', 'formless', 'undefined', 'void'],
        1: ['new', 'beginning', 'start', 'first', 'spark'],
        2: ['either', 'or', 'choice', 'binary', 'black and white'],
        3: ['action', 'doing', 'moving', 'executing', 'momentum'],
        4: ['stable', 'structure', 'foundation', 'organized', 'system'],
        5: ['crossroads', 'decision', 'threshold', 'critical', 'turning point'],
        6: ['both and', 'integrate', 'nuanced', 'complex', 'balance'],
        7: ['question', 'doubt', 'test', 'uncertain', 'maybe'],
        8: ['always', 'never', 'absolute', 'permanent', 'guaranteed', 'certain'],
        9: ['let go', 'release', 'transcend', 'beyond', 'renewal']
    };

    const textLower = text.toLowerCase();
    const scores = {};

    // Count keyword matches
    for (let stage in keywords) {
        scores[stage] = keywords[stage].filter(kw => textLower.includes(kw)).length;
    }

    // Find stage with most matches
    let maxStage = 4; // Default to Foundation
    let maxScore = 0;

    for (let stage in scores) {
        if (scores[stage] > maxScore) {
            maxScore = scores[stage];
            maxStage = parseInt(stage);
        }
    }

    return maxStage;
}

document.getElementById('analyzeBtn').addEventListener('click', async () => {
    // Get selected text from active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: () => window.getSelection().toString()
    }, (results) => {
        const selectedText = results[0].result;

        if (!selectedText || selectedText.length < 10) {
            alert('Please select some text first (at least 10 characters)');
            return;
        }

        // Show loading
        document.getElementById('loading').style.display = 'block';
        document.getElementById('result').style.display = 'none';

        // Analyze
        setTimeout(() => {
            const stage = analyzeText(selectedText);
            const info = stageInfo[stage];

            // Display results
            document.getElementById('stageBadge').textContent = info.name;
            document.getElementById('stageDesc').textContent = info.desc;
            document.getElementById('guidance').textContent = '💡 ' + info.guidance;

            document.getElementById('loading').style.display = 'none';
            document.getElementById('result').style.display = 'block';

            // Track usage (in production, check limits)
            chrome.storage.local.get(['usageCount'], (result) => {
                const count = (result.usageCount || 0) + 1;
                chrome.storage.local.set({ usageCount: count });
            });
        }, 500);
    });
});
