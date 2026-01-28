import React, { useState, useEffect, useRef } from "react";
import { stageLibrary } from "@/lib/stageLibrary";
import { ElevenLabsClient } from "elevenlabs";
import { getVoiceForStage, speakWithStageVoice } from "@/lib/elevenLabsVoiceMap";
import EntropyControl from "./EntropyControl";
import SystemIndex from "./SystemIndex";
import '@/styles/oracle-voice.css';
import '@/styles/entropy-control.css';
import '@/styles/system-index.css';

// ===== CONFIG =====
const eleven = new ElevenLabsClient({
    apiKey: import.meta.env.VITE_ELEVENLABS_API_KEY
});

// Ambient sounds per stage
const ambientSounds = {
    0: "https://freesound.org/data/previews/145/145816_2613695-lq.mp3",
    1: "https://freesound.org/data/previews/123/123456_7891011-lq.mp3",
    2: "https://freesound.org/data/previews/234/234567_8901234-lq.mp3",
    3: "https://freesound.org/data/previews/345/345678_9012345-lq.mp3",
    4: "https://freesound.org/data/previews/456/456789_0123456-lq.mp3",
    5: "https://freesound.org/data/previews/567/567890_1234567-lq.mp3",
    6: "https://freesound.org/data/previews/678/678901_2345678-lq.mp3",
    7: "https://freesound.org/data/previews/789/789012_3456789-lq.mp3",
    8: "https://freesound.org/data/previews/890/890123_4567890-lq.mp3",
    9: "https://freesound.org/data/previews/901/901234_5678901-lq.mp3"
};

const phrases = [
    "Fractal pattern stabilized.",
    "System resonance achieved.",
    "Entropy recalculated.",
    "Signal aligned."
];

const StageResults = ({ currentStage, spatVectors, onRecalculate }) => {
    const stageContent = stageLibrary[currentStage] || stageLibrary[0];

    const [voices, setVoices] = useState([]);
    const [selectedVoice, setSelectedVoice] = useState(null);
    const [rate, setRate] = useState(0.95);
    const [pitch, setPitch] = useState(1.0);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [statusText, setStatusText] = useState("");
    const [useElevenLabs, setUseElevenLabs] = useState(false);

    const [ambientAudio, setAmbientAudio] = useState(null);
    const [ambientEnabled, setAmbientEnabled] = useState(true);

    const [transcript, setTranscript] = useState("");
    const [showSystemIndex, setShowSystemIndex] = useState(false);
    const [entropy, setEntropy] = useState(5);

    const synth = window.speechSynthesis;
    const recognitionRef = useRef(null);

    // ================= VOICE LOADING =================
    useEffect(() => {
        const loadVoices = () => {
            const available = synth.getVoices();
            if (!available.length) return;

            setVoices(available);

            // Use stage-specific voice profile
            const voiceProfile = stageContent.voiceProfile;
            let defaultVoice = available[0];

            if (voiceProfile?.preferredVoice) {
                defaultVoice = available.find(v =>
                    v.name.toLowerCase().includes(voiceProfile.preferredVoice)
                ) || available[0];
            }

            setSelectedVoice(defaultVoice);
            setPitch(voiceProfile?.pitch || 1.0);
            setRate(voiceProfile?.rate || 1.0);
        };

        loadVoices();
        synth.onvoiceschanged = loadVoices;
        return () => (synth.onvoiceschanged = null);
    }, [currentStage]);

    // ================= AMBIENT AUDIO =================
    useEffect(() => {
        if (!ambientEnabled) return;

        const url = ambientSounds[currentStage];
        if (!url) return;

        const audio = new Audio(url);
        audio.loop = true;
        audio.volume = 0.3;
        audio.play().catch(() => { });
        setAmbientAudio(audio);

        return () => {
            audio.pause();
            audio.currentTime = 0;
        };
    }, [currentStage, ambientEnabled]);

    // ================= SPEECH =================
    const speakBrowser = (text) => {
        if (!text || !selectedVoice) return;

        synth.cancel();
        setIsSpeaking(true);

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.voice = selectedVoice;
        utterance.pitch = pitch;
        utterance.rate = rate;

        utterance.onend = () => setIsSpeaking(false);
        utterance.onerror = () => setIsSpeaking(false);

        synth.speak(utterance);
    };

    const speakElevenLabsStage = async (text) => {
        try {
            setIsSpeaking(true);
            const audio = await speakWithStageVoice(eleven, text, currentStage);
            audio.play();
            audio.onended = () => setIsSpeaking(false);
        } catch (err) {
            console.error("ElevenLabs failed:", err);
            speakBrowser(text); // fallback
        }
    };

    const speak = (text) => {
        setStatusText(phrases[Math.floor(Math.random() * phrases.length)]);
        useElevenLabs ? speakElevenLabsStage(text) : speakBrowser(text);
    };

    // ================= SPEECH TO TEXT =================
    const startListening = () => {
        if (!("webkitSpeechRecognition" in window)) {
            alert("Speech recognition not supported.");
            return;
        }

        const recognition = new window.webkitSpeechRecognition();
        recognition.lang = "en-US";
        recognition.continuous = false;

        recognition.onresult = (event) => {
            setTranscript(event.results[0][0].transcript);
        };

        recognition.start();
        recognitionRef.current = recognition;
    };

    // ================= ENTROPY HANDLERS =================
    const handleEntropyChange = (newEntropy) => {
        setEntropy(newEntropy);
    };

    const handleRecalculate = async (entropyValue) => {
        if (onRecalculate) {
            await onRecalculate(entropyValue);
        }
    };

    // ================= FALLBACK =================
    if (!voices.length && !useElevenLabs) {
        return (
            <div className="p-4 border border-red-500 text-red-400 rounded">
                No voices available. Text only:
                <p>{stageContent.oracleGuidance}</p>
            </div>
        );
    }

    return (
        <div className="stage-result p-6 rounded-lg bg-gray-900 text-white">

            <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold">Stage {currentStage} - {stageContent.name}</h2>
                <button
                    onClick={() => setShowSystemIndex(true)}
                    className="oracle-button"
                >
                    📚 SYSTEM INDEX
                </button>
            </div>

            <p className="mb-4">{stageContent.oracleGuidance}</p>

            <p className="status-text">{statusText}</p>

            {/* Oracle Voice Controls */}
            <div className="oracle-controls">
                <h3>🎙️ Oracle Voice</h3>

                <div className="flex gap-4 flex-wrap">
                    <label>
                        Voice:
                        <select
                            value={selectedVoice?.name || ""}
                            onChange={(e) =>
                                setSelectedVoice(voices.find(v => v.name === e.target.value))
                            }
                        >
                            {voices.map(v => (
                                <option key={v.name}>{v.name}</option>
                            ))}
                        </select>
                    </label>

                    <label>
                        Speed
                        <input type="range" min="0.5" max="2" step="0.1" value={rate}
                            onChange={e => setRate(+e.target.value)} />
                        <span>{rate.toFixed(1)}x</span>
                    </label>

                    <label>
                        Pitch
                        <input type="range" min="0.5" max="2" step="0.1" value={pitch}
                            onChange={e => setPitch(+e.target.value)} />
                        <span>{pitch.toFixed(1)}</span>
                    </label>
                </div>

                <div className="flex gap-4 mt-4">
                    <button onClick={() => speak(stageContent.oracleGuidance)} className="oracle-button">
                        🎙️ Play Oracle
                    </button>

                    <button onClick={() => synth.cancel()} className="oracle-button stop">
                        ⏹️ Stop
                    </button>

                    <button onClick={startListening} className="oracle-button listen">
                        🎤 Speak Input
                    </button>
                </div>

                <div className="flex gap-4 mt-4">
                    <label className="ambient-toggle">
                        <input type="checkbox" checked={useElevenLabs} onChange={() => setUseElevenLabs(!useElevenLabs)} />
                        Use ElevenLabs (premium AI voices)
                    </label>

                    <label className="ambient-toggle">
                        <input type="checkbox" checked={ambientEnabled} onChange={() => setAmbientEnabled(!ambientEnabled)} />
                        Ambient Sound
                    </label>
                </div>

                {isSpeaking && (
                    <div className="waveform">
                        {[...Array(5)].map((_, i) => (
                            <div key={i} className={`bar delay-${i * 100}`} />
                        ))}
                    </div>
                )}

                {transcript && (
                    <div className="transcript-display">
                        <b>Heard:</b> {transcript}
                    </div>
                )}
            </div>

            {/* Entropy Control */}
            <EntropyControl
                onEntropyChange={handleEntropyChange}
                currentStage={currentStage}
                spatVectors={spatVectors}
                onRecalculate={handleRecalculate}
            />

            {/* System Index Overlay */}
            <SystemIndex
                isOpen={showSystemIndex}
                onClose={() => setShowSystemIndex(false)}
                currentStage={currentStage}
            />

        </div>
    );
};

export default StageResults;
