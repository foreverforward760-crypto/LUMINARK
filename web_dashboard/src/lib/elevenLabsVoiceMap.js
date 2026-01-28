/**
 * LUMINARK ELEVENLABS VOICE MAP
 * Stage-specific AI voice profiles for premium Oracle experience
 */

export const elevenLabsVoiceMap = {
    0: {
        voiceId: "21m00Tcm4TlvDq8ikWAM", // Rachel - Calm, neutral
        name: "Rachel",
        description: "Calm and contemplative, perfect for the void before form",
        settings: {
            stability: 0.75,
            similarity_boost: 0.75,
            style: 0.0,
            use_speaker_boost: true
        }
    },

    1: {
        voiceId: "ErXwobaYiN019PkySvjV", // Antoni - Energetic, young
        name: "Antoni",
        description: "Energetic and exploratory, captures the pulse of first movement",
        settings: {
            stability: 0.50,
            similarity_boost: 0.80,
            style: 0.30,
            use_speaker_boost: true
        }
    },

    2: {
        voiceId: "MF3mGyEYCl7XYWbV9V6O", // Elli - Clear, precise
        name: "Elli",
        description: "Clear and decisive, embodies the polarity split",
        settings: {
            stability: 0.85,
            similarity_boost: 0.75,
            style: 0.0,
            use_speaker_boost: true
        }
    },

    3: {
        voiceId: "TxGEqnHWrfWFTfGW9XjX", // Josh - Dynamic, powerful
        name: "Josh",
        description: "Dynamic and charismatic, channels breakthrough energy",
        settings: {
            stability: 0.40,
            similarity_boost: 0.85,
            style: 0.50,
            use_speaker_boost: true
        }
    },

    4: {
        voiceId: "VR6AewLTigWG4xSOukaG", // Arnold - Steady, reliable
        name: "Arnold",
        description: "Steady and grounded, represents sustainable foundation",
        settings: {
            stability: 0.90,
            similarity_boost: 0.70,
            style: 0.0,
            use_speaker_boost: true
        }
    },

    5: {
        voiceId: "AZnzlk1XvdvUeBnXmlld", // Domi - Contemplative
        name: "Domi",
        description: "Thoughtful and measured, guides threshold decisions",
        settings: {
            stability: 0.80,
            similarity_boost: 0.75,
            style: 0.10,
            use_speaker_boost: true
        }
    },

    6: {
        voiceId: "EXAVITQu4vr4xnSDxMaL", // Bella - Harmonious
        name: "Bella",
        description: "Harmonious and flowing, embodies integration",
        settings: {
            stability: 0.75,
            similarity_boost: 0.80,
            style: 0.20,
            use_speaker_boost: true
        }
    },

    7: {
        voiceId: "jsCqWAovK2LkecY7zXl4", // Freya - Compassionate
        name: "Freya",
        description: "Compassionate and gentle, supports crisis transformation",
        settings: {
            stability: 0.85,
            similarity_boost: 0.70,
            style: 0.0,
            use_speaker_boost: true
        }
    },

    8: {
        voiceId: "N2lVS1w4EtoT3dr4eOWO", // Callum - Warning, serious
        name: "Callum",
        description: "Serious and direct, warns of permanence trap",
        settings: {
            stability: 0.90,
            similarity_boost: 0.75,
            style: 0.0,
            use_speaker_boost: true
        }
    },

    9: {
        voiceId: "XB0fDUnXU5powFXDhCwa", // Charlotte - Wise, gentle
        name: "Charlotte",
        description: "Wise and serene, guides transparent return",
        settings: {
            stability: 0.85,
            similarity_boost: 0.70,
            style: 0.10,
            use_speaker_boost: true
        }
    }
};

/**
 * Get voice configuration for a specific stage
 */
export function getVoiceForStage(stage) {
    return elevenLabsVoiceMap[stage] || elevenLabsVoiceMap[0];
}

/**
 * Enhanced ElevenLabs TTS with stage-specific voices
 */
export async function speakWithStageVoice(eleven, text, stage) {
    const voiceConfig = getVoiceForStage(stage);

    try {
        const audioStream = await eleven.textToSpeech.convert(
            voiceConfig.voiceId,
            {
                text: text,
                model_id: "eleven_multilingual_v2",
                voice_settings: voiceConfig.settings
            }
        );

        // Convert stream to audio
        const chunks = [];
        for await (const chunk of audioStream) {
            chunks.push(chunk);
        }

        const audioBlob = new Blob(chunks, { type: 'audio/mpeg' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        return audio;

    } catch (error) {
        console.error(`ElevenLabs error for stage ${stage}:`, error);
        throw error;
    }
}

/**
 * Preload voices for faster playback
 */
export async function preloadStageVoices(eleven, stages = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) {
    const preloadPromises = stages.map(async (stage) => {
        const voiceConfig = getVoiceForStage(stage);
        const testText = "System initialized.";

        try {
            await speakWithStageVoice(eleven, testText, stage);
            console.log(`✅ Preloaded voice for Stage ${stage}: ${voiceConfig.name}`);
        } catch (error) {
            console.warn(`⚠️ Failed to preload Stage ${stage}:`, error);
        }
    });

    await Promise.allSettled(preloadPromises);
}

/**
 * Voice preview for UI
 */
export const voicePreviewText = {
    0: "You stand at the threshold of potential. All possibilities exist here.",
    1: "The void trembles. Something stirs. This is the pulse of creation.",
    2: "The world splits. You see clearly now: this versus that.",
    3: "You are the lightning bolt. Pure expression flows through you.",
    4: "You have built something that lasts. The structure holds.",
    5: "Two paths diverge. Both are valid. Both are terrifying.",
    6: "Everything works. Multiple systems dance in harmony.",
    7: "Everything falls apart. This is purification, not punishment.",
    8: "You believe you have arrived. This is the trap. Wake up.",
    9: "You have completed the cycle. Ego dissolves. Attachment releases."
};

/**
 * Get voice preview for testing
 */
export function getVoicePreview(stage) {
    return voicePreviewText[stage] || voicePreviewText[0];
}

// Export for use in components
export default {
    elevenLabsVoiceMap,
    getVoiceForStage,
    speakWithStageVoice,
    preloadStageVoices,
    getVoicePreview
};
