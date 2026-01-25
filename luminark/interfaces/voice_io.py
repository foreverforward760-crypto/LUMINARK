"""
Voice I/O Interface for LUMINARK
Optional speech recognition and text-to-speech
Install: pip install speechrecognition pyttsx3 pyaudio
"""
import sys
from typing import Optional

try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_DEPS_AVAILABLE = True
except ImportError:
    VOICE_DEPS_AVAILABLE = False
    print("Voice I/O disabled. Install with: pip install speechrecognition pyttsx3 pyaudio")


class VoiceInterface:
    """
    Speech recognition and text-to-speech interface

    Usage:
        voice = VoiceInterface()
        text = voice.listen()  # Listen to microphone
        voice.speak("Hello!")   # Speak text
    """

    def __init__(self, rate: int = 150, volume: float = 1.0):
        if not VOICE_DEPS_AVAILABLE:
            raise ImportError("Voice dependencies not available. Install: pip install speechrecognition pyttsx3 pyaudio")

        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

        # Configure TTS engine
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

        # Available voices
        voices = self.engine.getProperty('voices')
        self.available_voices = [(i, v.name) for i, v in enumerate(voices)]

        print(f"🎤 Voice Interface initialized")
        print(f"   Available voices: {len(self.available_voices)}")

    def list_voices(self):
        """List available TTS voices"""
        print("\nAvailable voices:")
        for idx, name in self.available_voices:
            print(f"  {idx}: {name}")

    def set_voice(self, voice_index: int = 0):
        """Set TTS voice by index"""
        voices = self.engine.getProperty('voices')
        if 0 <= voice_index < len(voices):
            self.engine.setProperty('voice', voices[voice_index].id)
            print(f"Voice set to: {voices[voice_index].name}")
        else:
            print(f"Invalid voice index. Use 0-{len(voices)-1}")

    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """
        Listen to microphone and return recognized text

        Args:
            timeout: Max seconds to wait for speech to start
            phrase_time_limit: Max seconds for phrase

        Returns:
            Recognized text or None if failed
        """
        try:
            with sr.Microphone() as source:
                print("🎤 Listening... (speak now)")

                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                # Listen
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

                print("🔄 Processing speech...")

                # Recognize using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                print(f"✓ Recognized: {text}")
                return text

        except sr.WaitTimeoutError:
            print("⏱️  No speech detected (timeout)")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    def speak(self, text: str, block: bool = True):
        """
        Convert text to speech

        Args:
            text: Text to speak
            block: Wait for speech to complete
        """
        if not text:
            return

        print(f"🔊 Speaking: {text[:50]}...")
        self.engine.say(text)

        if block:
            self.engine.runAndWait()
        else:
            # Non-blocking (speech happens in background)
            self.engine.startLoop(False)
            self.engine.iterate()
            self.engine.endLoop()

    def interactive_session(self):
        """
        Run interactive voice session
        Say "exit" or "quit" to end
        """
        print("\n🎤 Starting interactive voice session")
        print("   Say 'exit' or 'quit' to end")
        print("=" * 50)

        self.speak("Voice interface ready")

        while True:
            text = self.listen()

            if text is None:
                continue

            if text.lower() in ['exit', 'quit', 'stop', 'end']:
                self.speak("Goodbye")
                break

            # Echo back what was heard
            response = f"You said: {text}"
            self.speak(response)

    def get_voice_prompt(self,
                        prompt_text: str = "Please provide input",
                        max_attempts: int = 3) -> Optional[str]:
        """
        Get voice input with prompt

        Args:
            prompt_text: TTS prompt to user
            max_attempts: Max recognition attempts

        Returns:
            Recognized text or None
        """
        self.speak(prompt_text)

        for attempt in range(max_attempts):
            text = self.listen()
            if text:
                # Confirm
                self.speak(f"I heard: {text}. Is this correct?")
                confirmation = self.listen(timeout=3)

                if confirmation and any(word in confirmation.lower()
                                       for word in ['yes', 'correct', 'right', 'yeah']):
                    return text
                else:
                    if attempt < max_attempts - 1:
                        self.speak("Let's try again")

        self.speak("Could not get valid input")
        return None


# Utility functions for quick access
def quick_listen() -> Optional[str]:
    """Quick listen without creating interface object"""
    if not VOICE_DEPS_AVAILABLE:
        print("Voice I/O not available")
        return None
    voice = VoiceInterface()
    return voice.listen()


def quick_speak(text: str):
    """Quick speak without creating interface object"""
    if not VOICE_DEPS_AVAILABLE:
        print("Voice I/O not available")
        return
    voice = VoiceInterface()
    voice.speak(text)


if __name__ == '__main__':
    # Demo
    print("🎤 Voice I/O Demo\n")

    if not VOICE_DEPS_AVAILABLE:
        print("Install dependencies first:")
        print("  pip install speechrecognition pyttsx3 pyaudio")
        sys.exit(1)

    voice = VoiceInterface()

    # List available voices
    voice.list_voices()

    # Demo TTS
    print("\n1. Text-to-Speech Demo")
    voice.speak("Hello! I am the LUMINARK voice interface.")

    # Demo speech recognition
    print("\n2. Speech Recognition Demo")
    print("   (You have 5 seconds to speak)")

    result = voice.listen()
    if result:
        voice.speak(f"You said: {result}")

    print("\n✅ Demo complete!")
