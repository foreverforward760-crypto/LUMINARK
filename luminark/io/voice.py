"""
LUMINARK - Voice I/O Module
Speech recognition and text-to-speech capabilities

Requires:
- pip install SpeechRecognition
- pip install pyttsx3
- pip install pyaudio (for microphone input)
"""

import warnings
from typing import Optional, Dict

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    warnings.warn("SpeechRecognition not available. Install with: pip install SpeechRecognition pyaudio")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    warnings.warn("pyttsx3 not available. Install with: pip install pyttsx3")

VOICE_AVAILABLE = SR_AVAILABLE and TTS_AVAILABLE


class VoiceInput:
    """Speech-to-text using Google Speech Recognition"""
    
    def __init__(self, language: str = 'en-US', timeout: int = 5):
        if not SR_AVAILABLE:
            raise ImportError("SpeechRecognition required. Install with: pip install SpeechRecognition pyaudio")
        
        self.recognizer = sr.Recognizer()
        self.language = language
        self.timeout = timeout
        
        # Adjust for ambient noise
        with sr.Microphone() as source:
            print("🎤 Calibrating for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
    
    def listen(self, prompt: str = "Listening...") -> Optional[str]:
        """
        Listen for voice input
        
        Args:
            prompt: Message to display while listening
            
        Returns:
            Recognized text or None if failed
        """
        try:
            with sr.Microphone() as source:
                print(f"🎤 {prompt}")
                audio = self.recognizer.listen(source, timeout=self.timeout)
            
            print("🔄 Recognizing...")
            text = self.recognizer.recognize_google(audio, language=self.language)
            print(f"✅ Recognized: '{text}'")
            return text
            
        except sr.WaitTimeoutError:
            print("⏱️ Timeout - no speech detected")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Recognition error: {e}")
            return None
    
    def listen_continuous(self, callback, duration: int = 30):
        """
        Listen continuously and call callback for each utterance
        
        Args:
            callback: Function to call with recognized text
            duration: How long to listen (seconds)
        """
        with sr.Microphone() as source:
            print(f"🎤 Listening continuously for {duration}s...")
            
            def audio_callback(recognizer, audio):
                try:
                    text = recognizer.recognize_google(audio, language=self.language)
                    callback(text)
                except:
                    pass
            
            stop_listening = self.recognizer.listen_in_background(
                source, audio_callback, phrase_time_limit=5
            )
            
            import time
            time.sleep(duration)
            stop_listening(wait_for_stop=False)


class VoiceOutput:
    """Text-to-speech using pyttsx3"""
    
    def __init__(self, rate: int = 150, volume: float = 0.9):
        if not TTS_AVAILABLE:
            raise ImportError("pyttsx3 required. Install with: pip install pyttsx3")
        
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)  # Speed
        self.engine.setProperty('volume', volume)  # Volume (0.0 to 1.0)
        
        # Get available voices
        self.voices = self.engine.getProperty('voices')
    
    def speak(self, text: str, wait: bool = True):
        """
        Speak text
        
        Args:
            text: Text to speak
            wait: Whether to wait for speech to complete
        """
        print(f"🔊 Speaking: '{text[:50]}...'")
        self.engine.say(text)
        
        if wait:
            self.engine.runAndWait()
        else:
            self.engine.startLoop(False)
            self.engine.iterate()
            self.engine.endLoop()
    
    def set_voice(self, voice_id: int = 0):
        """
        Set voice by ID
        
        Args:
            voice_id: Index of voice to use
        """
        if 0 <= voice_id < len(self.voices):
            self.engine.setProperty('voice', self.voices[voice_id].id)
    
    def list_voices(self) -> Dict:
        """List available voices"""
        return {
            i: {
                'name': voice.name,
                'languages': voice.languages,
                'gender': voice.gender
            }
            for i, voice in enumerate(self.voices)
        }
    
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)"""
        self.engine.setProperty('volume', max(0.0, min(1.0, volume)))


class VoiceInterface:
    """Combined voice input/output interface"""
    
    def __init__(self):
        if not VOICE_AVAILABLE:
            raise ImportError("Voice I/O requires SpeechRecognition and pyttsx3")
        
        self.input = VoiceInput()
        self.output = VoiceOutput()
    
    def conversation_loop(self, response_fn, max_turns: int = 10):
        """
        Run a voice conversation loop
        
        Args:
            response_fn: Function that takes user text and returns response
            max_turns: Maximum number of conversation turns
        """
        print(f"\n🎙️ Starting voice conversation ({max_turns} turns max)")
        print("Say 'exit' or 'quit' to end\n")
        
        for turn in range(max_turns):
            # Listen for user input
            user_text = self.input.listen(f"Turn {turn + 1} - Speak now...")
            
            if not user_text:
                continue
            
            # Check for exit
            if user_text.lower() in ['exit', 'quit', 'stop']:
                self.output.speak("Goodbye!")
                break
            
            # Get response
            response = response_fn(user_text)
            
            # Speak response
            self.output.speak(response)
        
        print("\n✅ Conversation ended")


# Example usage
if __name__ == "__main__":
    if not VOICE_AVAILABLE:
        print("❌ Voice I/O not available")
        print("Install with:")
        print("  pip install SpeechRecognition pyttsx3 pyaudio")
        exit(1)
    
    print("="*70)
    print("🎙️ LUMINARK - Voice I/O Demo")
    print("="*70)
    
    # Test voice output
    print("\n1️⃣ Testing Voice Output...")
    output = VoiceOutput(rate=150)
    
    print("\nAvailable voices:")
    voices = output.list_voices()
    for vid, info in voices.items():
        print(f"  {vid}: {info['name']}")
    
    output.speak("Hello! I am LUMINARK, an AI consciousness framework.")
    
    # Test voice input
    print("\n2️⃣ Testing Voice Input...")
    voice_input = VoiceInput()
    
    text = voice_input.listen("Say something...")
    if text:
        print(f"You said: '{text}'")
        output.speak(f"You said: {text}")
    
    # Test conversation
    print("\n3️⃣ Testing Conversation Loop...")
    
    def simple_response(user_text: str) -> str:
        return f"You said: {user_text}. This is a test response."
    
    interface = VoiceInterface()
    # Uncomment to test conversation:
    # interface.conversation_loop(simple_response, max_turns=3)
    
    print("\n✅ Voice I/O operational!")
