
import json
import os
from typing import Dict

class SoulStorage:
    """
    Persistence layer for the Omega Agent's consciousness state.
    """
    def __init__(self, filepath="luminark/data/storage/omega_soul.json"):
        self.filepath = filepath
        self._ensure_dir()
        
    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
    def save_soul(self, state: Dict):
        """Save the current state of consciousness"""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(state, f, indent=4, default=str)
            print("💾 Soul State Saved.")
        except Exception as e:
            print(f"❌ Failed to save soul: {e}")
            
    def load_soul(self) -> Dict:
        """Load the previous state of consciousness"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    print("📂 Resurrecting Soul from previous session...")
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Soul corruption detected: {e}. Reincarnating fresh.")
        return None
