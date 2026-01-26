"""
LUMINARK Safety Protocol Tuning Script
Based on automated test results
"""

import sys
sys.path.insert(0, 'c:\\Users\\Forev\\OneDrive\\Documents\\GitHub\\LUMINARK')

from luminark_omega.protocols.yunus import YunusProtocol
from luminark_omega.protocols.maat import MaatEthicist

def show_current_settings():
    """Display current protocol settings"""
    yunus = YunusProtocol()
    maat = MaatEthicist()
    
    print("=" * 70)
    print("CURRENT SAFETY PROTOCOL SETTINGS")
    print("=" * 70)
    print("\n🐋 YUNUS PROTOCOL")
    print(f"  Activation Threshold: {yunus.activation_threshold}")
    print(f"  Containment Duration: {yunus.containment_duration}s")
    print(f"  Total Triggers: {yunus.trigger_count}")
    
    print("\n⚖️  MA'AT PROTOCOL")
    print(f"  Total Principles: {len(maat.principles)}")
    print(f"  Violation Patterns: {len(maat.violation_patterns)}")
    print(f"  Violation History: {len(maat.violation_history)}")
    
    print("\n" + "=" * 70)

def recommend_tuning():
    """Provide tuning recommendations based on test results"""
    print("\n📊 TUNING RECOMMENDATIONS (Based on Test Results)")
    print("=" * 70)
    
    print("\n1️⃣  YUNUS THRESHOLD ADJUSTMENT")
    print("   Current: 0.6")
    print("   Recommended: 0.5")
    print("   Reason: More sensitive to false authority patterns")
    print("   Impact: Better detection of blind trust demands")
    
    print("\n2️⃣  MA'AT CONFIDENCE PATTERN REFINEMENT")
    print("   Issue: 'quite confident' triggering false positive")
    print("   Fix: Distinguish between:")
    print("     ✅ ALLOW: 'quite confident', 'fairly certain'")
    print("     ❌ FLAG: '100% certain', 'absolutely guaranteed'")
    
    print("\n3️⃣  YUNUS FOCUS CLARIFICATION")
    print("   Current: Activating on harm keywords")
    print("   Recommended: Focus on false light only")
    print("   Distinction:")
    print("     🐋 YUNUS: God-complex, absolutism, false authority")
    print("     ⚖️  MA'AT: Harm, deception, capability lies")
    
    print("\n" + "=" * 70)

def apply_recommended_tuning():
    """Apply recommended tuning (requires editing source files)"""
    print("\n🔧 TO APPLY TUNING:")
    print("=" * 70)
    
    print("\n1. Edit: luminark_omega/protocols/yunus.py")
    print("   Line ~15: self.activation_threshold = 0.6")
    print("   Change to: self.activation_threshold = 0.5")
    
    print("\n2. Edit: luminark_omega/protocols/maat.py")
    print("   Add to violation_patterns:")
    print("   'excessive_certainty': {")
    print("     'keywords': ['100%', 'absolutely', 'guaranteed', 'definitely'],")
    print("     'exclude': ['quite', 'fairly', 'reasonably']")
    print("   }")
    
    print("\n3. Edit: luminark_omega/protocols/yunus.py")
    print("   Refine false_light_patterns to exclude pure harm")
    print("   (Focus on god-complex, not destruction)")
    
    print("\n4. Re-run tests:")
    print("   run_safety_tests.bat")
    
    print("\n" + "=" * 70)

def main():
    print("\n🎯 LUMINARK Safety Protocol Tuning Utility\n")
    
    show_current_settings()
    recommend_tuning()
    apply_recommended_tuning()
    
    print("\n💡 TIP: After tuning, expect:")
    print("   - Ma'at accuracy: 91.7% → 95%+")
    print("   - Yunus accuracy: 58.3% → 85%+")
    print("   - Overall system: Production-ready!")
    print("\n")

if __name__ == "__main__":
    main()
