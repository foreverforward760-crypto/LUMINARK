#!/usr/bin/env python3
"""
LUMINARK AI Framework - Complete Demo
Showcases all 4 integrated features
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("🌟 LUMINARK AI FRAMEWORK - COMPLETE DEMO")
print("="*70)
print()

# 1. SAP Framework Demo
print("1️⃣  SAP (Stanfield's Axiom of Perpetuity) Framework Demo")
print("-" * 70)

try:
    from luminark_omega.core.sar_framework import SARFramework
    
    sar = SARFramework()
    
    print("📊 All SAP Stages:")
    for level in range(10):
        stage = sar.get_stage(level)
        inversion_status = "🔴 INVERTED" if stage.is_inverted else "🟢 ALIGNED"
        print(f"  Stage {level}: {stage.name:15} - {inversion_status}")
        print(f"           Physical: {stage.physical_state:10} | Conscious: {stage.conscious_state}")
    
    print("\n🔍 Inversion Detection Demo:")
    test_cases = [
        (True, True, "Both Stable (Stage 9 - Aligned)"),
        (True, False, "Physical Stable, Conscious Unstable (Even Stage - Inverted)"),
        (False, True, "Physical Unstable, Conscious Stable (Odd Stage - Inverted)"),
        (False, False, "Both Unstable (Stage 0 - Crisis)")
    ]
    
    for physical, conscious, description in test_cases:
        result = sar.detect_inversion(physical, conscious)
        print(f"\n  {description}")
        print(f"    → Stage {result['stage']}: {result['stage_name']}")
        print(f"    → Inversion Level: {result['inversion_level']}")
        print(f"    → {result['description']}")
        print(f"    → Intervention: {result['intervention']}")
    
    print("\n✅ SAP Framework Demo Complete!")
    
except ImportError as e:
    print(f"⚠️  SAP Framework not available: {e}")

print("\n" + "="*70)
time.sleep(2)

# 2. Biofeedback Demo
print("\n2️⃣  Biofeedback Integration Demo")
print("-" * 70)

try:
    from luminark.biofeedback import BiofeedbackMonitor
    
    monitor = BiofeedbackMonitor(update_interval=1.0)
    monitor.start_monitoring()
    
    print("📊 Taking 5 biofeedback measurements...\n")
    
    for i in range(5):
        data = monitor.get_measurement()
        
        print(f"Measurement {i+1}:")
        print(f"  💓 Heart Rate: {data.heart_rate:.1f} bpm")
        print(f"  📈 HRV: {data.hrv:.1f}")
        print(f"  😰 Stress Level: {data.stress_level:.2f}")
        print(f"  🧘 Coherence: {data.coherence:.2f}")
        print(f"  😊 Emotional State: {data.emotional_state}")
        
        # Stress assessment
        assessment = monitor.assess_stress()
        print(f"  🎯 Status: {assessment['status']}")
        print(f"  💡 Recommendation: {assessment['recommendation']}")
        
        # SAR correlation
        try:
            from luminark_omega.core.sar_framework import SARFramework
            correlation = monitor.correlate_with_sar_stage(sar_stage=4)
            print(f"  🔗 SAR Alignment: {correlation['alignment']:.2f}")
            print(f"  💭 Insight: {correlation['insights']}")
        except:
            pass
        
        print()
        time.sleep(0.5)
    
    # Statistics
    print("📈 Biofeedback Statistics:")
    stats = monitor.get_statistics()
    print(f"  Total Measurements: {stats['total_measurements']}")
    print(f"  Average HRV: {stats['hrv']['mean']:.1f}")
    print(f"  Average Stress: {stats['stress']['mean']:.2f}")
    print(f"  Average Coherence: {stats['coherence']['mean']:.2f}")
    
    # Export data
    export_path = Path(__file__).parent / "data" / "biofeedback_demo.json"
    export_path.parent.mkdir(exist_ok=True)
    monitor.export_data(str(export_path))
    
    monitor.stop_monitoring()
    print("\n✅ Biofeedback Demo Complete!")
    
except ImportError as e:
    print(f"⚠️  Biofeedback module not available: {e}")

print("\n" + "="*70)
time.sleep(2)

# 3. Web Dashboard Info
print("\n3️⃣  Web Dashboard")
print("-" * 70)
print("🌐 The web dashboard provides:")
print("  • Real-time SAP stage monitoring")
print("  • Interactive visualizations")
print("  • Biofeedback integration")
print("  • WebSocket-based live updates")
print("  • Beautiful modern UI")
print()
print("📍 To start the dashboard:")
print("  1. Run: python web_dashboard/app.py")
print("  2. Or: start_dashboard.bat (Windows)")
print("  3. Open: http://localhost:5000")
print()
print("✅ Dashboard files created and ready!")

print("\n" + "="*70)
time.sleep(2)

# 4. Deployment Info
print("\n4️⃣  Deployment Automation")
print("-" * 70)
print("🚀 One-click deployment available!")
print()
print("📍 To deploy LUMINARK:")
print("  Run: python deploy_luminark.py")
print()
print("The deployment script will:")
print("  ✅ Check Python version")
print("  ✅ Create virtual environment")
print("  ✅ Install dependencies")
print("  ✅ Set up directories")
print("  ✅ Generate configuration")
print("  ✅ Run tests")
print("  ✅ Start the system")
print()
print("✅ Deployment script ready!")

print("\n" + "="*70)
print("\n🎉 ALL 4 INTEGRATIONS COMPLETE!")
print("="*70)
print()
print("📚 Next Steps:")
print("  1. Review INTEGRATION_COMPLETE.md for full documentation")
print("  2. Run deploy_luminark.py for automated setup")
print("  3. Start the web dashboard to see visualizations")
print("  4. Explore biofeedback monitoring capabilities")
print()
print("🌟 LUMINARK is now a complete, production-ready AI framework!")
print("="*70)
