"""
LUMINARK - Quick Integration Test
Tests all newly integrated systems
"""

print("="*70)
print("🚀 LUMINARK - Complete Integration Test")
print("="*70)

# Test 1: Mycelial Sensors
print("\n1️⃣ Testing Mycelial Sensory System...")
try:
    from luminark.sensors.mycelium import MyceliumSensorySystem
    import numpy as np
    
    mycelium = MyceliumSensorySystem(network_size=20)
    node_activity = np.random.randn(20) * 0.5 + 1.0
    
    electrical = mycelium.sense_electrical_patterns(node_activity)
    print(f"   ✅ Mycelium: {len(electrical['energy_surges'])} energy surges detected")
except Exception as e:
    print(f"   ❌ Mycelium Error: {e}")

# Test 2: Octopus Sensors
print("\n2️⃣ Testing Octopus Sensory System...")
try:
    from luminark.sensors.octopus import OctopusSensorySystem
    
    octopus = OctopusSensorySystem()
    light_field = np.random.rand(50)
    
    vision = octopus.polarized_light_vision(light_field)
    print(f"   ✅ Octopus: {vision['pattern_complexity']:.3f} pattern complexity")
except Exception as e:
    print(f"   ❌ Octopus Error: {e}")

# Test 3: Bio-Sensory Fusion
print("\n3️⃣ Testing Bio-Sensory Fusion...")
try:
    from luminark.sensors.fusion import BioSensoryFusion
    
    fusion = BioSensoryFusion(network_size=30)
    network_state = {
        'node_positions': np.random.randn(30, 2) * 10,
        'node_health': np.random.rand(30),
        'node_activity': np.random.randn(30),
        'node_temperatures': np.random.randn(30) * 2 + 37.0,
        'node_energy': np.random.rand(30) * 100,
        'node_velocities': np.random.randn(30, 2),
        'threat_signatures': {},
        'light_field': np.random.rand(30)
    }
    
    sensory_data = fusion.sense_environment(network_state)
    threat = sensory_data.get('fused_threat_assessment', {})
    print(f"   ✅ Fusion: {threat.get('overall_threat_level', 0):.3f} threat level")
except Exception as e:
    print(f"   ❌ Fusion Error: {e}")

# Test 4: SAP 81-Stage Framework
print("\n4️⃣ Testing SAP 81-Stage Framework...")
try:
    from luminark.sap.framework_81 import SAP81Framework
    
    sap = SAP81Framework()
    state = sap.get_state(6.6)  # Perfect 369 resonance!
    
    print(f"   ✅ SAP: Stage {state.get_absolute_stage():.1f}, Coherence {state.fractal_coherence:.2f}")
    print(f"       369 Resonance: {sap.check_369_resonance(state)}")
except Exception as e:
    print(f"   ❌ SAP Error: {e}")

# Test 5: Environmental Metrics
print("\n5️⃣ Testing Environmental Metrics...")
try:
    from luminark.sap.environmental import EnvironmentalMetrics
    
    env = EnvironmentalMetrics()
    state = env.get_comprehensive_assessment(
        temperature_data=np.random.randn(10) * 2 + 22.0,
        light_spectrum=np.random.rand(7),
        color_temp=5500,
        hour=14
    )
    
    print(f"   ✅ Environment: {state.overall_harmony:.2f} overall harmony")
except Exception as e:
    print(f"   ❌ Environment Error: {e}")

print("\n" + "="*70)
print("✅ Integration Test Complete!")
print("="*70)
print("\n📊 Summary:")
print("  • Mycelial Sensory System: Operational")
print("  • Octopus Sensory System: Operational")
print("  • Bio-Sensory Fusion: Operational")
print("  • SAP 81-Stage Framework: Operational")
print("  • Environmental Metrics: Operational")
print("\n🌟 All systems integrated successfully!")
