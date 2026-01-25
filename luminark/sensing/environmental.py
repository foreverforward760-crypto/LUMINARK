"""
Environmental Metrics - Contextual Awareness System
Tracks environmental conditions, system state, and contextual factors
"""
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class EnvironmentalDomain(Enum):
    """Different environmental domains to monitor"""
    COMPUTATIONAL = 0      # System resources, performance
    ENERGETIC = 1          # Energy levels, flows
    INFORMATIONAL = 2      # Data quality, entropy
    TEMPORAL = 3           # Time-based patterns
    SPATIAL = 4            # Spatial distribution
    SOCIAL = 5             # Multi-agent interactions
    QUANTUM = 6            # Quantum field states
    BIOLOGICAL = 7         # Bio-inspired metrics


@dataclass
class EnvironmentalSnapshot:
    """Complete environmental state at a point in time"""
    timestamp: float
    domains: Dict[EnvironmentalDomain, Dict[str, float]]
    overall_health: float
    anomalies: List[str] = field(default_factory=list)
    trends: Dict[str, str] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class EnvironmentalMetrics:
    """
    Comprehensive environmental monitoring system
    Tracks multiple domains of environmental factors
    """

    def __init__(self):
        self.history = []
        self.max_history = 10000

        # Baseline metrics (learned over time)
        self.baselines = {}

        # Alert thresholds
        self.thresholds = {
            'cpu_usage': 0.9,
            'memory_usage': 0.85,
            'entropy': 0.95,
            'energy_depletion': 0.2,
            'anomaly_score': 0.7
        }

        print("🌍 Environmental Metrics initialized")
        print("   Monitoring domains: 8")
        print("   History capacity: 10000 snapshots")

    def measure_environment(self, external_inputs: Optional[Dict] = None) -> EnvironmentalSnapshot:
        """
        Measure complete environmental state

        Args:
            external_inputs: Optional external metrics to incorporate

        Returns:
            EnvironmentalSnapshot with all domain metrics
        """

        timestamp = time.time()
        external_inputs = external_inputs or {}

        # Measure each domain
        domains = {
            EnvironmentalDomain.COMPUTATIONAL: self._measure_computational(external_inputs),
            EnvironmentalDomain.ENERGETIC: self._measure_energetic(external_inputs),
            EnvironmentalDomain.INFORMATIONAL: self._measure_informational(external_inputs),
            EnvironmentalDomain.TEMPORAL: self._measure_temporal(external_inputs),
            EnvironmentalDomain.SPATIAL: self._measure_spatial(external_inputs),
            EnvironmentalDomain.SOCIAL: self._measure_social(external_inputs),
            EnvironmentalDomain.QUANTUM: self._measure_quantum(external_inputs),
            EnvironmentalDomain.BIOLOGICAL: self._measure_biological(external_inputs)
        }

        # Calculate overall health
        overall_health = self._calculate_overall_health(domains)

        # Detect anomalies
        anomalies = self._detect_anomalies(domains, external_inputs)

        # Identify trends
        trends = self._identify_trends(domains)

        # Generate recommendations
        recommendations = self._generate_recommendations(domains, anomalies, overall_health)

        # Create snapshot
        snapshot = EnvironmentalSnapshot(
            timestamp=timestamp,
            domains=domains,
            overall_health=overall_health,
            anomalies=anomalies,
            trends=trends,
            recommendations=recommendations
        )

        # Store in history
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Update baselines
        self._update_baselines(domains)

        return snapshot

    def _measure_computational(self, inputs: Dict) -> Dict[str, float]:
        """Measure computational environment"""
        # In production, would use psutil or similar
        # Here we use mock/simulated values

        return {
            'cpu_usage': inputs.get('cpu_usage', np.random.uniform(0.3, 0.7)),
            'memory_usage': inputs.get('memory_usage', np.random.uniform(0.4, 0.6)),
            'disk_io': inputs.get('disk_io', np.random.uniform(0.2, 0.5)),
            'network_throughput': inputs.get('network_throughput', np.random.uniform(0.3, 0.7)),
            'process_count': inputs.get('process_count', np.random.uniform(0.2, 0.4)),
            'thread_efficiency': inputs.get('thread_efficiency', np.random.uniform(0.6, 0.9))
        }

    def _measure_energetic(self, inputs: Dict) -> Dict[str, float]:
        """Measure energetic environment (power, heat, etc.)"""
        return {
            'energy_level': inputs.get('energy_level', np.random.uniform(0.5, 0.9)),
            'power_consumption': inputs.get('power_consumption', np.random.uniform(0.3, 0.6)),
            'thermal_load': inputs.get('thermal_load', np.random.uniform(0.3, 0.7)),
            'energy_efficiency': inputs.get('energy_efficiency', np.random.uniform(0.6, 0.9)),
            'energy_flow_rate': inputs.get('energy_flow_rate', np.random.uniform(0.4, 0.8)),
            'battery_health': inputs.get('battery_health', np.random.uniform(0.7, 1.0))
        }

    def _measure_informational(self, inputs: Dict) -> Dict[str, float]:
        """Measure informational environment (data quality, entropy)"""

        # Calculate entropy if data provided
        if 'data_stream' in inputs:
            data = np.array(inputs['data_stream'])
            if data.size > 0:
                # Normalized entropy
                unique, counts = np.unique(data, return_counts=True)
                probabilities = counts / counts.sum()
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                max_entropy = np.log2(len(unique)) if len(unique) > 0 else 1
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            else:
                normalized_entropy = 0.5
        else:
            normalized_entropy = inputs.get('entropy', np.random.uniform(0.4, 0.7))

        return {
            'entropy': normalized_entropy,
            'data_quality': inputs.get('data_quality', np.random.uniform(0.6, 0.9)),
            'signal_to_noise': inputs.get('signal_to_noise', np.random.uniform(0.5, 0.8)),
            'information_density': inputs.get('information_density', np.random.uniform(0.4, 0.7)),
            'coherence': inputs.get('coherence', np.random.uniform(0.5, 0.85)),
            'redundancy': inputs.get('redundancy', np.random.uniform(0.3, 0.6))
        }

    def _measure_temporal(self, inputs: Dict) -> Dict[str, float]:
        """Measure temporal environment (time-based patterns)"""

        # Analyze history for temporal patterns
        if len(self.history) > 10:
            recent = self.history[-10:]

            # Calculate rate of change
            timestamps = [s.timestamp for s in recent]
            healths = [s.overall_health for s in recent]

            time_variance = np.var(np.diff(timestamps)) if len(timestamps) > 1 else 0
            health_variance = np.var(healths) if len(healths) > 1 else 0

            temporal_stability = 1.0 / (1.0 + time_variance + health_variance)
        else:
            temporal_stability = 0.5

        return {
            'temporal_stability': temporal_stability,
            'rhythm_coherence': inputs.get('rhythm_coherence', np.random.uniform(0.5, 0.8)),
            'cycle_completion': inputs.get('cycle_completion', np.random.uniform(0.3, 0.9)),
            'time_dilation_factor': inputs.get('time_dilation', 1.0),
            'chronological_health': inputs.get('chronological_health', np.random.uniform(0.6, 0.9)),
            'temporal_flow': inputs.get('temporal_flow', np.random.uniform(0.4, 0.8))
        }

    def _measure_spatial(self, inputs: Dict) -> Dict[str, float]:
        """Measure spatial environment (distribution, locality)"""
        return {
            'spatial_distribution': inputs.get('spatial_distribution', np.random.uniform(0.4, 0.7)),
            'locality_coherence': inputs.get('locality_coherence', np.random.uniform(0.5, 0.8)),
            'dimensional_alignment': inputs.get('dimensional_alignment', np.random.uniform(0.6, 0.9)),
            'boundary_integrity': inputs.get('boundary_integrity', np.random.uniform(0.7, 0.95)),
            'spatial_density': inputs.get('spatial_density', np.random.uniform(0.3, 0.7)),
            'geometric_harmony': inputs.get('geometric_harmony', np.random.uniform(0.5, 0.85))
        }

    def _measure_social(self, inputs: Dict) -> Dict[str, float]:
        """Measure social environment (multi-agent if applicable)"""
        return {
            'agent_connectivity': inputs.get('agent_connectivity', np.random.uniform(0.3, 0.7)),
            'collaboration_index': inputs.get('collaboration_index', np.random.uniform(0.4, 0.8)),
            'consensus_strength': inputs.get('consensus_strength', np.random.uniform(0.5, 0.85)),
            'network_health': inputs.get('network_health', np.random.uniform(0.6, 0.9)),
            'communication_efficiency': inputs.get('communication_efficiency', np.random.uniform(0.5, 0.8)),
            'social_coherence': inputs.get('social_coherence', np.random.uniform(0.4, 0.75))
        }

    def _measure_quantum(self, inputs: Dict) -> Dict[str, float]:
        """Measure quantum environment (field states, coherence)"""
        return {
            'quantum_coherence': inputs.get('quantum_coherence', np.random.uniform(0.3, 0.7)),
            'entanglement_strength': inputs.get('entanglement_strength', np.random.uniform(0.2, 0.6)),
            'superposition_state': inputs.get('superposition_state', np.random.uniform(0.4, 0.8)),
            'decoherence_rate': inputs.get('decoherence_rate', np.random.uniform(0.1, 0.4)),
            'quantum_noise': inputs.get('quantum_noise', np.random.uniform(0.2, 0.5)),
            'wave_function_integrity': inputs.get('wave_function_integrity', np.random.uniform(0.6, 0.95))
        }

    def _measure_biological(self, inputs: Dict) -> Dict[str, float]:
        """Measure biological-inspired metrics (adaptation, homeostasis)"""

        # Calculate adaptation rate from history
        if len(self.history) > 5:
            recent_healths = [s.overall_health for s in self.history[-5:]]
            adaptation_rate = 1.0 - np.std(recent_healths) if len(recent_healths) > 1 else 0.5
        else:
            adaptation_rate = 0.5

        return {
            'homeostasis_level': inputs.get('homeostasis', np.random.uniform(0.6, 0.9)),
            'adaptation_rate': adaptation_rate,
            'vitality_index': inputs.get('vitality', np.random.uniform(0.5, 0.85)),
            'resilience_score': inputs.get('resilience', np.random.uniform(0.6, 0.9)),
            'growth_potential': inputs.get('growth_potential', np.random.uniform(0.4, 0.8)),
            'systemic_balance': inputs.get('systemic_balance', np.random.uniform(0.5, 0.85))
        }

    def _calculate_overall_health(self, domains: Dict[EnvironmentalDomain, Dict[str, float]]) -> float:
        """Calculate overall environmental health score"""

        # Weight each domain equally for simplicity
        # In practice, weights could be adaptive

        domain_healths = []

        for domain, metrics in domains.items():
            # Average metrics in this domain
            domain_health = np.mean(list(metrics.values()))
            domain_healths.append(domain_health)

        overall = np.mean(domain_healths)

        return float(overall)

    def _detect_anomalies(self, domains: Dict, inputs: Dict) -> List[str]:
        """Detect environmental anomalies"""

        anomalies = []

        # Check computational domain
        comp = domains[EnvironmentalDomain.COMPUTATIONAL]
        if comp['cpu_usage'] > self.thresholds['cpu_usage']:
            anomalies.append(f"HIGH_CPU: {comp['cpu_usage']:.2f}")
        if comp['memory_usage'] > self.thresholds['memory_usage']:
            anomalies.append(f"HIGH_MEMORY: {comp['memory_usage']:.2f}")

        # Check energetic domain
        energetic = domains[EnvironmentalDomain.ENERGETIC]
        if energetic['energy_level'] < self.thresholds['energy_depletion']:
            anomalies.append(f"LOW_ENERGY: {energetic['energy_level']:.2f}")

        # Check informational domain
        info = domains[EnvironmentalDomain.INFORMATIONAL]
        if info['entropy'] > self.thresholds['entropy']:
            anomalies.append(f"HIGH_ENTROPY: {info['entropy']:.2f}")

        # Check for rapid changes
        if len(self.history) > 2:
            last_health = self.history[-1].overall_health
            current_health = self._calculate_overall_health(domains)

            if abs(current_health - last_health) > 0.3:
                anomalies.append(f"RAPID_CHANGE: Δ{current_health - last_health:.2f}")

        return anomalies

    def _identify_trends(self, domains: Dict) -> Dict[str, str]:
        """Identify trends from historical data"""

        trends = {}

        if len(self.history) < 5:
            return trends

        # Analyze health trend
        recent_healths = [s.overall_health for s in self.history[-10:]]
        health_slope = np.polyfit(range(len(recent_healths)), recent_healths, 1)[0]

        if health_slope > 0.01:
            trends['overall_health'] = 'improving'
        elif health_slope < -0.01:
            trends['overall_health'] = 'declining'
        else:
            trends['overall_health'] = 'stable'

        # Analyze entropy trend (informational domain)
        recent_snapshots = self.history[-10:]
        entropies = [s.domains[EnvironmentalDomain.INFORMATIONAL]['entropy']
                    for s in recent_snapshots]
        entropy_slope = np.polyfit(range(len(entropies)), entropies, 1)[0]

        if entropy_slope > 0.01:
            trends['entropy'] = 'increasing'
        elif entropy_slope < -0.01:
            trends['entropy'] = 'decreasing'
        else:
            trends['entropy'] = 'stable'

        return trends

    def _generate_recommendations(self, domains: Dict, anomalies: List[str],
                                  overall_health: float) -> List[str]:
        """Generate environmental recommendations"""

        recommendations = []

        # Based on overall health
        if overall_health < 0.4:
            recommendations.append("CRITICAL: System health critically low - immediate intervention needed")
        elif overall_health < 0.6:
            recommendations.append("WARNING: System health below optimal - review resource allocation")

        # Based on specific domains
        comp = domains[EnvironmentalDomain.COMPUTATIONAL]
        if comp['cpu_usage'] > 0.85:
            recommendations.append("OPTIMIZE: High CPU usage - consider load balancing")

        energetic = domains[EnvironmentalDomain.ENERGETIC]
        if energetic['energy_level'] < 0.3:
            recommendations.append("RECHARGE: Low energy - reduce activity or increase resources")

        info = domains[EnvironmentalDomain.INFORMATIONAL]
        if info['entropy'] > 0.85:
            recommendations.append("STRUCTURE: High entropy - implement organization/filtering")

        # Based on trends
        if len(self.history) > 5:
            recent = [s.overall_health for s in self.history[-5:]]
            if all(recent[i] < recent[i-1] for i in range(1, len(recent))):
                recommendations.append("ALERT: Consistent decline detected - investigate root cause")

        return recommendations

    def _update_baselines(self, domains: Dict):
        """Update baseline metrics for anomaly detection"""

        for domain, metrics in domains.items():
            domain_key = domain.name

            if domain_key not in self.baselines:
                self.baselines[domain_key] = {}

            for metric_name, value in metrics.items():
                if metric_name not in self.baselines[domain_key]:
                    self.baselines[domain_key][metric_name] = {
                        'mean': value,
                        'std': 0,
                        'min': value,
                        'max': value,
                        'count': 1
                    }
                else:
                    baseline = self.baselines[domain_key][metric_name]

                    # Update running statistics
                    count = baseline['count'] + 1
                    old_mean = baseline['mean']
                    new_mean = old_mean + (value - old_mean) / count

                    baseline['mean'] = new_mean
                    baseline['count'] = count
                    baseline['min'] = min(baseline['min'], value)
                    baseline['max'] = max(baseline['max'], value)

                    # Update std (simplified)
                    if count > 1:
                        baseline['std'] = np.sqrt(
                            (baseline['std']**2 * (count-2) + (value - new_mean)**2) / (count-1)
                        ) if count > 2 else abs(value - old_mean)

    def get_environmental_summary(self) -> Dict[str, Any]:
        """Get summary of environmental metrics"""

        if not self.history:
            return {'status': 'no_data'}

        latest = self.history[-1]

        return {
            'current_health': latest.overall_health,
            'active_anomalies': len(latest.anomalies),
            'anomaly_list': latest.anomalies,
            'trends': latest.trends,
            'recommendations': latest.recommendations,
            'history_size': len(self.history),
            'baselines_learned': {domain: len(metrics)
                                 for domain, metrics in self.baselines.items()},
            'status': 'healthy' if latest.overall_health > 0.7 else
                     'warning' if latest.overall_health > 0.4 else 'critical'
        }


if __name__ == '__main__':
    # Demo
    print("🌍 Environmental Metrics Demo\n")

    env = EnvironmentalMetrics()

    # Simulate environmental measurements
    print("📊 Taking environmental snapshots...\n")

    for i in range(5):
        # Simulate some variation
        inputs = {
            'cpu_usage': 0.5 + 0.1 * np.sin(i),
            'memory_usage': 0.6 + 0.05 * np.cos(i),
            'energy_level': 0.8 - 0.1 * i / 5,
            'entropy': 0.5 + 0.1 * i / 5,
            'data_stream': np.random.randint(0, 10, 100)
        }

        snapshot = env.measure_environment(inputs)

        print(f"Snapshot {i+1}:")
        print(f"   Overall Health: {snapshot.overall_health:.3f}")
        print(f"   Anomalies: {len(snapshot.anomalies)}")
        for anomaly in snapshot.anomalies:
            print(f"      - {anomaly}")

        if snapshot.trends:
            print(f"   Trends:")
            for key, trend in snapshot.trends.items():
                print(f"      {key}: {trend}")

        if snapshot.recommendations:
            print(f"   Recommendations:")
            for rec in snapshot.recommendations[:2]:
                print(f"      - {rec}")

        print()

    # Summary
    print("📋 Environmental Summary:")
    summary = env.get_environmental_summary()
    for key, val in summary.items():
        if key not in ['anomaly_list', 'trends', 'recommendations']:
            print(f"   {key}: {val}")
