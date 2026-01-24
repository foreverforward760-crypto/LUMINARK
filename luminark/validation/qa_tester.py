"""
Automated QA & Pressure Testing System
Inspired by DeepAgent - validates AI outputs under adversarial conditions

Features:
- Adversarial input generation
- Edge case probing
- Robustness validation
- Performance regression detection
- Integration with SAR stage monitoring
"""
import numpy as np
from typing import Dict, List, Any, Callable
import time
from collections import defaultdict


class AutomatedQATester:
    """
    Pressure-tests AI models with adversarial inputs and edge cases
    Ensures robust performance across SAR stages
    """

    def __init__(self, noise_levels=[0.05, 0.1, 0.2, 0.5]):
        """
        Args:
            noise_levels: Different noise intensities for adversarial testing
        """
        self.noise_levels = noise_levels
        self.test_history = []
        self.baseline_performance = None
        self.vulnerability_log = []

    def pressure_test(self, model, test_inputs: np.ndarray,
                     test_targets: np.ndarray, num_tests=50) -> Dict[str, Any]:
        """
        Run comprehensive pressure tests on model

        Tests include:
        1. Gaussian noise injection
        2. Input perturbation
        3. Boundary value testing
        4. Extreme value testing
        5. Consistency checks

        Args:
            model: Model to test (must have forward method)
            test_inputs: Clean test inputs
            test_targets: Expected targets
            num_tests: Number of test iterations per noise level

        Returns:
            Dict with test results and vulnerabilities detected
        """
        from luminark.core import Tensor
        from luminark.nn import MSELoss

        criterion = MSELoss()
        results = {
            'noise_level_results': {},
            'vulnerabilities': [],
            'overall_status': 'unknown',
            'degradation_score': 0.0
        }

        # Get baseline performance
        clean_input = Tensor(test_inputs, requires_grad=False)
        clean_output = model(clean_input)
        clean_target = Tensor(test_targets, requires_grad=False)
        baseline_loss = criterion(clean_output, clean_target).data

        if self.baseline_performance is None:
            self.baseline_performance = baseline_loss

        results['baseline_loss'] = float(baseline_loss)

        # Test each noise level
        for noise_level in self.noise_levels:
            noise_losses = []

            for _ in range(num_tests):
                # Add Gaussian noise
                noise = np.random.randn(*test_inputs.shape) * noise_level
                noisy_inputs = test_inputs + noise.astype(test_inputs.dtype)

                # Forward pass
                noisy_tensor = Tensor(noisy_inputs, requires_grad=False)
                noisy_output = model(noisy_tensor)
                loss = criterion(noisy_output, clean_target)

                noise_losses.append(loss.data)

            avg_noise_loss = np.mean(noise_losses)
            std_noise_loss = np.std(noise_losses)
            degradation = (avg_noise_loss - baseline_loss) / (baseline_loss + 1e-10)

            results['noise_level_results'][noise_level] = {
                'avg_loss': float(avg_noise_loss),
                'std_loss': float(std_noise_loss),
                'degradation': float(degradation),
                'num_tests': num_tests
            }

            # Check for vulnerabilities
            if degradation > 0.5:  # 50% degradation threshold
                vulnerability = {
                    'type': 'high_noise_sensitivity',
                    'noise_level': noise_level,
                    'degradation': float(degradation),
                    'severity': 'HIGH' if degradation > 1.0 else 'MEDIUM'
                }
                results['vulnerabilities'].append(vulnerability)
                self.vulnerability_log.append(vulnerability)

        # Calculate overall degradation score
        avg_degradation = np.mean([
            res['degradation']
            for res in results['noise_level_results'].values()
        ])
        results['degradation_score'] = float(avg_degradation)

        # Determine overall status
        if avg_degradation < 0.3:
            results['overall_status'] = 'ROBUST'
        elif avg_degradation < 0.7:
            results['overall_status'] = 'WARNING'
        else:
            results['overall_status'] = 'VULNERABLE'

        # Record test
        self.test_history.append({
            'timestamp': time.time(),
            'results': results
        })

        return results

    def boundary_value_test(self, model, input_shape: tuple,
                           input_range: tuple = (-10, 10)) -> Dict[str, Any]:
        """
        Test model at boundary values (min, max, zero)

        Args:
            model: Model to test
            input_shape: Shape of inputs
            input_range: Min/max values for inputs

        Returns:
            Dict with boundary test results
        """
        from luminark.core import Tensor

        test_cases = {
            'zero': np.zeros(input_shape, dtype=np.float32),
            'min': np.full(input_shape, input_range[0], dtype=np.float32),
            'max': np.full(input_shape, input_range[1], dtype=np.float32),
            'random_extreme': np.random.choice(
                [input_range[0], input_range[1]],
                size=input_shape
            ).astype(np.float32)
        }

        results = {}

        for case_name, test_input in test_cases.items():
            try:
                tensor_input = Tensor(test_input, requires_grad=False)
                output = model(tensor_input)

                # Check for NaN/Inf
                has_nan = np.isnan(output.data).any()
                has_inf = np.isinf(output.data).any()
                output_range = (float(output.data.min()), float(output.data.max()))

                results[case_name] = {
                    'success': True,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'output_range': output_range,
                    'stable': not (has_nan or has_inf)
                }

                if has_nan or has_inf:
                    self.vulnerability_log.append({
                        'type': 'numerical_instability',
                        'case': case_name,
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                        'severity': 'HIGH'
                    })

            except Exception as e:
                results[case_name] = {
                    'success': False,
                    'error': str(e),
                    'stable': False
                }

        # Overall boundary stability
        all_stable = all(r.get('stable', False) for r in results.values())
        results['overall_boundary_status'] = 'STABLE' if all_stable else 'UNSTABLE'

        return results

    def consistency_test(self, model, test_input: np.ndarray,
                        num_runs=10) -> Dict[str, Any]:
        """
        Test output consistency for same input (stochastic models)

        Args:
            model: Model to test
            test_input: Input to test
            num_runs: Number of repeated runs

        Returns:
            Dict with consistency metrics
        """
        from luminark.core import Tensor

        outputs = []
        tensor_input = Tensor(test_input, requires_grad=False)

        for _ in range(num_runs):
            output = model(tensor_input)
            outputs.append(output.data.copy())

        outputs = np.array(outputs)

        # Calculate variance across runs
        output_variance = np.var(outputs, axis=0).mean()
        output_std = np.std(outputs, axis=0).mean()

        # High variance might indicate instability
        is_consistent = output_variance < 0.1

        result = {
            'num_runs': num_runs,
            'output_variance': float(output_variance),
            'output_std': float(output_std),
            'is_consistent': is_consistent,
            'consistency_status': 'CONSISTENT' if is_consistent else 'INCONSISTENT'
        }

        if not is_consistent:
            self.vulnerability_log.append({
                'type': 'output_inconsistency',
                'variance': float(output_variance),
                'severity': 'MEDIUM'
            })

        return result

    def regression_test(self, model, reference_performance: float,
                       test_inputs: np.ndarray, test_targets: np.ndarray) -> Dict[str, Any]:
        """
        Check for performance regression against baseline

        Args:
            model: Current model
            reference_performance: Expected baseline performance
            test_inputs: Test inputs
            test_targets: Test targets

        Returns:
            Dict with regression test results
        """
        from luminark.core import Tensor
        from luminark.nn import MSELoss

        criterion = MSELoss()

        # Current performance
        input_tensor = Tensor(test_inputs, requires_grad=False)
        output = model(input_tensor)
        target_tensor = Tensor(test_targets, requires_grad=False)
        current_loss = criterion(output, target_tensor).data

        # Compare to reference
        regression = (current_loss - reference_performance) / (reference_performance + 1e-10)

        result = {
            'reference_performance': float(reference_performance),
            'current_performance': float(current_loss),
            'regression_percent': float(regression * 100),
            'status': 'IMPROVED' if regression < -0.05 else
                     'STABLE' if abs(regression) < 0.05 else
                     'REGRESSED'
        }

        if regression > 0.1:  # 10% regression
            self.vulnerability_log.append({
                'type': 'performance_regression',
                'regression_percent': float(regression * 100),
                'severity': 'HIGH' if regression > 0.3 else 'MEDIUM'
            })

        return result

    def comprehensive_qa_suite(self, model, test_data: Dict[str, np.ndarray],
                              reference_performance: float = None) -> Dict[str, Any]:
        """
        Run complete QA test suite

        Args:
            model: Model to test
            test_data: Dict with 'inputs' and 'targets'
            reference_performance: Optional baseline for regression testing

        Returns:
            Comprehensive QA report
        """
        print("\n" + "="*70)
        print("AUTOMATED QA SUITE - PRESSURE TESTING")
        print("="*70 + "\n")

        report = {
            'timestamp': time.time(),
            'tests_run': [],
            'overall_status': 'UNKNOWN',
            'critical_vulnerabilities': 0,
            'warnings': 0
        }

        # 1. Pressure test
        print("1️⃣  Running adversarial pressure tests...")
        pressure_results = self.pressure_test(
            model,
            test_data['inputs'],
            test_data['targets']
        )
        report['pressure_test'] = pressure_results
        report['tests_run'].append('pressure_test')
        print(f"   Status: {pressure_results['overall_status']}")
        print(f"   Degradation: {pressure_results['degradation_score']*100:.1f}%")

        # 2. Boundary test
        print("\n2️⃣  Running boundary value tests...")
        boundary_results = self.boundary_value_test(
            model,
            test_data['inputs'].shape
        )
        report['boundary_test'] = boundary_results
        report['tests_run'].append('boundary_test')
        print(f"   Status: {boundary_results['overall_boundary_status']}")

        # 3. Consistency test
        print("\n3️⃣  Running consistency tests...")
        consistency_results = self.consistency_test(
            model,
            test_data['inputs'][:1]  # Use first sample
        )
        report['consistency_test'] = consistency_results
        report['tests_run'].append('consistency_test')
        print(f"   Status: {consistency_results['consistency_status']}")

        # 4. Regression test (if baseline provided)
        if reference_performance is not None:
            print("\n4️⃣  Running regression tests...")
            regression_results = self.regression_test(
                model,
                reference_performance,
                test_data['inputs'],
                test_data['targets']
            )
            report['regression_test'] = regression_results
            report['tests_run'].append('regression_test')
            print(f"   Status: {regression_results['status']}")
            if regression_results['status'] == 'REGRESSED':
                print(f"   ⚠️  Regression: {regression_results['regression_percent']:.1f}%")

        # Count vulnerabilities
        for vuln in self.vulnerability_log[-10:]:  # Recent vulnerabilities
            if vuln.get('severity') == 'HIGH':
                report['critical_vulnerabilities'] += 1
            elif vuln.get('severity') == 'MEDIUM':
                report['warnings'] += 1

        # Determine overall status
        if report['critical_vulnerabilities'] > 0:
            report['overall_status'] = 'CRITICAL'
        elif report['warnings'] > 2:
            report['overall_status'] = 'WARNING'
        elif pressure_results['overall_status'] == 'ROBUST':
            report['overall_status'] = 'PASSED'
        else:
            report['overall_status'] = 'NEEDS_REVIEW'

        print("\n" + "="*70)
        print("QA SUITE COMPLETE")
        print("="*70)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Critical Issues: {report['critical_vulnerabilities']}")
        print(f"Warnings: {report['warnings']}")
        print("="*70 + "\n")

        return report

    def get_vulnerability_summary(self) -> Dict[str, Any]:
        """Get summary of all detected vulnerabilities"""
        if not self.vulnerability_log:
            return {
                'total_vulnerabilities': 0,
                'by_severity': {},
                'by_type': {}
            }

        from collections import Counter

        severities = Counter(v['severity'] for v in self.vulnerability_log)
        types = Counter(v['type'] for v in self.vulnerability_log)

        return {
            'total_vulnerabilities': len(self.vulnerability_log),
            'by_severity': dict(severities),
            'by_type': dict(types),
            'recent_vulnerabilities': self.vulnerability_log[-5:]
        }


__all__ = ['AutomatedQATester']
