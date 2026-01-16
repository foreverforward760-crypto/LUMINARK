#!/usr/bin/env python3
"""
LUMINARK Demo - Simulated AI/ML Visualization
"""
import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def generate_sample_data():
    """Generate sample data for visualization"""
    return {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'accuracy': random.uniform(0.85, 0.99),
        'loss': random.uniform(0.01, 0.15),
        'throughput': random.uniform(100, 500),
    }


def run_basic_demo(duration):
    """Run basic demo mode with real-time metrics"""
    print("=" * 60)
    print("LUMINARK - Basic Demo Mode")
    print("=" * 60)
    print(f"Duration: {duration} seconds")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    start_time = time.time()
    iteration = 0

    # Storage for plotting
    timestamps = []
    accuracies = []
    losses = []

    try:
        while (time.time() - start_time) < duration:
            iteration += 1
            data = generate_sample_data()

            # Store data for plotting
            timestamps.append(iteration)
            accuracies.append(data['accuracy'])
            losses.append(data['loss'])

            # Display metrics
            print(f"[{data['timestamp']}] Iteration {iteration:3d} | "
                  f"Accuracy: {data['accuracy']:.4f} | "
                  f"Loss: {data['loss']:.4f} | "
                  f"Throughput: {data['throughput']:.1f} ops/s")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Demo completed!")
    print(f"Total iterations: {iteration}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Average accuracy: {np.mean(accuracies):.4f}")
    print(f"Average loss: {np.mean(losses):.4f}")
    print("=" * 60)

    # Generate visualization
    if len(timestamps) > 0:
        print("\nGenerating visualization...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Accuracy plot
        ax1.plot(timestamps, accuracies, 'b-', linewidth=2, label='Accuracy')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('LUMINARK - Model Accuracy Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Loss plot
        ax2.plot(timestamps, losses, 'r-', linewidth=2, label='Loss')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('LUMINARK - Loss Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Save the plot
        output_file = 'demo_results.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")

        # Try to show plot (won't work in headless environment but won't error)
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except:
            pass


def run_advanced_demo(duration):
    """Run advanced demo mode with additional features"""
    print("=" * 60)
    print("LUMINARK - Advanced Demo Mode")
    print("=" * 60)
    print("Advanced features: Multi-task learning, adaptive optimization")
    print(f"Duration: {duration} seconds")
    print("=" * 60)
    print()

    # Placeholder for advanced features
    print("Advanced mode coming soon!")
    print("Running basic mode instead...")
    print()
    run_basic_demo(duration)


def main():
    parser = argparse.ArgumentParser(
        description='LUMINARK Demo - AI/ML Visualization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python octo_demo.py --mode basic --duration 30
  python octo_demo.py --mode advanced --duration 60
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['basic', 'advanced'],
        default='basic',
        help='Demo mode to run (default: basic)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Duration of demo in seconds (default: 30)'
    )

    args = parser.parse_args()

    if args.mode == 'basic':
        run_basic_demo(args.duration)
    elif args.mode == 'advanced':
        run_advanced_demo(args.duration)


if __name__ == '__main__':
    main()
