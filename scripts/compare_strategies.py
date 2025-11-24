"""
FedAvg vs FedAvg+ADM 성능 비교 시각화

사용법:
  python compare_strategies.py --baseline results_fedavg_3clients.json --adm results_fedavg_adm_3clients.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_results(filename):
    """Load experiment results from JSON"""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(baseline_results, adm_results, save_path='comparison_accuracy.png'):
    """Plot accuracy comparison between FedAvg and FedAvg+ADM"""

    baseline_acc = [acc * 100 for acc in baseline_results['accuracies']]
    adm_acc = [acc * 100 for acc in adm_results['accuracies']]

    rounds = range(1, len(baseline_acc) + 1)

    plt.figure(figsize=(12, 6))

    # Plot lines
    plt.plot(rounds, baseline_acc, marker='o', linewidth=2, markersize=6,
             label='FedAvg (Baseline)', color='#1f77b4')
    plt.plot(rounds, adm_acc, marker='s', linewidth=2, markersize=6,
             label='FedAvg + ADM (Proposed)', color='#ff7f0e')

    # Add final accuracy annotations
    plt.annotate(f'{baseline_acc[-1]:.2f}%',
                xy=(len(baseline_acc), baseline_acc[-1]),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, color='#1f77b4')
    plt.annotate(f'{adm_acc[-1]:.2f}%',
                xy=(len(adm_acc), adm_acc[-1]),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, color='#ff7f0e')

    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('FedAvg vs FedAvg+ADM: Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy comparison saved to {save_path}")

    plt.show()


def plot_v_n_evolution(adm_results, save_path='v_n_evolution.png'):
    """Plot v_n values evolution over rounds"""

    if 'v_n_history' not in adm_results:
        print("No v_n history found in ADM results")
        return

    v_n_history = adm_results['v_n_history']
    num_clients = len(v_n_history[0]) if v_n_history else 0

    if num_clients == 0:
        print("No v_n data available")
        return

    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, num_clients))

    for client_id in range(num_clients):
        v_n_values = [round_vn[client_id] for round_vn in v_n_history]
        rounds = range(1, len(v_n_values) + 1)

        plt.plot(rounds, v_n_values, marker='o', linewidth=2, markersize=5,
                label=f'Client {client_id}', color=colors[client_id])

    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Maximum (v_n=1.0)')
    plt.axhline(y=0.4, color='gray', linestyle='--', alpha=0.5, label='Minimum (Gamma=0.4)')

    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('v_n (Data Usage Ratio)', fontsize=12)
    plt.title('ADM: Client Data Usage (v_n) Evolution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0.35, 1.05])
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"v_n evolution plot saved to {save_path}")

    plt.show()


def print_summary(baseline_results, adm_results):
    """Print comparison summary"""

    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    baseline_acc = baseline_results['accuracies']
    adm_acc = adm_results['accuracies']

    print(f"\nStrategy: {baseline_results['strategy']}")
    print(f"  Final Accuracy: {baseline_acc[-1]*100:.2f}%")
    print(f"  Max Accuracy:   {max(baseline_acc)*100:.2f}%")
    print(f"  Avg Accuracy:   {np.mean(baseline_acc)*100:.2f}%")

    print(f"\nStrategy: {adm_results['strategy']}")
    print(f"  Final Accuracy: {adm_acc[-1]*100:.2f}%")
    print(f"  Max Accuracy:   {max(adm_acc)*100:.2f}%")
    print(f"  Avg Accuracy:   {np.mean(adm_acc)*100:.2f}%")

    # Improvement
    final_improvement = (adm_acc[-1] - baseline_acc[-1]) * 100
    avg_improvement = (np.mean(adm_acc) - np.mean(baseline_acc)) * 100

    print(f"\nImprovement (FedAvg+ADM vs FedAvg):")
    print(f"  Final Accuracy: {final_improvement:+.2f}%")
    print(f"  Average Accuracy: {avg_improvement:+.2f}%")

    # Convergence speed
    target_acc = 0.8  # 80% threshold
    baseline_rounds = next((i+1 for i, acc in enumerate(baseline_acc) if acc >= target_acc), len(baseline_acc))
    adm_rounds = next((i+1 for i, acc in enumerate(adm_acc) if acc >= target_acc), len(adm_acc))

    print(f"\nConvergence Speed (to {target_acc*100:.0f}% accuracy):")
    print(f"  FedAvg:     {baseline_rounds} rounds")
    print(f"  FedAvg+ADM: {adm_rounds} rounds")

    if adm_rounds < baseline_rounds:
        speedup = ((baseline_rounds - adm_rounds) / baseline_rounds) * 100
        print(f"  Speedup: {speedup:.1f}% faster")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Compare FedAvg vs FedAvg+ADM')
    parser.add_argument('--baseline', type=str, required=True,
                       help='Path to FedAvg baseline results JSON')
    parser.add_argument('--adm', type=str, required=True,
                       help='Path to FedAvg+ADM results JSON')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for plots')

    args = parser.parse_args()

    print("Loading results...")
    baseline_results = load_results(args.baseline)
    adm_results = load_results(args.adm)

    print(f"Baseline: {baseline_results['strategy']}")
    print(f"Proposed: {adm_results['strategy']}")

    # Print summary
    print_summary(baseline_results, adm_results)

    # Plot accuracy comparison
    plot_accuracy_comparison(
        baseline_results,
        adm_results,
        save_path=f'{args.output_dir}/comparison_accuracy.png'
    )

    # Plot v_n evolution (only for ADM)
    plot_v_n_evolution(
        adm_results,
        save_path=f'{args.output_dir}/v_n_evolution.png'
    )

    print("\nComparison completed!")


if __name__ == "__main__":
    main()
