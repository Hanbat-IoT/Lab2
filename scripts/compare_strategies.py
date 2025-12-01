"""
FedAvg vs FedAvg+ADM 성능 비교 시각화

사용법:
  python compare_strategies.py --baseline results_fedavg_mnist_3clients_20241202_120000.json --adm results_fedavg_adm_mnist_3clients_20241202_120500.json
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

    baseline_acc = [acc * 100 for acc in baseline_results.get('accuracies', [])]
    adm_acc = [acc * 100 for acc in adm_results.get('accuracies', [])]

    if not baseline_acc or not adm_acc:
        print("⚠️  Cannot plot accuracy comparison: missing accuracy data")
        print("   Make sure both experiments completed successfully")
        return

    # Handle different lengths by using separate round ranges
    baseline_rounds = range(1, len(baseline_acc) + 1)
    adm_rounds = range(1, len(adm_acc) + 1)

    plt.figure(figsize=(14, 7))

    # Plot lines with separate round ranges
    plt.plot(baseline_rounds, baseline_acc, marker='o', linewidth=2.5, markersize=8,
             label='FedAvg (Baseline)', color='#1f77b4', alpha=0.8)
    plt.plot(adm_rounds, adm_acc, marker='s', linewidth=2.5, markersize=8,
             label='FedAvg + ADM (Proposed)', color='#ff7f0e', alpha=0.8)

    # Add accuracy annotations for each round
    for i, (r, acc) in enumerate(zip(baseline_rounds, baseline_acc)):
        plt.annotate(f'{acc:.1f}%',
                    xy=(r, acc),
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontsize=9,
                    color='#1f77b4',
                    ha='center',
                    fontweight='bold')
    
    for i, (r, acc) in enumerate(zip(adm_rounds, adm_acc)):
        plt.annotate(f'{acc:.1f}%',
                    xy=(r, acc),
                    xytext=(0, -15),
                    textcoords='offset points',
                    fontsize=9,
                    color='#ff7f0e',
                    ha='center',
                    fontweight='bold')

    plt.xlabel('Communication Round', fontsize=13, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    
    # Add dataset info to title
    dataset = baseline_results.get('dataset', 'unknown').upper()
    plt.title(f'FedAvg vs FedAvg+ADM: Accuracy Comparison ({dataset})', fontsize=15, fontweight='bold')
    
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Accuracy comparison saved to {save_path}")

    plt.show()


def plot_training_time_comparison(baseline_results, adm_results, save_path='comparison_training_time.png'):
    """Plot training time comparison between FedAvg and FedAvg+ADM"""

    baseline_times = baseline_results.get('round_times', [])
    adm_times = adm_results.get('round_times', [])

    if not baseline_times or not adm_times:
        print("⚠️  Cannot plot training time comparison: missing time data")
        return

    # Calculate cumulative times
    baseline_cumulative = np.cumsum(baseline_times)
    adm_cumulative = np.cumsum(adm_times)

    baseline_rounds = range(1, len(baseline_times) + 1)
    adm_rounds = range(1, len(adm_times) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Round time comparison
    ax1.plot(baseline_rounds, baseline_times, marker='o', linewidth=2.5, markersize=8,
             label='FedAvg (Baseline)', color='#1f77b4', alpha=0.8)
    ax1.plot(adm_rounds, adm_times, marker='s', linewidth=2.5, markersize=8,
             label='FedAvg + ADM (Proposed)', color='#ff7f0e', alpha=0.8)

    ax1.set_xlabel('Communication Round', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Round Time (seconds)', fontsize=13, fontweight='bold')
    
    dataset = baseline_results.get('dataset', 'unknown').upper()
    ax1.set_title(f'Per-Round Training Time ({dataset})', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add average time annotations
    baseline_avg = np.mean(baseline_times)
    adm_avg = np.mean(adm_times)
    ax1.axhline(y=baseline_avg, color='#1f77b4', linestyle=':', alpha=0.5, linewidth=2)
    ax1.axhline(y=adm_avg, color='#ff7f0e', linestyle=':', alpha=0.5, linewidth=2)
    ax1.text(len(baseline_rounds)*0.7, baseline_avg*1.05, f'Avg: {baseline_avg:.1f}s', 
             color='#1f77b4', fontweight='bold', fontsize=10)
    ax1.text(len(adm_rounds)*0.7, adm_avg*1.05, f'Avg: {adm_avg:.1f}s', 
             color='#ff7f0e', fontweight='bold', fontsize=10)

    # Plot 2: Cumulative time comparison
    ax2.plot(baseline_rounds, baseline_cumulative, marker='o', linewidth=2.5, markersize=8,
             label='FedAvg (Baseline)', color='#1f77b4', alpha=0.8)
    ax2.plot(adm_rounds, adm_cumulative, marker='s', linewidth=2.5, markersize=8,
             label='FedAvg + ADM (Proposed)', color='#ff7f0e', alpha=0.8)

    ax2.set_xlabel('Communication Round', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Time (seconds)', fontsize=13, fontweight='bold')
    ax2.set_title(f'Cumulative Training Time ({dataset})', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Add time savings annotation
    if len(baseline_cumulative) > 0 and len(adm_cumulative) > 0:
        baseline_total = baseline_cumulative[-1]
        adm_total = adm_cumulative[-1]
        time_saved = baseline_total - adm_total
        savings_pct = (time_saved / baseline_total) * 100 if baseline_total > 0 else 0
        
        ax2.text(len(adm_rounds)*0.5, max(baseline_total, adm_total)*0.5,
                f'Time Saved: {time_saved:.1f}s ({savings_pct:.1f}%)',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=12, fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training time comparison saved to {save_path}")

    plt.show()


def plot_batch_size_evolution(bwa_results, save_path='batch_size_evolution.png'):
    """Plot batch size evolution for BWA strategy"""
    
    if 'batch_size_history' not in bwa_results:
        print("No batch size history found in BWA results")
        return
    
    batch_sizes = bwa_results['batch_size_history']
    
    if not batch_sizes:
        print("No batch size data available")
        return
    
    plt.figure(figsize=(14, 7))
    
    rounds = range(1, len(batch_sizes) + 1)
    
    plt.plot(rounds, batch_sizes, marker='o', linewidth=2.5, markersize=10,
            color='#2ca02c', alpha=0.8, label='BWA Batch Size')
    
    # Add batch size annotations
    for r, bs in zip(rounds, batch_sizes):
        plt.annotate(f'{bs}',
                    xy=(r, bs),
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontsize=10,
                    color='#2ca02c',
                    ha='center',
                    fontweight='bold')
    
    # Add average line
    avg_batch_size = np.mean(batch_sizes)
    plt.axhline(y=avg_batch_size, color='gray', linestyle='--', alpha=0.5, 
                linewidth=2, label=f'Average: {avg_batch_size:.1f}')
    
    plt.xlabel('Communication Round', fontsize=13, fontweight='bold')
    plt.ylabel('Batch Size', fontsize=13, fontweight='bold')
    
    dataset = bwa_results.get('dataset', 'unknown').upper()
    plt.title(f'BWA: Dynamic Batch Size Evolution ({dataset})', fontsize=15, fontweight='bold')
    
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Batch size evolution plot saved to {save_path}")
    
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

    plt.figure(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, num_clients))

    for client_id in range(num_clients):
        v_n_values = [round_vn[client_id] for round_vn in v_n_history]
        rounds = range(1, len(v_n_values) + 1)

        plt.plot(rounds, v_n_values, marker='o', linewidth=2.5, markersize=8,
                label=f'Client {client_id}', color=colors[client_id])
        
        # Add v_n value annotations for each round
        for r, v_n in zip(rounds, v_n_values):
            plt.annotate(f'{v_n:.3f}',
                        xy=(r, v_n),
                        xytext=(0, 10),
                        textcoords='offset points',
                        fontsize=9,
                        color=colors[client_id],
                        ha='center',
                        fontweight='bold')

    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Maximum (v_n=1.0)')
    plt.axhline(y=0.4, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Minimum (Gamma=0.4)')

    plt.xlabel('Communication Round', fontsize=13, fontweight='bold')
    plt.ylabel('v_n (Data Usage Ratio)', fontsize=13, fontweight='bold')
    
    # Add dataset info to title
    dataset = adm_results.get('dataset', 'unknown').upper()
    plt.title(f'ADM: Client Data Usage (v_n) Evolution ({dataset})', fontsize=15, fontweight='bold')
    
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([0.35, 1.05])
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 v_n evolution plot saved to {save_path}")

    plt.show()


def print_summary(baseline_results, adm_results):
    """Print comparison summary"""

    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    baseline_acc = baseline_results.get('accuracies', [])
    adm_acc = adm_results.get('accuracies', [])
    baseline_times = baseline_results.get('round_times', [])
    adm_times = adm_results.get('round_times', [])
    
    # Print dataset info
    baseline_dataset = baseline_results.get('dataset', 'unknown')
    adm_dataset = adm_results.get('dataset', 'unknown')
    baseline_clients = baseline_results.get('num_clients', 'unknown')
    adm_clients = adm_results.get('num_clients', 'unknown')

    # Check if accuracies are available
    if not baseline_acc:
        print(f"\nStrategy: {baseline_results['strategy']}")
        print(f"  Dataset: {baseline_dataset.upper()}")
        print(f"  Clients: {baseline_clients}")
        print("  ⚠️  No accuracy data available")
        print(f"  Rounds: {baseline_results.get('num_rounds', 0)}")
    else:
        print(f"\nStrategy: {baseline_results['strategy']}")
        print(f"  Dataset: {baseline_dataset.upper()}")
        print(f"  Clients: {baseline_clients}")
        print(f"  Rounds: {len(baseline_acc)}")
        print(f"  Final Accuracy: {baseline_acc[-1]*100:.2f}%")
        print(f"  Max Accuracy:   {max(baseline_acc)*100:.2f}%")
        print(f"  Avg Accuracy:   {np.mean(baseline_acc)*100:.2f}%")

    if not adm_acc:
        print(f"\nStrategy: {adm_results['strategy']}")
        print(f"  Dataset: {adm_dataset.upper()}")
        print(f"  Clients: {adm_clients}")
        print("  ⚠️  No accuracy data available")
        print(f"  Rounds: {adm_results.get('num_rounds', 0)}")
        
        # Show v_n history if available
        if 'v_n_history' in adm_results and adm_results['v_n_history']:
            print(f"\n  ✓ ADM v_n optimization recorded:")
            for i, v_n in enumerate(adm_results['v_n_history'][:3], 1):
                print(f"    Round {i}: v_n = {v_n}")
            if len(adm_results['v_n_history']) > 3:
                print(f"    ... ({len(adm_results['v_n_history'])} rounds total)")
    else:
        print(f"\nStrategy: {adm_results['strategy']}")
        print(f"  Dataset: {adm_dataset.upper()}")
        print(f"  Clients: {adm_clients}")
        print(f"  Rounds: {len(adm_acc)}")
        print(f"  Final Accuracy: {adm_acc[-1]*100:.2f}%")
        print(f"  Max Accuracy:   {max(adm_acc)*100:.2f}%")
        print(f"  Avg Accuracy:   {np.mean(adm_acc)*100:.2f}%")

    # Only compare if both have accuracies
    if baseline_acc and adm_acc:
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

    # Training time comparison
    if baseline_times and adm_times:
        baseline_total = sum(baseline_times)
        adm_total = sum(adm_times)
        baseline_avg = np.mean(baseline_times)
        adm_avg = np.mean(adm_times)
        
        print(f"\nTraining Time:")
        print(f"  FedAvg Total:     {baseline_total:.2f}s")
        print(f"  FedAvg+ADM Total: {adm_total:.2f}s")
        print(f"  Time Saved:       {baseline_total - adm_total:.2f}s ({((baseline_total - adm_total)/baseline_total*100):.1f}%)")
        print(f"\n  FedAvg Avg/Round:     {baseline_avg:.2f}s")
        print(f"  FedAvg+ADM Avg/Round: {adm_avg:.2f}s")
        print(f"  Speedup:              {(baseline_avg/adm_avg):.2f}x faster per round")

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

    # Plot accuracy comparison (if data available)
    if baseline_results.get('accuracies') and adm_results.get('accuracies'):
        # Generate filename with dataset info
        dataset = baseline_results.get('dataset', 'unknown')
        save_path = f'{args.output_dir}/comparison_accuracy_{dataset}.png'
        
        plot_accuracy_comparison(
            baseline_results,
            adm_results,
            save_path=save_path
        )
    else:
        print("\n⚠️  Skipping accuracy comparison plot (missing data)")

    # Plot training time comparison
    if baseline_results.get('round_times') and adm_results.get('round_times'):
        dataset = baseline_results.get('dataset', 'unknown')
        save_path = f'{args.output_dir}/comparison_training_time_{dataset}.png'
        
        plot_training_time_comparison(
            baseline_results,
            adm_results,
            save_path=save_path
        )
    else:
        print("\n⚠️  Skipping training time comparison plot (missing data)")

    # Plot v_n evolution (only for ADM)
    if adm_results.get('v_n_history'):
        # Generate filename with dataset info
        dataset = adm_results.get('dataset', 'unknown')
        save_path = f'{args.output_dir}/v_n_evolution_{dataset}.png'
        
        plot_v_n_evolution(
            adm_results,
            save_path=save_path
        )
    else:
        print("\n⚠️  Skipping v_n evolution plot (no v_n history)")
    
    # Plot batch size evolution (only for BWA)
    if adm_results.get('batch_size_history'):
        dataset = adm_results.get('dataset', 'unknown')
        save_path = f'{args.output_dir}/batch_size_evolution_{dataset}.png'
        
        plot_batch_size_evolution(
            adm_results,
            save_path=save_path
        )
        print("\n✅ BWA batch size evolution plotted!")

    print("\n✅ Comparison completed!")


if __name__ == "__main__":
    main()
