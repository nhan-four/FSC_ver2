"""
Plot Accuracy vs SNR: FuzSemCom vs L-DeepSC System Opt
========================================================
File riêng chỉ để vẽ biểu đồ Accuracy vs SNR
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR = Path(__file__).parent / "results"
FUZSEMCOM_JSON = RESULTS_DIR / "full_comparison_20251217_220929.json"
LDEEPSC_JSON = Path(__file__).parent.parent / "experiments" / "l_deepsc_system_opt" / "results" / "l_deepsc_system_opt_results.json"


def load_data():
    """Load FuzSemCom và L-DeepSC results"""
    with open(FUZSEMCOM_JSON, 'r') as f:
        fuzsemcom = json.load(f)
    
    with open(LDEEPSC_JSON, 'r') as f:
        ldeepsc = json.load(f)
    
    return fuzsemcom, ldeepsc


def plot_accuracy_vs_snr_updated(fuzsemcom, ldeepsc, output_path):
    """
    Plot Accuracy vs SNR: FuzSemCom (4 kênh) + L-DeepSC System Opt
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors
    colors = {
        'awgn': '#2196F3',
        'rayleigh': '#F44336',
        'rician': '#4CAF50',
        'lora': '#FF9800',
        'ldeepsc_before': '#9C27B0',
        'ldeepsc_after': '#795548',
    }
    
    # FuzSemCom channels
    for ch in ['awgn', 'rayleigh', 'rician', 'lora']:
        data = fuzsemcom['results'][ch]
        snr = data['snr_db']
        acc = [x * 100 for x in data['semantic_accuracy']]
        ax.plot(snr, acc, '-o', color=colors[ch], linewidth=2.5, markersize=8,
                label=f'FuzSemCom ({ch.upper()})')
    
    # L-DeepSC System Opt
    snr_range = ldeepsc['config']['snr_range']
    acc_before = [ldeepsc['results_by_snr'][str(s)]['accuracy_before'] * 100 for s in snr_range]
    acc_after = [ldeepsc['results_by_snr'][str(s)]['accuracy_after'] * 100 for s in snr_range]
    
    ax.plot(snr_range, acc_before, '--^', color=colors['ldeepsc_before'], linewidth=2.5, markersize=9,
            label='L-DeepSC Sys.Opt (Before Channel)')
    ax.plot(snr_range, acc_after, '-s', color=colors['ldeepsc_after'], linewidth=2.5, markersize=9,
            label='L-DeepSC Sys.Opt (After Channel)')
    
    ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Semantic Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('FuzSemCom vs L-DeepSC System Optimization: Accuracy over SNR', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95)
    ax.set_ylim(0, 105)
    ax.set_xlim(-16, 31)
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15, 20, 25, 30])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Plotting Accuracy vs SNR: FuzSemCom vs L-DeepSC System Opt")
    print("=" * 60)
    
    # Load data
    fuzsemcom, ldeepsc = load_data()
    print(f"Loaded FuzSemCom: {len(fuzsemcom['results'])} channels")
    print(f"Loaded L-DeepSC Sys.Opt: best acc = {ldeepsc['best_accuracy_after_channel']*100:.2f}%")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot Accuracy vs SNR
    output = RESULTS_DIR / f"comparison_acc_vs_snr_updated_{timestamp}.png"
    plot_accuracy_vs_snr_updated(fuzsemcom, ldeepsc, output)
    
    print("\n" + "=" * 60)
    print(f"DONE! Created plot: {output.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

