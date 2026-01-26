"""
Analyze and compare importance scores between math and general datasets.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from scipy import stats

# Paths
MATH_PATH = "importance_scores_tensors/importance_scores_math.safetensors"
GENERAL_PATH = "importance_scores_tensors/importance_scores_general.safetensors"
OUTPUT_DIR = Path("importance_analysis_plots")


def load_importance_scores():
    """Load importance scores for both math and general datasets."""
    math_scores = load_file(MATH_PATH)["mlp"]
    general_scores = load_file(GENERAL_PATH)["mlp"]
    return math_scores, general_scores


def compute_layer_statistics(math_scores, general_scores):
    """Compute per-layer statistics for both datasets."""
    num_layers = math_scores.shape[0]

    stats_dict = {
        "math_mean": [],
        "math_max": [],
        "general_mean": [],
        "general_max": [],
        "spearman_corr": [],
        "spearman_pval": [],
    }

    for layer_idx in range(num_layers):
        math_layer = math_scores[layer_idx].numpy()
        general_layer = general_scores[layer_idx].numpy()

        stats_dict["math_mean"].append(math_layer.mean())
        stats_dict["math_max"].append(math_layer.max())
        stats_dict["general_mean"].append(general_layer.mean())
        stats_dict["general_max"].append(general_layer.max())

        # Spearman rank correlation
        corr, pval = stats.spearmanr(math_layer, general_layer)
        stats_dict["spearman_corr"].append(corr)
        stats_dict["spearman_pval"].append(pval)

    return {k: np.array(v) for k, v in stats_dict.items()}


def compute_topk_overlap(math_scores, general_scores, k_values=(100, 500, 1000)):
    """Compute top-k neuron overlap between math and general for each layer."""
    num_layers = math_scores.shape[0]
    overlap_results = {k: [] for k in k_values}

    for layer_idx in range(num_layers):
        math_topk = set(torch.topk(math_scores[layer_idx], max(k_values)).indices.tolist())
        general_topk = set(torch.topk(general_scores[layer_idx], max(k_values)).indices.tolist())

        for k in k_values:
            math_k = set(torch.topk(math_scores[layer_idx], k).indices.tolist())
            general_k = set(torch.topk(general_scores[layer_idx], k).indices.tolist())

            # Jaccard similarity
            intersection = len(math_k & general_k)
            union = len(math_k | general_k)
            jaccard = intersection / union if union > 0 else 0

            # Overlap percentage
            overlap_pct = intersection / k * 100

            overlap_results[k].append({"jaccard": jaccard, "overlap_pct": overlap_pct})

    return overlap_results


def find_shifted_neurons(math_scores, general_scores, top_n=20):
    """Find neurons with largest rank shifts between math and general."""
    num_layers = math_scores.shape[0]
    intermediate_size = math_scores.shape[1]

    all_shifts = []

    for layer_idx in range(num_layers):
        # Compute ranks (higher importance = lower rank number)
        math_ranks = stats.rankdata(-math_scores[layer_idx].numpy())
        general_ranks = stats.rankdata(-general_scores[layer_idx].numpy())

        for neuron_idx in range(intermediate_size):
            rank_shift = general_ranks[neuron_idx] - math_ranks[neuron_idx]
            all_shifts.append({
                "layer": layer_idx,
                "neuron": neuron_idx,
                "math_rank": int(math_ranks[neuron_idx]),
                "general_rank": int(general_ranks[neuron_idx]),
                "rank_shift": rank_shift,  # positive = more important in math
                "math_importance": math_scores[layer_idx, neuron_idx].item(),
                "general_importance": general_scores[layer_idx, neuron_idx].item(),
            })

    # Sort by rank shift
    math_specialized = sorted(all_shifts, key=lambda x: -x["rank_shift"])[:top_n]
    general_specialized = sorted(all_shifts, key=lambda x: x["rank_shift"])[:top_n]

    return math_specialized, general_specialized


def plot_layer_statistics(layer_stats, output_dir):
    """Plot layer-wise statistics comparison."""
    num_layers = len(layer_stats["math_mean"])
    layers = np.arange(num_layers)

    # Figure 1: Mean and Max importance per layer
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean importance
    ax = axes[0]
    width = 0.35
    ax.bar(layers - width / 2, layer_stats["math_mean"], width, label="Math", alpha=0.8)
    ax.bar(layers + width / 2, layer_stats["general_mean"], width, label="General", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Importance")
    ax.set_title("Mean Importance per Layer")
    ax.legend()
    ax.set_xticks(layers[::4])

    # Max importance
    ax = axes[1]
    ax.bar(layers - width / 2, layer_stats["math_max"], width, label="Math", alpha=0.8)
    ax.bar(layers + width / 2, layer_stats["general_max"], width, label="General", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max Importance")
    ax.set_title("Max Importance per Layer")
    ax.legend()
    ax.set_xticks(layers[::4])

    plt.tight_layout()
    plt.savefig(output_dir / "layer_importance_comparison.png", dpi=150)
    plt.close()


def plot_divergence_summary(layer_stats, overlap_results, output_dir, math_scores, general_scores):
    """Plot a combined figure showing how different the importance rankings are."""
    num_layers = len(layer_stats["spearman_corr"])
    layers = np.arange(num_layers)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Spearman correlation (lower = more different)
    ax = axes[0]
    colors = ["#d62728" if c < 0.3 else "#ff7f0e" if c < 0.5 else "#2ca02c" for c in layer_stats["spearman_corr"]]
    bars = ax.bar(layers, layer_stats["spearman_corr"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.axhline(y=layer_stats["spearman_corr"].mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean: {layer_stats['spearman_corr'].mean():.2f}")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Spearman Rank Correlation", fontsize=11)
    ax.set_title("Rank Correlation Between Math & General", fontsize=12)
    ax.set_xticks(layers[::4])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")

    # Right: Top-k "different" neurons (100% - overlap%)
    ax = axes[1]
    k = 2048  # Actual pruning target

    # Compute top-2048 overlap directly (not in overlap_results which has 100, 500, 1000)
    different_pct = []
    for layer_idx in range(num_layers):
        math_topk = set(torch.topk(math_scores[layer_idx], k).indices.tolist())
        general_topk = set(torch.topk(general_scores[layer_idx], k).indices.tolist())
        overlap = len(math_topk & general_topk)
        different_pct.append((1 - overlap / k) * 100)

    colors = ["#d62728" if d > 50 else "#ff7f0e" if d > 35 else "#2ca02c" for d in different_pct]
    bars = ax.bar(layers, different_pct, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    mean_diff = np.mean(different_pct)
    ax.axhline(y=mean_diff, color="black", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_diff:.0f}%")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("% of Top-2048 Neurons Different", fontsize=11)
    ax.set_title("Top-2048 Neurons: Math vs General", fontsize=12)
    ax.set_xticks(layers[::4])
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_dir / "importance_divergence.png", dpi=150)
    plt.close()


def print_summary_statistics(layer_stats, overlap_results, math_specialized, general_specialized):
    """Print summary statistics to console."""
    print("\n" + "=" * 70)
    print("IMPORTANCE SCORE ANALYSIS SUMMARY")
    print("=" * 70)

    # Layer-wise statistics
    print("\n--- Layer-wise Statistics ---")
    print(f"{'Layer':<8} {'Math Mean':>12} {'Gen Mean':>12} {'Math Max':>12} {'Gen Max':>12} {'Spearman':>10}")
    print("-" * 70)

    for i in range(len(layer_stats["math_mean"])):
        print(f"{i:<8} {layer_stats['math_mean'][i]:>12.4f} {layer_stats['general_mean'][i]:>12.4f} "
              f"{layer_stats['math_max'][i]:>12.4f} {layer_stats['general_max'][i]:>12.4f} "
              f"{layer_stats['spearman_corr'][i]:>10.4f}")

    # Overall statistics
    print("\n--- Overall Statistics ---")
    print(f"Mean Spearman correlation: {layer_stats['spearman_corr'].mean():.4f}")
    print(f"Min Spearman correlation:  {layer_stats['spearman_corr'].min():.4f} (Layer {layer_stats['spearman_corr'].argmin()})")
    print(f"Max Spearman correlation:  {layer_stats['spearman_corr'].max():.4f} (Layer {layer_stats['spearman_corr'].argmax()})")

    # Top-k overlap
    print("\n--- Top-k Overlap Summary ---")
    for k, results in overlap_results.items():
        mean_overlap = np.mean([r["overlap_pct"] for r in results])
        min_overlap = min([r["overlap_pct"] for r in results])
        max_overlap = max([r["overlap_pct"] for r in results])
        print(f"Top-{k}: Mean={mean_overlap:.1f}%, Min={min_overlap:.1f}%, Max={max_overlap:.1f}%")

    # Most shifted neurons
    print("\n--- Top 20 Math-Specialized Neurons ---")
    print("(Higher rank in math than general)")
    print(f"{'Layer':<8} {'Neuron':<10} {'Math Rank':>12} {'Gen Rank':>12} {'Rank Shift':>12}")
    print("-" * 60)
    for n in math_specialized:
        print(f"{n['layer']:<8} {n['neuron']:<10} {n['math_rank']:>12} {n['general_rank']:>12} {n['rank_shift']:>12.0f}")

    print("\n--- Top 20 General-Specialized Neurons ---")
    print("(Higher rank in general than math)")
    print(f"{'Layer':<8} {'Neuron':<10} {'Math Rank':>12} {'Gen Rank':>12} {'Rank Shift':>12}")
    print("-" * 60)
    for n in general_specialized:
        print(f"{n['layer']:<8} {n['neuron']:<10} {n['math_rank']:>12} {n['general_rank']:>12} {n['rank_shift']:>12.0f}")

    print("\n" + "=" * 70)


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading importance scores...")
    math_scores, general_scores = load_importance_scores()
    print(f"  Math scores shape: {math_scores.shape}")
    print(f"  General scores shape: {general_scores.shape}")

    # Compute statistics
    print("\nComputing layer-wise statistics...")
    layer_stats = compute_layer_statistics(math_scores, general_scores)

    print("Computing top-k overlap...")
    overlap_results = compute_topk_overlap(math_scores, general_scores)

    print("Finding shifted neurons...")
    math_specialized, general_specialized = find_shifted_neurons(math_scores, general_scores)

    # Generate plots
    print("\nGenerating plots...")
    plot_layer_statistics(layer_stats, OUTPUT_DIR)
    plot_divergence_summary(layer_stats, overlap_results, OUTPUT_DIR, math_scores, general_scores)

    # Print summary
    print_summary_statistics(layer_stats, overlap_results, math_specialized, general_specialized)

    print(f"\nPlots saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
