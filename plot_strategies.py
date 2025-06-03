import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

STRATEGIES = ["vanilla", "random", "noisy_best", "hybrid"]

def extract_function_name(folder_name):
    m = re.search(r"problem\s+(f\d+)", folder_name)
    return m.group(1) if m else None

def extract_strategy_name(folder_name):
    for strat in STRATEGIES:
        if f"_{strat}_" in folder_name:
            return strat
    return None

def extract_dimension(folder_name):
    m = re.search(r"_dim(\d+)_", folder_name)
    return int(m.group(1)) if m else None

def load_all_data(logs_dir):
    conv_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    div_data  = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for folder in os.listdir(logs_dir):
        path = os.path.join(logs_dir, folder)
        if not os.path.isdir(path):
            continue
        func = extract_function_name(folder)
        strat = extract_strategy_name(folder)
        dim = extract_dimension(folder)
        if func is None or strat is None or dim is None:
            continue
        conv_path = os.path.join(path, "convergence.csv")
        div_path  = os.path.join(path, "diversity.csv")
        if os.path.isfile(conv_path):
            df_conv = pd.read_csv(conv_path)
            conv_data[func][dim][strat].append(df_conv)
        if os.path.isfile(div_path):
            df_div = pd.read_csv(div_path)
            div_data[func][dim][strat].append(df_div)
    return conv_data, div_data

def plot_aggregated_convergence(conv_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for func, dims in conv_data.items():
        for dim, strat_dict in dims.items():
            plt.figure(figsize=(10, 6))
            for strat in STRATEGIES:
                dfs = strat_dict.get(strat, [])
                if not dfs:
                    continue
                df_concat = pd.concat(dfs, ignore_index=True)
                grouped = df_concat.groupby("generation")["best_score"]
                mean = grouped.mean()
                std = grouped.std()
                plt.plot(mean.index, mean.values, label=strat, linewidth=2)
                plt.fill_between(
                    mean.index,
                    mean.values - std.values,
                    mean.values + std.values,
                    alpha=0.2
                )
            plt.title(f"Aggregated Convergence for {func} ({dim}D)")
            plt.xlabel("Generation")
            plt.ylabel("Best Score")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            fname = f"{func}_dim{dim}_convergence.png"
            plt.savefig(os.path.join(output_dir, fname))
            plt.close()
            print(f"Saved: {fname}")

def plot_aggregated_diversity(div_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for func, dims in div_data.items():
        for dim, strat_dict in dims.items():
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Aggregated Diversity for {func} ({dim}D)", fontsize=16)
            ax_dist = axs[0, 0]
            ax_var  = axs[0, 1]
            ax_sp   = axs[1, 0]
            axs[1, 1].axis("off")
            for strat in STRATEGIES:
                dfs = strat_dict.get(strat, [])
                if not dfs:
                    continue
                df_concat = pd.concat(dfs, ignore_index=True)
                gd = df_concat.groupby("generation")
                mean_dist = gd["mean_distance"].mean()
                std_dist = gd["mean_distance"].std()
                mean_var = gd["fitness_variance"].mean()
                std_var = gd["fitness_variance"].std()
                mean_sp = gd["fitness_spread"].mean()
                std_sp = gd["fitness_spread"].std()
                ax_dist.plot(mean_dist.index, mean_dist.values, label=strat, linewidth=2)
                ax_dist.fill_between(
                    mean_dist.index,
                    mean_dist.values - std_dist.values,
                    mean_dist.values + std_dist.values,
                    alpha=0.2
                )
                ax_var.plot(mean_var.index, mean_var.values, label=strat, linewidth=2)
                ax_var.fill_between(
                    mean_var.index,
                    mean_var.values - std_var.values,
                    mean_var.values + std_var.values,
                    alpha=0.2
                )
                ax_sp.plot(mean_sp.index, mean_sp.values, label=strat, linewidth=2)
                ax_sp.fill_between(
                    mean_sp.index,
                    mean_sp.values - std_sp.values,
                    mean_sp.values + std_sp.values,
                    alpha=0.2
                )
            ax_dist.set_title("Mean Distance to Centroid")
            ax_dist.set_xlabel("Generation")
            ax_dist.set_ylabel("Mean Distance")
            ax_dist.grid(True)
            ax_var.set_title("Fitness Variance")
            ax_var.set_xlabel("Generation")
            ax_var.set_ylabel("Variance")
            ax_var.grid(True)
            ax_sp.set_title("Fitness Spread from Best")
            ax_sp.set_xlabel("Generation")
            ax_sp.set_ylabel("Spread")
            ax_sp.grid(True)
            ax_dist.legend(loc="best", fontsize=10)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fname = f"{func}_dim{dim}_diversity.png"
            plt.savefig(os.path.join(output_dir, fname))
            plt.close()
            print(f"Saved: {fname}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_logs_comparison>")
        sys.exit(1)
    LOGS_DIR = sys.argv[1]
    conv_data, div_data = load_all_data(LOGS_DIR)
    plot_aggregated_convergence(conv_data, output_dir=os.path.join("plots", "convergence"))
    plot_aggregated_diversity(div_data, output_dir=os.path.join("plots", "diversity"))
