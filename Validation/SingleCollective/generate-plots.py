#!/usr/bin/env python3
"""
Generate plots for comparing simulation data with NCCL benchmark results using matplotlib.
"""

import os
import re
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 6
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.fancybox'] = False

# Paths
BASE_DIR = Path("/home/yxli/data/Validation/SingleCollective")
SET_NAME = "1x8-allgather-ring"
GROUND_TRUTH_DIR = BASE_DIR / "GroundTruth" / SET_NAME
SIM_RESULT_FILE = BASE_DIR / "SimResult" / f"{SET_NAME}.txt"
OUTPUT_DIR = BASE_DIR / "PostData" / SET_NAME

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_txt_file(txt_file, config_name):
    """
    Parse the 1x8.txt file and extract data for a specific configuration.
    Returns dict: {size_in_bytes: time_in_ns}
    """
    data = {}
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Config'):
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            
            config, time_ns = parts
            
            # Check if this line matches our config
            if config.startswith(config_name + "_on_"):
                # Extract size (e.g., "128B" from "ag_allpair_LL128_x_1_on_128B")
                match = re.search(r'_on_(\d+)B', config)
                if match:
                    size = int(match.group(1))
                    time_ns = int(time_ns)
                    data[size] = time_ns
    
    return data

def parse_csv_file(csv_file):
    """
    Parse CSV file and extract data from rows where size >= 128.
    Returns dict: {size: time_in_us}
    """
    data = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            size = int(row['size'])
            if size >= 128:
                time_us = float(row['time'])
                data[size] = time_us
    
    return data

def save_comparison_data(txt_data, csv_data, output_file):
    """
    Save comparison data to CSV file with columns: size, sim_result, ground_truth, ratio
    """
    # Get common sizes
    common_sizes = sorted(set(txt_data.keys()) & set(csv_data.keys()))
    
    if len(common_sizes) == 0:
        return False
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'sim_result', 'ground_truth', 'ratio'])
        
        for size in common_sizes:
            sim_time_us = txt_data[size] / 1000.0  # Convert ns to us
            nccl_time_us = csv_data[size]
            ratio = nccl_time_us / sim_time_us if sim_time_us > 0 else 0
            writer.writerow([size, f'{sim_time_us:.6f}', f'{nccl_time_us:.6f}', f'{ratio:.6f}'])
    
    return True

def generate_plot(txt_data, csv_data, output_file, config_name, title_suffix):
    """
    Generate a matplotlib plot comparing simulation and NCCL benchmark data.
    """
    # Get common sizes
    common_sizes = sorted(set(txt_data.keys()) & set(csv_data.keys()))
    
    if len(common_sizes) == 0:
        return False
    
    # Prepare data
    sizes = np.array(common_sizes)
    sim_times = np.array([txt_data[size] / 1000.0 for size in common_sizes])  # Convert ns to us
    nccl_times = np.array([csv_data[size] for size in common_sizes])
    
    # Create figure with better aspect ratio for publications
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot both lines with distinct markers and colors
    ax.loglog(sizes, sim_times, 'o-', label='Simulation Result', 
              color='#2E86AB', linewidth=2.5, markersize=7, 
              markerfacecolor='white', markeredgewidth=1.5, markeredgecolor='#2E86AB')
    ax.loglog(sizes, nccl_times, 's-', label='Ground Truth (MSCCL over RCCL)', 
              color='#A23B72', linewidth=2.5, markersize=7,
              markerfacecolor='white', markeredgewidth=1.5, markeredgecolor='#A23B72')
    
    # Formatting
    ax.set_xlabel('Message Size (Bytes)', fontsize=13, fontweight='semibold')
    ax.set_ylabel('Time (μs)', fontsize=13, fontweight='semibold')
    ax.set_title(f'{config_name} - {title_suffix}', fontsize=14, fontweight='bold', pad=15)
    
    # Enhanced grid
    ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Legend with better positioning
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, 
              edgecolor='black', fancybox=False, shadow=False)
    
    # Enable minor ticks for log scale
    ax.minorticks_on()
    
    # Adjust tick label size
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='both', which='minor', labelsize=9)
    
    # Save figure with high quality
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return True

def main():
    """Main function to process all configurations and generate plots."""
    
    # Find all unique configuration names (base names without _in_place or _out_of_place)
    csv_files = list(GROUND_TRUTH_DIR.glob("*_in_place.csv"))
    
    configs_processed = 0
    configs_skipped = 0
    
    for in_place_csv in csv_files:
        # Extract config name
        config_name = in_place_csv.stem.replace('_in_place', '')
        out_of_place_csv = GROUND_TRUTH_DIR / f"{config_name}_out_of_place.csv"
        
        # Check if out_of_place file exists
        if not out_of_place_csv.exists():
            print(f"Warning: {out_of_place_csv} not found, skipping {config_name}")
            configs_skipped += 1
            continue
        
        print(f"Processing {config_name}...")
        
        # Parse txt file data
        txt_data = parse_txt_file(SIM_RESULT_FILE, config_name)
        
        if not txt_data:
            print(f"  No data found in 1x8.txt for {config_name}, skipping")
            configs_skipped += 1
            continue
        
        # Parse CSV files
        try:
            in_place_data = parse_csv_file(in_place_csv)
            out_of_place_data = parse_csv_file(out_of_place_csv)
        except Exception as e:
            print(f"  Error parsing CSV files: {e}, skipping")
            configs_skipped += 1
            continue
        
        # Generate in_place comparison
        in_place_png = OUTPUT_DIR / f"{config_name}_in_place.png"
        in_place_csv_out = OUTPUT_DIR / f"{config_name}_in_place.csv"
        
        if generate_plot(txt_data, in_place_data, str(in_place_png), 
                        config_name, "In-Place"):
            print(f"  Generated {in_place_png.name}")
            # Save comparison data
            if save_comparison_data(txt_data, in_place_data, str(in_place_csv_out)):
                print(f"  Saved data to {in_place_csv_out.name}")
        else:
            print(f"  No common data points for in_place")
        
        # Generate out_of_place comparison
        out_of_place_png = OUTPUT_DIR / f"{config_name}_out_of_place.png"
        out_of_place_csv_out = OUTPUT_DIR / f"{config_name}_out_of_place.csv"
        
        if generate_plot(txt_data, out_of_place_data, str(out_of_place_png), 
                        config_name, "Out-of-Place"):
            print(f"  Generated {out_of_place_png.name}")
            # Save comparison data
            if save_comparison_data(txt_data, out_of_place_data, str(out_of_place_csv_out)):
                print(f"  Saved data to {out_of_place_csv_out.name}")
        else:
            print(f"  No common data points for out_of_place")
        
        configs_processed += 1
    
    print(f"\nSummary:")
    print(f"  Configurations processed: {configs_processed}")
    print(f"  Configurations skipped: {configs_skipped}")
    print(f"  Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
