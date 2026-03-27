"""
DARLIN simulation script
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="DARLIN simulation script")
    parser.add_argument('--input', type=Path, required=True, help='Path to input data (e.g., gene expression profiles)')
    parser.add_argument('--RA', type=float, default=3.0, help='The number of RA per grids')
    parser.add_argument('--TA', type=float, default=4.5, help='The number of TA per grids')
    parser.add_argument('--informative_UMI', type=float, default=0.2, help='Proportion of informative UMI')
    parser.add_argument('--cutoff', type=int, default=200, help='Cutoff for informative UMI number')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output figures')
    return parser.parse_args()


def simulate_darlin(input_data, RA, TA, informative_UMI, cutoff, output, dpi=300):
    # Simulate DARLIN data based on input data and parameters
    simulated_data = input_data.copy()
    
    # Simulate RA and TA counts
    simulated_data['RA'] = simulated_data['interp_grid_number'] * RA
    simulated_data['TA'] = simulated_data['interp_grid_number'] * TA
    simulated_data['total_UMI'] = simulated_data['RA'] + simulated_data['TA']

    simulated_data['informative_RA'] = simulated_data['RA'] * informative_UMI
    simulated_data['informative_TA'] = simulated_data['TA'] * informative_UMI
    simulated_data['informative_UMI'] = simulated_data['total_UMI'] * informative_UMI
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(simulated_data['informative_RA'], bins=50, color='#1f77b4', alpha=0.7)
    ax[0].axvline(cutoff, color='red', linestyle='--', label=f'Cutoff = {cutoff}')
    ax[0].legend()
    ax[0].set_title('RA Distribution')

    ax[1].hist(simulated_data['informative_TA'], bins=50, color='#2ca02c', alpha=0.7)
    ax[1].axvline(cutoff, color='red', linestyle='--', label=f'Cutoff = {cutoff}')
    ax[1].legend()
    ax[1].set_title('TA Distribution')

    ax[2].hist(simulated_data['informative_UMI'], bins=50, color='#d62728', alpha=0.7)
    ax[2].axvline(cutoff, color='red', linestyle='--', label=f'Cutoff = {cutoff}')
    ax[2].legend()
    ax[2].set_title('Total UMI Distribution')

    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close()

    simulated_data['is_informative_RA'] = simulated_data['informative_RA'] >= cutoff
    simulated_data['is_informative_TA'] = simulated_data['informative_TA'] >= cutoff
    simulated_data['is_informative'] = simulated_data['informative_UMI'] >= cutoff

    return simulated_data

def main():
    args = parse_arguments()
    output_path = args.input.parent / 'simulated_darlin'
    output_path.mkdir(exist_ok=True)
    name = args.input.stem
    # Load input data
    input_data = pd.read_csv(args.input)
    datasets = input_data['dataset'].unique()
    summary_data = []
    for dataset in datasets:
        subset = input_data[input_data['dataset'] == dataset]
        simulated_data = simulate_darlin(subset, args.RA, args.TA, args.informative_UMI, args.cutoff, output_path / f'{name}_{dataset}.png', args.dpi)
        summary_data.append(simulated_data)

    summary_data = pd.concat(summary_data, ignore_index=True)
    summary_data.to_csv(output_path / f'{name}_simulated_darlin_data.csv', index=False)
    print(f"Simulated DARLIN data saved to {output_path}")

if __name__ == "__main__":
    main()