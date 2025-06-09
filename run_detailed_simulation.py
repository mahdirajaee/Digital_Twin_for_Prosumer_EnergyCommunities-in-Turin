#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import json
import sys
import os

sys.path.append(os.path.dirname(__file__))

from run_simulation import SimpleCommunity, SimpleCommunitySimulation

def plot_simulation_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Prosumer Community Simulation Results', fontsize=16, fontweight='bold')
    
    timestamps = pd.to_datetime(results['timestamps'])
    
    ax1 = axes[0, 0]
    ax1.plot(timestamps, results['total_load'], label='Total Load', linewidth=2, color='blue')
    ax1.plot(timestamps, results['pv_generation'], label='PV Generation', linewidth=2, color='orange')
    ax1.set_title('Community Load vs PV Generation')
    ax1.set_ylabel('Power (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    ax2 = axes[0, 1]
    ax2.plot(timestamps, results['grid_exchange'], linewidth=2, color='red')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Grid Import/Export')
    ax2.set_ylabel('Power (kW)')
    ax2.fill_between(timestamps, 0, results['grid_exchange'], 
                     where=np.array(results['grid_exchange']) > 0, 
                     color='red', alpha=0.3, label='Import')
    ax2.fill_between(timestamps, 0, results['grid_exchange'], 
                     where=np.array(results['grid_exchange']) < 0, 
                     color='green', alpha=0.3, label='Export')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    ax3 = axes[1, 0]
    ax3.plot(timestamps, results['battery_soc'], linewidth=2, color='green')
    ax3.set_title('Average Battery State of Charge')
    ax3.set_ylabel('SOC')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    ax4 = axes[1, 1]
    building_costs = results['building_savings']
    ax4.hist(building_costs, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title('Distribution of Building Costs')
    ax4.set_xlabel('Cost (€)')
    ax4.set_ylabel('Number of Buildings')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results plot saved as 'simulation_results.png'")

def create_tariff_comparison_chart(results):
    tariff_comparison = results['tariff_comparison']
    
    tariffs = list(tariff_comparison.keys())
    costs = [tariff_comparison[t]['total_cost'] for t in tariffs]
    savings_pct = [tariff_comparison[t]['cost_savings_percent'] for t in tariffs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(tariffs)]
    
    ax1.bar(tariffs, costs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Total Cost by Tariff Type')
    ax1.set_ylabel('Total Cost (€)')
    ax1.grid(True, alpha=0.3)
    
    for i, cost in enumerate(costs):
        ax1.text(i, cost + max(costs) * 0.01, f'€{cost:.0f}', 
                ha='center', va='bottom', fontweight='bold')
    
    colors_savings = ['green' if x > 0 else 'red' for x in savings_pct]
    bars = ax2.bar(tariffs, savings_pct, color=colors_savings, alpha=0.7, edgecolor='black')
    ax2.set_title('Savings Percentage by Tariff Type')
    ax2.set_ylabel('Savings (%)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    for i, savings in enumerate(savings_pct):
        ax2.text(i, savings + (max(savings_pct) * 0.05 if savings > 0 else min(savings_pct) * 0.05), 
                f'{savings:.1f}%', ha='center', 
                va='bottom' if savings > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('tariff_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Tariff comparison chart saved as 'tariff_comparison.png'")

def run_detailed_simulation():
    print("=== Running Detailed Community Simulation ===")
    
    community = SimpleCommunity(
        num_buildings=50,
        residents_per_building=20,
        pv_capacity=15,
        battery_capacity=35,
        tariff_types=['tou', 'cpp', 'rtp', 'edr']
    )
    
    sim = SimpleCommunitySimulation(community)
    
    results = sim.run_simulation(
        duration_days=14,
        comfort_threshold=0.8,
        price_sensitivity=1.2
    )
    
    with open('simulation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to 'simulation_results.json'")
    
    plot_simulation_results(results)
    create_tariff_comparison_chart(results)
    
    return results

if __name__ == "__main__":
    try:
        results = run_detailed_simulation()
        
        print("\n=== DETAILED SIMULATION SUMMARY ===")
        summary = results['simulation_summary']
        print(f"Community Size: {summary['total_buildings']} buildings")
        print(f"Best Performing Tariff: {summary['best_tariff'].upper()}")
        print(f"Cost Savings: €{summary['total_savings']:.2f} ({summary['savings_percent']:.1f}%)")
        print(f"Peak Load Reduction: {summary['peak_reduction']:.2f} kW")
        
        fairness = results['fairness_analysis']
        print(f"\nFairness Metrics:")
        print(f"- Gini Coefficient: {fairness['gini_coefficient']:.3f} (lower is better)")
        print(f"- Cost Variation: {fairness['coefficient_of_variation']:.3f}")
        
        print(f"\nSimulation completed successfully!")
        
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Please install matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"Error running simulation: {e}")
