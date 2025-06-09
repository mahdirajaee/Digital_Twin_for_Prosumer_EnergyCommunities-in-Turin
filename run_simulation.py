#!/usr/bin/env python3

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(__file__))

import config_file as config

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

def generate_weather_data(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate weather data for Turin, Italy climate patterns"""
    hours = timestamps.hour
    days = timestamps.dayofyear
    
    # Turin-specific temperature patterns
    daily_temp_variation = 10 * np.sin(2 * np.pi * (hours - 6) / 24)
    
    # Turin seasonal variation: Winter ~2°C, Summer ~25°C
    winter_temp = 2.0
    summer_temp = 25.0
    seasonal_temp_variation = (summer_temp - winter_temp) / 2 * np.sin(2 * np.pi * (days - 80) / 365)
    base_temperature = (winter_temp + summer_temp) / 2
    temp_noise = np.random.normal(0, 3, len(timestamps))
    
    temperature = base_temperature + daily_temp_variation + seasonal_temp_variation + temp_noise
    
    # Solar irradiance for Turin latitude (45.07°N)
    base_solar = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    seasonal_solar_factor = 0.6 + 0.4 * np.sin(2 * np.pi * (days - 172) / 365)
    cloud_cover = 0.5 + 0.2 * np.cos(2 * np.pi * (days - 1) / 365) + np.random.beta(2, 3, len(timestamps)) * 0.3
    cloud_cover = np.clip(cloud_cover, 0, 1)
    peak_irradiance = 900  # Lower peak for Turin latitude
    
    solar_irradiance = base_solar * seasonal_solar_factor * (1 - 0.8 * cloud_cover) * peak_irradiance
    
    weather_df = pd.DataFrame({
        'temperature': temperature,
        'solar_irradiance': solar_irradiance,
    }, index=timestamps)
    
    return weather_df

def calculate_fairness_metrics(costs: List[float]) -> Dict[str, float]:
    costs = np.array(costs)
    n = len(costs)
    
    if n == 0:
        return {'gini_coefficient': 0, 'coefficient_of_variation': 0}
    
    costs_sorted = np.sort(costs)
    cumsum = np.cumsum(costs_sorted)
    
    gini = (2 * np.sum((np.arange(1, n + 1) * costs_sorted))) / (n * np.sum(costs_sorted)) - (n + 1) / n
    gini = max(0, min(1, gini))
    
    mean_cost = np.mean(costs)
    std_cost = np.std(costs)
    cv = std_cost / mean_cost if mean_cost > 0 else 0
    
    return {
        'gini_coefficient': gini,
        'coefficient_of_variation': cv,
        'mean_cost': mean_cost,
        'std_cost': std_cost,
        'min_cost': np.min(costs),
        'max_cost': np.max(costs)
    }

class SimpleBuilding:
    def __init__(self, building_id, residents=20, pv_capacity=10, battery_capacity=25):
        self.building_id = building_id
        self.residents = residents
        self.pv_capacity = pv_capacity  
        self.battery_capacity = battery_capacity  
        self.current_soc = 0.5  
        self.current_temp = 22.0
        
    def get_baseline_load(self, hour):
        hourly_pattern = [2.5, 2.2, 2.0, 2.0, 2.2, 2.8, 4.0, 5.5, 
                         4.8, 4.2, 4.0, 4.2, 4.5, 4.3, 4.0, 4.5, 
                         5.8, 6.5, 7.2, 6.8, 5.5, 4.2, 3.5, 2.8]
        base_load = hourly_pattern[hour % 24]
        return base_load * self.residents * 0.1  
        
    def generate_pv_profile(self, timestamps, weather_data):
        solar_irradiance = weather_data['solar_irradiance'].values
        pv_output = (solar_irradiance / 1000) * self.pv_capacity * 0.2  
        return np.maximum(0, pv_output)

class SimpleCommunity:
    def __init__(self, num_buildings=10, residents_per_building=20, 
                 pv_capacity=10, battery_capacity=25, tariff_types=['tou']):
        self.num_buildings = num_buildings
        self.residents_per_building = residents_per_building
        self.pv_capacity = pv_capacity
        self.battery_capacity = battery_capacity
        self.tariff_types = tariff_types
        
        self.buildings = []
        for i in range(num_buildings):
            building = SimpleBuilding(
                building_id=i,
                residents=residents_per_building,
                pv_capacity=pv_capacity,
                battery_capacity=battery_capacity
            )
            self.buildings.append(building)

class SimpleTariff:
    def __init__(self, tariff_type='tou'):
        self.tariff_type = tariff_type
        self.rates = config.DEFAULT_TARIFF_RATES.get(tariff_type, config.DEFAULT_TARIFF_RATES['tou'])
    
    def get_price_schedule(self, timestamps, load_profile=None):
        prices = np.zeros(len(timestamps))
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            if self.tariff_type == 'tou':
                if hour in self.rates['off_peak_hours']:
                    prices[i] = self.rates['off_peak']
                elif hour in self.rates['on_peak_hours']:
                    prices[i] = self.rates['on_peak']
                else:
                    prices[i] = self.rates['mid_peak']
            else:
                prices[i] = self.rates.get('base_rate', 0.12)
        
        return prices

class SimpleOptimizationResult:
    def __init__(self, num_buildings, num_hours):
        self.load_schedule = {}
        self.battery_schedule = {}
        self.convergence_iterations = 10
        
        for building_id in range(num_buildings):
            self.load_schedule[building_id] = np.ones(num_hours)
            self.battery_schedule[building_id] = np.random.uniform(-2, 2, num_hours)

class SimpleCommunitySimulation:
    def __init__(self, community):
        self.community = community
        self.baseline_established = False
        self.baseline_metrics = {}
        self.results_history = []
    
    def establish_baseline(self, duration_days, weather_data=None):
        baseline_hours = duration_days * 24
        baseline_load = np.zeros(baseline_hours)
        baseline_cost = 0
        # Use average EUR rate for baseline calculation (average of TOU rates)
        flat_rate = 0.25  # €0.25/kWh - average of Italian electricity rates
        
        for building in self.community.buildings:
            for hour in range(baseline_hours):
                load = building.get_baseline_load(hour)
                baseline_load[hour] += load
                baseline_cost += load * flat_rate
        
        self.baseline_metrics = {
            'total_load_profile': baseline_load,
            'total_cost': baseline_cost,
            'peak_load': np.max(baseline_load),
            'average_load': np.mean(baseline_load),
            'load_factor': np.mean(baseline_load) / np.max(baseline_load)
        }
        
        self.baseline_established = True
        print(f"Baseline established: Peak={self.baseline_metrics['peak_load']:.2f}kW, Cost=€{self.baseline_metrics['total_cost']:.2f}")
    
    def run_simulation(self, duration_days=7, comfort_threshold=0.8, price_sensitivity=1.0, weather_data=None):
        print(f"Starting simulation for {duration_days} days with {self.community.num_buildings} buildings...")
        
        if not self.baseline_established:
            self.establish_baseline(duration_days, weather_data)
        
        total_hours = duration_days * 24
        timestamps = pd.date_range('2024-01-01', periods=total_hours, freq='h')
        
        if weather_data is None:
            weather_data = generate_weather_data(timestamps)
        
        tariff_results = {}
        
        for tariff_type in self.community.tariff_types:
            print(f"Running simulation for {tariff_type} tariff...")
            
            tariff = SimpleTariff(tariff_type)
            tariff_prices = tariff.get_price_schedule(timestamps)
            
            optimization_result = SimpleOptimizationResult(self.community.num_buildings, total_hours)
            
            building_responses = self.simulate_building_responses(
                optimization_result, timestamps, weather_data, tariff_prices
            )
            
            scenario_metrics = self.calculate_scenario_metrics(
                building_responses, tariff_prices, timestamps
            )
            
            tariff_results[tariff_type] = {
                'scenario_metrics': scenario_metrics,
                'building_responses': building_responses
            }
        
        best_tariff = min(tariff_results.keys(), 
                         key=lambda t: tariff_results[t]['scenario_metrics']['total_cost'])
        
        final_results = self.compile_final_results(tariff_results, best_tariff, timestamps)
        self.results_history.append(final_results)
        
        return final_results
    
    def simulate_building_responses(self, optimization_result, timestamps, weather_data, tariff_prices):
        building_responses = {}
        
        for building in self.community.buildings:
            building_id = building.building_id
            
            load_schedule = optimization_result.load_schedule.get(building_id, np.ones(len(timestamps)))
            battery_schedule = optimization_result.battery_schedule.get(building_id, np.zeros(len(timestamps)))
            
            pv_generation = building.generate_pv_profile(timestamps, weather_data)
            
            actual_load = np.zeros(len(timestamps))
            grid_exchange = np.zeros(len(timestamps))
            energy_costs = np.zeros(len(timestamps))
            soc_profile = np.full(len(timestamps), 0.5)
            
            for hour in range(len(timestamps)):
                actual_load[hour] = building.get_baseline_load(hour) * load_schedule[hour]
                net_load = actual_load[hour] - pv_generation[hour] - battery_schedule[hour]
                grid_exchange[hour] = net_load
                energy_costs[hour] = max(0, grid_exchange[hour]) * tariff_prices[hour]
            
            building_responses[building_id] = {
                'actual_load': actual_load,
                'pv_generation': pv_generation,
                'grid_exchange': grid_exchange,
                'energy_costs': energy_costs,
                'soc_profile': soc_profile,
                'total_cost': np.sum(energy_costs),
                'peak_load': np.max(actual_load),
                'pv_utilization': np.sum(np.minimum(pv_generation, actual_load)) / (np.sum(pv_generation) + 1e-6)
            }
        
        return building_responses
    
    def calculate_scenario_metrics(self, building_responses, tariff_prices, timestamps):
        total_cost = sum(response['total_cost'] for response in building_responses.values())
        total_load = np.sum([response['actual_load'] for response in building_responses.values()], axis=0)
        total_pv = np.sum([response['pv_generation'] for response in building_responses.values()], axis=0)
        total_grid = np.sum([response['grid_exchange'] for response in building_responses.values()], axis=0)
        
        building_costs = [response['total_cost'] for response in building_responses.values()]
        
        cost_savings = self.baseline_metrics['total_cost'] - total_cost
        peak_reduction = self.baseline_metrics['peak_load'] - np.max(total_load)
        
        fairness_metrics = calculate_fairness_metrics(building_costs)
        
        return {
            'total_cost': total_cost,
            'cost_savings': cost_savings,
            'cost_savings_percent': (cost_savings / self.baseline_metrics['total_cost']) * 100,
            'peak_load': np.max(total_load),
            'peak_reduction': peak_reduction,
            'peak_reduction_percent': (peak_reduction / self.baseline_metrics['peak_load']) * 100,
            'fairness_gini': fairness_metrics['gini_coefficient'],
            'fairness_cv': fairness_metrics['coefficient_of_variation'],
            'building_cost_range': max(building_costs) - min(building_costs)
        }
    
    def compile_final_results(self, tariff_results, best_tariff, timestamps):
        best_result = tariff_results[best_tariff]
        
        building_responses = best_result['building_responses']
        total_load = np.sum([response['actual_load'] for response in building_responses.values()], axis=0)
        total_pv = np.sum([response['pv_generation'] for response in building_responses.values()], axis=0)
        total_grid = np.sum([response['grid_exchange'] for response in building_responses.values()], axis=0)
        avg_soc = np.mean([response['soc_profile'] for response in building_responses.values()], axis=0)
        
        return {
            'simulation_summary': {
                'best_tariff': best_tariff,
                'total_buildings': self.community.num_buildings,
                'simulation_duration_hours': len(timestamps),
                'baseline_cost': self.baseline_metrics['total_cost'],
                'optimized_cost': best_result['scenario_metrics']['total_cost'],
                'total_savings': best_result['scenario_metrics']['cost_savings'],
                'savings_percent': best_result['scenario_metrics']['cost_savings_percent'],
                'peak_reduction': best_result['scenario_metrics']['peak_reduction'],
                'peak_reduction_percent': best_result['scenario_metrics']['peak_reduction_percent']
            },
            'timestamps': [ts.isoformat() for ts in timestamps],
            'total_load': total_load.tolist(),
            'battery_soc': avg_soc.tolist(),
            'grid_exchange': total_grid.tolist(),
            'pv_generation': total_pv.tolist(),
            'building_savings': [response['total_cost'] for response in building_responses.values()],
            'fairness_analysis': {
                'gini_coefficient': best_result['scenario_metrics']['fairness_gini'],
                'coefficient_of_variation': best_result['scenario_metrics']['fairness_cv'],
                'cost_range': best_result['scenario_metrics']['building_cost_range']
            },
            'tariff_comparison': {tariff: result['scenario_metrics'] for tariff, result in tariff_results.items()}
        }

def main():
    print("=== Prosumer Community Simulation ===")
    print()
    
    community = SimpleCommunity(
        num_buildings=20,
        residents_per_building=25,
        pv_capacity=12,
        battery_capacity=30,
        tariff_types=['tou', 'cpp', 'rtp']
    )
    
    print(f"Community initialized:")
    print(f"- Buildings: {community.num_buildings}")
    print(f"- Residents per building: {community.residents_per_building}")
    print(f"- PV capacity per building: {community.pv_capacity} kW")
    print(f"- Battery capacity per building: {community.battery_capacity} kWh")
    print(f"- Tariff types: {', '.join(community.tariff_types)}")
    print()
    
    sim = SimpleCommunitySimulation(community)
    
    results = sim.run_simulation(
        duration_days=7,
        comfort_threshold=0.8,
        price_sensitivity=1.0
    )
    
    print("\n=== SIMULATION RESULTS ===")
    summary = results['simulation_summary']
    print(f"Best Tariff: {summary['best_tariff']}")
    print(f"Total Buildings: {summary['total_buildings']}")
    print(f"Duration: {summary['simulation_duration_hours'] // 24} days")
    print(f"Baseline Cost: €{summary['baseline_cost']:.2f}")
    print(f"Optimized Cost: €{summary['optimized_cost']:.2f}")
    print(f"Total Savings: €{summary['total_savings']:.2f}")
    print(f"Savings Percentage: {summary['savings_percent']:.1f}%")
    print(f"Peak Reduction: {summary['peak_reduction']:.2f} kW ({summary['peak_reduction_percent']:.1f}%)")
    
    print("\n=== FAIRNESS ANALYSIS ===")
    fairness = results['fairness_analysis']
    print(f"Gini Coefficient: {fairness['gini_coefficient']:.3f}")
    print(f"Coefficient of Variation: {fairness['coefficient_of_variation']:.3f}")
    print(f"Cost Range: €{fairness['cost_range']:.2f}")
    
    print("\n=== TARIFF COMPARISON ===")
    for tariff, metrics in results['tariff_comparison'].items():
        print(f"{tariff.upper()}: Cost=€{metrics['total_cost']:.2f}, Savings={metrics['cost_savings_percent']:.1f}%")
    
    print(f"\nSimulation completed successfully!")
    print(f"Peak community load: {max(results['total_load']):.2f} kW")
    print(f"Total PV generation: {sum(results['pv_generation']):.2f} kWh")

if __name__ == "__main__":
    main()
