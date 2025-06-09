import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import asdict

from models import Community, Building
from tariffs import TariffManager
from optimization import CommunityOptimizer, TariffOptimizer, SensitivityAnalyzer, OptimizationResult
from utils import calculate_fairness_metrics, calculate_peak_reduction, generate_weather_data
import config

class CommunitySimulation:
    def __init__(self, community: Community):
        self.community = community
        self.tariff_manager = TariffManager()
        self.results_history = []
        self.baseline_established = False
        self.baseline_metrics = {}
        
    def run_simulation(self, duration_days: int = 7, 
                      comfort_threshold: float = 0.8,
                      price_sensitivity: float = 1.0,
                      weather_data: Optional[pd.DataFrame] = None) -> Dict:
        
        if not self.baseline_established:
            self.establish_baseline(duration_days, weather_data)
        
        simulation_results = {}
        total_hours = duration_days * 24
        
        timestamps = pd.date_range('2024-01-01', periods=total_hours, freq='H')
        
        if weather_data is None:
            weather_data = generate_weather_data(timestamps)
        
        tariff_results = {}
        
        for tariff_type in self.community.tariff_types:
            print(f"Running simulation for {tariff_type} tariff...")
            
            tariff_result = self.simulate_tariff_scenario(
                tariff_type=tariff_type,
                timestamps=timestamps,
                weather_data=weather_data,
                comfort_threshold=comfort_threshold,
                price_sensitivity=price_sensitivity
            )
            
            tariff_results[tariff_type] = tariff_result
        
        best_tariff = min(tariff_results.keys(), 
                         key=lambda t: tariff_results[t]['total_cost'])
        
        simulation_results = self.compile_final_results(
            tariff_results, best_tariff, timestamps
        )
        
        self.results_history.append(simulation_results)
        
        return simulation_results
    
    def establish_baseline(self, duration_days: int, weather_data: Optional[pd.DataFrame] = None):
        
        baseline_hours = min(duration_days * 24, 168)  
        timestamps = pd.date_range('2024-01-01', periods=baseline_hours, freq='H')
        
        if weather_data is None:
            weather_data = generate_weather_data(timestamps)
        
        baseline_load = np.zeros(baseline_hours)
        baseline_cost = 0
        
        flat_rate = 0.25  # €0.25/kWh - average Italian electricity rate for baseline  
        
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
        print(f"Baseline established: Peak={self.baseline_metrics['peak_load']:.2f}kW, "
              f"Cost=€{self.baseline_metrics['total_cost']:.2f}")
    
    def simulate_tariff_scenario(self, tariff_type: str, timestamps: pd.DatetimeIndex,
                               weather_data: pd.DataFrame, comfort_threshold: float,
                               price_sensitivity: float) -> Dict:
        
        tariff = self.tariff_manager.get_tariff(tariff_type)
        community_profile = self.community.get_community_profile(len(timestamps))
        
        tariff_prices = tariff.get_price_schedule(timestamps, community_profile['load_profile'])
        
        optimizer = CommunityOptimizer(self.community)
        optimization_result = optimizer.bilevel_optimization(
            num_hours=len(timestamps),
            comfort_threshold=comfort_threshold,
            price_sensitivity=price_sensitivity
        )
        
        building_responses = self.simulate_building_responses(
            optimization_result, timestamps, weather_data, tariff_prices
        )
        
        scenario_metrics = self.calculate_scenario_metrics(
            building_responses, optimization_result, tariff_prices, timestamps
        )
        
        return {
            'tariff_type': tariff_type,
            'optimization_result': optimization_result,
            'building_responses': building_responses,
            'scenario_metrics': scenario_metrics,
            'tariff_prices': tariff_prices.tolist(),
            'timestamps': timestamps.tolist()
        }
    
    def simulate_building_responses(self, optimization_result: OptimizationResult,
                                  timestamps: pd.DatetimeIndex, weather_data: pd.DataFrame,
                                  tariff_prices: np.ndarray) -> Dict:
        
        building_responses = {}
        
        for building in self.community.buildings:
            building_id = building.building_id
            
            load_schedule = optimization_result.load_schedule.get(building_id, np.zeros(len(timestamps)))
            battery_schedule = optimization_result.battery_schedule.get(building_id, np.zeros(len(timestamps)))
            
            pv_generation = building.pv_system.generate_profile(timestamps, weather_data)
            
            building_response = self.simulate_single_building_response(
                building=building,
                load_schedule=load_schedule,
                battery_schedule=battery_schedule,
                pv_generation=pv_generation,
                weather_data=weather_data,
                tariff_prices=tariff_prices
            )
            
            building_responses[building_id] = building_response
        
        return building_responses
    
    def simulate_single_building_response(self, building: Building, load_schedule: np.ndarray,
                                        battery_schedule: np.ndarray, pv_generation: np.ndarray,
                                        weather_data: pd.DataFrame, tariff_prices: np.ndarray) -> Dict:
        
        num_hours = len(load_schedule)
        
        actual_load = np.zeros(num_hours)
        actual_battery = np.zeros(num_hours)
        grid_exchange = np.zeros(num_hours)
        energy_costs = np.zeros(num_hours)
        soc_profile = np.zeros(num_hours)
        temperature_profile = np.zeros(num_hours)
        
        current_soc = building.current_soc
        current_temp = building.current_temp
        
        for hour in range(num_hours):
            outdoor_temp = weather_data.iloc[hour]['temperature'] if 'temperature' in weather_data.columns else 20.0
            
            actual_load[hour] = load_schedule[hour] * (1 + np.random.normal(0, 0.05))
            
            max_discharge, max_charge = building.battery_system.get_charge_limits(current_soc)
            actual_battery[hour] = np.clip(battery_schedule[hour], -max_discharge, max_charge)
            
            net_load = actual_load[hour] - pv_generation[hour] - actual_battery[hour]
            grid_exchange[hour] = net_load
            
            energy_costs[hour] = max(0, grid_exchange[hour]) * tariff_prices[hour]
            
            current_soc = building.battery_system.update_soc(current_soc, actual_battery[hour])
            soc_profile[hour] = current_soc
            
            hvac_power = building.hvac_schedule.get(hour, 0)
            current_temp = building.hvac_system.update_temperature(current_temp, outdoor_temp, hvac_power)
            temperature_profile[hour] = current_temp
        
        return {
            'actual_load': actual_load,
            'actual_battery': actual_battery,
            'pv_generation': pv_generation,
            'grid_exchange': grid_exchange,
            'energy_costs': energy_costs,
            'soc_profile': soc_profile,
            'temperature_profile': temperature_profile,
            'total_cost': np.sum(energy_costs),
            'peak_load': np.max(actual_load),
            'pv_utilization': np.sum(np.minimum(pv_generation, actual_load)) / (np.sum(pv_generation) + 1e-6),
            'battery_cycles': self.calculate_battery_cycles(actual_battery)
        }
    
    def calculate_battery_cycles(self, battery_schedule: np.ndarray) -> float:
        
        discharge_energy = np.sum(np.abs(battery_schedule[battery_schedule < 0]))
        return discharge_energy / (self.community.battery_capacity * 2)  
    
    def calculate_scenario_metrics(self, building_responses: Dict, optimization_result: OptimizationResult,
                                 tariff_prices: np.ndarray, timestamps: pd.DatetimeIndex) -> Dict:
        
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
            'average_price': np.mean(tariff_prices),
            'price_volatility': np.std(tariff_prices),
            'total_pv_generation': np.sum(total_pv),
            'pv_penetration': np.sum(total_pv) / np.sum(total_load) * 100,
            'grid_import': np.sum(total_grid[total_grid > 0]),
            'grid_export': np.sum(abs(total_grid[total_grid < 0])),
            'self_consumption': (np.sum(total_pv) - np.sum(abs(total_grid[total_grid < 0]))) / np.sum(total_pv) * 100,
            'fairness_gini': fairness_metrics['gini_coefficient'],
            'fairness_cv': fairness_metrics['coefficient_of_variation'],
            'building_cost_range': max(building_costs) - min(building_costs),
            'convergence_iterations': optimization_result.convergence_iterations
        }
    
    def compile_final_results(self, tariff_results: Dict, best_tariff: str, 
                            timestamps: pd.DatetimeIndex) -> Dict:
        
        best_result = tariff_results[best_tariff]
        
        community_totals = self.aggregate_community_results(tariff_results, timestamps)
        
        comparison_data = self.create_tariff_comparison(tariff_results)
        
        final_results = {
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
            
            'cost_breakdown': self.create_cost_breakdown(best_result),
            
            'timestamps': [ts.isoformat() for ts in timestamps],
            'total_load': community_totals['total_load'].tolist(),
            'battery_soc': community_totals['average_soc'].tolist(),
            'grid_exchange': community_totals['total_grid'].tolist(),
            'pv_generation': community_totals['total_pv'].tolist(),
            
            'building_savings': [tariff_results[best_tariff]['building_responses'][bid]['total_cost'] 
                               for bid in range(self.community.num_buildings)],
            
            'peak_analysis': {
                'before': self.baseline_metrics['peak_load'],
                'after': best_result['scenario_metrics']['peak_load'],
                'reduction': best_result['scenario_metrics']['peak_reduction']
            },
            
            'tariff_comparison': comparison_data,
            
            'fairness_analysis': {
                'gini_coefficient': best_result['scenario_metrics']['fairness_gini'],
                'coefficient_of_variation': best_result['scenario_metrics']['fairness_cv'],
                'cost_range': best_result['scenario_metrics']['building_cost_range']
            },
            
            'sensitivity_analysis': self.run_sensitivity_analysis(),
            
            'performance_metrics': {
                'pv_utilization': best_result['scenario_metrics']['pv_penetration'],
                'self_consumption': best_result['scenario_metrics']['self_consumption'],
                'grid_independence': (1 - best_result['scenario_metrics']['grid_import'] / 
                                    (best_result['scenario_metrics']['grid_import'] + 
                                     sum(community_totals['total_pv']))) * 100,
                'convergence_iterations': best_result['scenario_metrics']['convergence_iterations']
            }
        }
        
        return final_results
    
    def aggregate_community_results(self, tariff_results: Dict, timestamps: pd.DatetimeIndex) -> Dict:
        
        best_tariff = min(tariff_results.keys(), 
                         key=lambda t: tariff_results[t]['scenario_metrics']['total_cost'])
        
        building_responses = tariff_results[best_tariff]['building_responses']
        
        num_hours = len(timestamps)
        total_load = np.zeros(num_hours)
        total_pv = np.zeros(num_hours)
        total_grid = np.zeros(num_hours)
        total_soc = np.zeros(num_hours)
        
        for response in building_responses.values():
            total_load += response['actual_load']
            total_pv += response['pv_generation']
            total_grid += response['grid_exchange']
            total_soc += response['soc_profile']
        
        average_soc = total_soc / len(building_responses)
        
        return {
            'total_load': total_load,
            'total_pv': total_pv,
            'total_grid': total_grid,
            'average_soc': average_soc
        }
    
    def create_cost_breakdown(self, best_result: Dict) -> List[Dict]:
        
        building_responses = best_result['building_responses']
        
        cost_data = []
        for building_id, response in building_responses.items():
            baseline_cost = self.baseline_metrics['total_cost'] / self.community.num_buildings
            
            cost_data.append({
                'Building': f"Building {building_id + 1}",
                'Baseline_Cost': f"€{baseline_cost:.2f}",
                'Optimized_Cost': f"€{response['total_cost']:.2f}",
                'Savings': f"€{baseline_cost - response['total_cost']:.2f}",
                'Savings_Percent': f"{((baseline_cost - response['total_cost']) / baseline_cost) * 100:.1f}%",
                'Peak_Load': f"{response['peak_load']:.2f} kW",
                'PV_Utilization': f"{response['pv_utilization'] * 100:.1f}%"
            })
        
        return cost_data
    
    def create_tariff_comparison(self, tariff_results: Dict) -> Dict:
        
        comparison = {}
        
        for tariff_type, result in tariff_results.items():
            metrics = result['scenario_metrics']
            comparison[tariff_type] = {
                'total_cost': metrics['total_cost'],
                'cost_savings': metrics['cost_savings'],
                'peak_reduction': metrics['peak_reduction'],
                'fairness_score': 1 - metrics['fairness_gini'],  
                'overall_score': self.calculate_overall_score(metrics)
            }
        
        return comparison
    
    def calculate_overall_score(self, metrics: Dict) -> float:
        
        cost_score = metrics['cost_savings_percent'] / 100
        peak_score = metrics['peak_reduction_percent'] / 100
        fairness_score = 1 - metrics['fairness_gini']
        
        weights = {'cost': 0.5, 'peak': 0.3, 'fairness': 0.2}
        
        overall_score = (weights['cost'] * cost_score + 
                        weights['peak'] * peak_score + 
                        weights['fairness'] * fairness_score)
        
        return max(0, min(1, overall_score))
    
    def run_sensitivity_analysis(self) -> Dict:
        
        analyzer = SensitivityAnalyzer(self.community)
        
        price_sensitivity = analyzer.analyze_price_sensitivity(
            sensitivity_range=(0.5, 2.0), num_points=5
        )
        
        comfort_sensitivity = analyzer.analyze_comfort_sensitivity(
            comfort_range=(0.6, 1.0), num_points=5
        )
        
        return {
            'price_sensitivity': {str(k): v for k, v in price_sensitivity.items()},
            'comfort_sensitivity': {str(k): v for k, v in comfort_sensitivity.items()}
        }
    
    def export_results(self, filename: Optional[str] = None) -> str:
        
        if not self.results_history:
            return "No simulation results to export"
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"community_simulation_results_{timestamp}.json"
        
        export_data = {
            'metadata': {
                'community_size': self.community.num_buildings,
                'residents_per_building': self.community.residents_per_building,
                'pv_capacity': self.community.pv_capacity,
                'battery_capacity': self.community.battery_capacity,
                'simulation_count': len(self.results_history),
                'export_timestamp': datetime.now().isoformat()
            },
            'baseline_metrics': self.baseline_metrics,
            'simulation_results': self.results_history
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            return f"Results exported to {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"
    
    def get_summary_statistics(self) -> Dict:
        
        if not self.results_history:
            return {}
        
        latest_result = self.results_history[-1]
        
        return {
            'average_savings': latest_result['simulation_summary']['savings_percent'],
            'peak_reduction': latest_result['simulation_summary']['peak_reduction_percent'],
            'fairness_score': 1 - latest_result['fairness_analysis']['gini_coefficient'],
            'best_tariff': latest_result['simulation_summary']['best_tariff'],
            'pv_utilization': latest_result['performance_metrics']['pv_utilization'],
            'grid_independence': latest_result['performance_metrics']['grid_independence']
        }