import numpy as np
import pandas as pd
import base64
import io
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import config_file

def parse_uploaded_file(contents: str, filename: str) -> pd.DataFrame:
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xlsx' in filename.lower() or 'xls' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        df = process_electricity_data(df)
        return df
        
    except Exception as e:
        raise ValueError(f"Error parsing file {filename}: {str(e)}")

def process_electricity_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df.columns = df.columns.str.strip()
    
    timestamp_columns = ['Time', 'Timestamp', 'DateTime', 'Date']
    electricity_columns = ['Electricity', 'Power', 'Load', 'Consumption', 'kWh', 'kW']
    
    timestamp_col = None
    electricity_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if any(ts_name.lower() in col_lower for ts_name in timestamp_columns):
            timestamp_col = col
            break
    
    for col in df.columns:
        col_lower = col.lower()
        if any(elec_name.lower() in col_lower for elec_name in electricity_columns):
            electricity_col = col
            break
    
    if timestamp_col is None and len(df.columns) >= 2:
        timestamp_col = df.columns[0]
    if electricity_col is None and len(df.columns) >= 2:
        electricity_col = df.columns[-1]
    
    if timestamp_col and timestamp_col in df.columns:
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df.set_index(timestamp_col, inplace=True)
        except:
            df.index = pd.date_range('2024-01-01', periods=len(df), freq='H')
    else:
        df.index = pd.date_range('2024-01-01', periods=len(df), freq='H')
    
    if electricity_col and electricity_col in df.columns:
        df['electricity'] = pd.to_numeric(df[electricity_col], errors='coerce')
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df['electricity'] = df[numeric_cols[0]]
        else:
            df['electricity'] = np.random.uniform(1, 10, len(df))
    
    df['electricity'] = df['electricity'].fillna(df['electricity'].mean())
    
    df = df[['electricity']]
    
    return df

def load_electricity_data(filepath: str = None) -> pd.DataFrame:
    
    if filepath:
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                raise ValueError("Unsupported file format")
            
            return process_electricity_data(df)
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
    
    hours = 8760  
    timestamps = pd.date_range('2024-01-01', periods=hours, freq='H')
    
    hourly_pattern = generate_daily_load_pattern()
    seasonal_factor = generate_seasonal_pattern(timestamps)
    
    base_load = np.tile(hourly_pattern, 365)[:hours]
    seasonal_load = base_load * seasonal_factor
    
    noise = np.random.normal(1.0, 0.1, hours)
    final_load = seasonal_load * noise
    
    df = pd.DataFrame({
        'electricity': final_load
    }, index=timestamps)
    
    return df

def generate_daily_load_pattern() -> np.ndarray:
    
    hours = np.arange(24)
    
    night_base = 0.3
    morning_peak = 0.8 * np.exp(-0.5 * ((hours - 8) / 2) ** 2)
    afternoon_dip = -0.2 * np.exp(-0.5 * ((hours - 14) / 2) ** 2)
    evening_peak = 1.0 * np.exp(-0.5 * ((hours - 19) / 3) ** 2)
    
    pattern = night_base + morning_peak + afternoon_dip + evening_peak
    
    pattern = np.maximum(pattern, 0.2)
    pattern = pattern / np.max(pattern)
    
    return pattern

def generate_seasonal_pattern(timestamps: pd.DatetimeIndex) -> np.ndarray:
    
    day_of_year = timestamps.dayofyear
    
    summer_peak = 0.3 * np.sin(2 * np.pi * (day_of_year - 172) / 365)  
    winter_heating = 0.2 * np.cos(2 * np.pi * (day_of_year - 1) / 365)   
    
    seasonal_factor = 1.0 + summer_peak + winter_heating
    
    seasonal_factor = np.maximum(seasonal_factor, 0.7)
    
    return seasonal_factor

def generate_weather_data(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate weather data for Turin, Italy climate patterns"""
    hours = timestamps.hour
    days = timestamps.dayofyear
    
    # Turin-specific temperature patterns
    # Daily variation: larger in continental climate
    daily_temp_variation = 10 * np.sin(2 * np.pi * (hours - 6) / 24)
    
    # Seasonal variation: more pronounced for Turin (continental climate)
    # Winter average: 2°C, Summer average: 25°C
    winter_temp = 2.0
    summer_temp = 25.0
    seasonal_temp_variation = (summer_temp - winter_temp) / 2 * np.sin(2 * np.pi * (days - 80) / 365)
    base_temperature = (winter_temp + summer_temp) / 2  # 13.5°C average
    temp_noise = np.random.normal(0, 3, len(timestamps))  # Higher variance
    
    temperature = base_temperature + daily_temp_variation + seasonal_temp_variation + temp_noise
    
    # Turin humidity patterns (higher due to Po Valley location)
    base_humidity = 65
    humidity_variation = 25 * np.sin(2 * np.pi * (hours - 12) / 24)
    seasonal_humidity = 15 * np.sin(2 * np.pi * (days - 172) / 365)
    humidity_noise = np.random.normal(0, 8, len(timestamps))
    
    humidity = base_humidity + humidity_variation + seasonal_humidity + humidity_noise
    humidity = np.clip(humidity, 30, 95)
    
    # Turin cloud cover (more cloudy in winter)
    cloud_base = 0.5  # Higher base cloud cover
    cloud_variation = 0.2 * np.sin(2 * np.pi * (hours - 14) / 24)
    seasonal_cloud = 0.2 * np.cos(2 * np.pi * (days - 1) / 365)  # More clouds in winter
    cloud_noise = np.random.beta(2, 3, len(timestamps)) * 0.3
    
    cloud_cover = cloud_base + cloud_variation + seasonal_cloud + cloud_noise
    cloud_cover = np.clip(cloud_cover, 0, 1)
    
    # Solar irradiance adjusted for Turin latitude (45.07°N)
    # Lower peak irradiance due to higher latitude
    base_solar = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    
    # Seasonal solar variation (much lower in winter)
    seasonal_solar_factor = 0.6 + 0.4 * np.sin(2 * np.pi * (days - 172) / 365)
    peak_irradiance = 900  # Lower peak than equatorial regions
    
    solar_irradiance = base_solar * seasonal_solar_factor * (1 - 0.8 * cloud_cover) * peak_irradiance
    
    # Turin wind patterns (lower average wind speed)
    wind_speed = np.random.gamma(1.5, 1.8, len(timestamps))  # Lower wind speeds
    wind_speed = np.clip(wind_speed, 0, 15)
    
    weather_df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'cloud_cover': cloud_cover,
        'solar_irradiance': solar_irradiance,
        'wind_speed': wind_speed
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

def calculate_peak_reduction(baseline_load: np.ndarray, optimized_load: np.ndarray) -> Dict[str, float]:
    
    baseline_peak = np.max(baseline_load)
    optimized_peak = np.max(optimized_load)
    
    absolute_reduction = baseline_peak - optimized_peak
    percentage_reduction = (absolute_reduction / baseline_peak) * 100 if baseline_peak > 0 else 0
    
    baseline_avg = np.mean(baseline_load)
    optimized_avg = np.mean(optimized_load)
    
    baseline_load_factor = baseline_avg / baseline_peak if baseline_peak > 0 else 0
    optimized_load_factor = optimized_avg / optimized_peak if optimized_peak > 0 else 0
    
    return {
        'baseline_peak': baseline_peak,
        'optimized_peak': optimized_peak,
        'absolute_reduction': absolute_reduction,
        'percentage_reduction': percentage_reduction,
        'baseline_load_factor': baseline_load_factor,
        'optimized_load_factor': optimized_load_factor,
        'load_factor_improvement': optimized_load_factor - baseline_load_factor
    }

def calculate_energy_metrics(load_profile: np.ndarray, pv_profile: np.ndarray, 
                           battery_schedule: np.ndarray) -> Dict[str, float]:
    
    total_load = np.sum(load_profile)
    total_pv = np.sum(pv_profile)
    
    direct_pv_consumption = np.sum(np.minimum(load_profile, pv_profile))
    
    battery_charging = np.sum(battery_schedule[battery_schedule > 0])
    battery_discharging = np.sum(abs(battery_schedule[battery_schedule < 0]))
    
    grid_import = np.sum(np.maximum(0, load_profile - pv_profile - battery_schedule))
    grid_export = np.sum(np.maximum(0, pv_profile - load_profile + battery_schedule))
    
    self_consumption_rate = (direct_pv_consumption + battery_discharging) / total_pv if total_pv > 0 else 0
    self_sufficiency_rate = (direct_pv_consumption + battery_discharging) / total_load if total_load > 0 else 0
    
    return {
        'total_load': total_load,
        'total_pv_generation': total_pv,
        'direct_pv_consumption': direct_pv_consumption,
        'battery_charging': battery_charging,
        'battery_discharging': battery_discharging,
        'grid_import': grid_import,
        'grid_export': grid_export,
        'self_consumption_rate': self_consumption_rate,
        'self_sufficiency_rate': self_sufficiency_rate,
        'pv_utilization': total_pv / total_load if total_load > 0 else 0
    }

def calculate_battery_metrics(soc_profile: np.ndarray, battery_schedule: np.ndarray, 
                            battery_capacity: float) -> Dict[str, float]:
    
    energy_charged = np.sum(battery_schedule[battery_schedule > 0])
    energy_discharged = np.sum(abs(battery_schedule[battery_schedule < 0]))
    
    full_cycles = energy_discharged / battery_capacity if battery_capacity > 0 else 0
    
    avg_soc = np.mean(soc_profile)
    min_soc = np.min(soc_profile)
    max_soc = np.max(soc_profile)
    
    soc_range = max_soc - min_soc
    
    charge_events = np.sum(np.diff(np.sign(battery_schedule)) > 0)
    discharge_events = np.sum(np.diff(np.sign(battery_schedule)) < 0)
    
    return {
        'energy_charged': energy_charged,
        'energy_discharged': energy_discharged,
        'full_cycles': full_cycles,
        'average_soc': avg_soc,
        'min_soc': min_soc,
        'max_soc': max_soc,
        'soc_range': soc_range,
        'charge_events': charge_events,
        'discharge_events': discharge_events,
        'roundtrip_efficiency': energy_discharged / energy_charged if energy_charged > 0 else 0
    }

def validate_simulation_inputs(community_params: Dict, simulation_params: Dict) -> List[str]:
    
    errors = []
    
    if community_params.get('num_buildings', 0) < 1:
        errors.append("Number of buildings must be at least 1")
    if community_params.get('num_buildings', 0) > 1000:
        errors.append("Number of buildings cannot exceed 1000")
    
    if community_params.get('residents_per_building', 0) < 1:
        errors.append("Residents per building must be at least 1")
    
    if community_params.get('pv_capacity', 0) <= 0:
        errors.append("PV capacity must be positive")
    if community_params.get('pv_capacity', 0) > 100:
        errors.append("PV capacity seems unreasonably high (>100kW)")
    
    if community_params.get('battery_capacity', 0) <= 0:
        errors.append("Battery capacity must be positive")
    if community_params.get('battery_capacity', 0) > 500:
        errors.append("Battery capacity seems unreasonably high (>500kWh)")
    
    if not community_params.get('tariff_types'):
        errors.append("At least one tariff type must be selected")
    
    comfort_threshold = simulation_params.get('comfort_threshold', 0.8)
    if not 0.5 <= comfort_threshold <= 1.0:
        errors.append("Comfort threshold must be between 0.5 and 1.0")
    
    price_sensitivity = simulation_params.get('price_sensitivity', 1.0)
    if not 0.1 <= price_sensitivity <= 5.0:
        errors.append("Price sensitivity must be between 0.1 and 5.0")
    
    duration = simulation_params.get('duration_days', 7)
    if not 1 <= duration <= 365:
        errors.append("Simulation duration must be between 1 and 365 days")
    
    return errors

def format_currency(amount: float) -> str:
    """Format amount as Euro currency for Turin, Italy"""
    return f"€{amount:,.2f}"

def format_energy(amount: float, unit: str = "kWh") -> str:
    
    if amount >= 1000:
        return f"{amount/1000:.2f} M{unit}"
    else:
        return f"{amount:.2f} {unit}"

def format_percentage(value: float) -> str:
    
    return f"{value:.1f}%"

def export_to_csv(data: Dict, filename: str) -> bool:
    
    try:
        if 'timestamps' in data and 'total_load' in data:
            df = pd.DataFrame({
                'timestamp': data['timestamps'],
                'total_load': data['total_load'],
                'pv_generation': data.get('pv_generation', [0] * len(data['timestamps'])),
                'grid_exchange': data.get('grid_exchange', [0] * len(data['timestamps'])),
                'battery_soc': data.get('battery_soc', [0] * len(data['timestamps']))
            })
            df.to_csv(filename, index=False)
            return True
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False

def create_summary_report(simulation_results: Dict) -> str:
    
    summary = simulation_results.get('simulation_summary', {})
    
    report = f"""
PROSUMER COMMUNITY SIMULATION REPORT
=====================================

Community Overview:
- Total Buildings: {summary.get('total_buildings', 'N/A')}
- Simulation Duration: {summary.get('simulation_duration_hours', 0) // 24} days

Financial Results:
- Baseline Cost: {format_currency(summary.get('baseline_cost', 0))}
- Optimized Cost: {format_currency(summary.get('optimized_cost', 0))}
- Total Savings: {format_currency(summary.get('total_savings', 0))}
- Savings Percentage: {format_percentage(summary.get('savings_percent', 0))}

Peak Load Management:
- Peak Reduction: {format_energy(summary.get('peak_reduction', 0), 'kW')}
- Peak Reduction Percentage: {format_percentage(summary.get('peak_reduction_percent', 0))}

Best Performing Tariff: {summary.get('best_tariff', 'N/A')}

Performance Metrics:
- PV Utilization: {format_percentage(simulation_results.get('performance_metrics', {}).get('pv_utilization', 0))}
- Grid Independence: {format_percentage(simulation_results.get('performance_metrics', {}).get('grid_independence', 0))}

Fairness Analysis:
- Gini Coefficient: {simulation_results.get('fairness_analysis', {}).get('gini_coefficient', 0):.3f}
- Cost Range: {format_currency(simulation_results.get('fairness_analysis', {}).get('cost_range', 0))}
"""
    
    return report