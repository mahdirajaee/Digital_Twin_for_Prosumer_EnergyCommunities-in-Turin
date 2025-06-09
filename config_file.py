import os
from typing import Dict, List

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///simulation.db')

SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')

DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'

MAX_BUILDINGS = 100
MIN_BUILDINGS = 10
DEFAULT_BUILDINGS = 50

MAX_RESIDENTS_PER_BUILDING = 50
MIN_RESIDENTS_PER_BUILDING = 10
DEFAULT_RESIDENTS_PER_BUILDING = 20

MAX_PV_CAPACITY_KW = 50
MIN_PV_CAPACITY_KW = 1
DEFAULT_PV_CAPACITY_KW = 10

MAX_BATTERY_CAPACITY_KWH = 100
MIN_BATTERY_CAPACITY_KWH = 5
DEFAULT_BATTERY_CAPACITY_KWH = 25

MAX_BUILDING_LOAD_KW = 15
MIN_BUILDING_LOAD_KW = 2

MAX_COMMUNITY_LOAD_MW = 10

MAX_SIMULATION_DURATION_DAYS = 365
MIN_SIMULATION_DURATION_DAYS = 1
DEFAULT_SIMULATION_DURATION_DAYS = 7

DEFAULT_COMFORT_THRESHOLD = 0.8
MIN_COMFORT_THRESHOLD = 0.5
MAX_COMFORT_THRESHOLD = 1.0

DEFAULT_PRICE_SENSITIVITY = 1.0
MIN_PRICE_SENSITIVITY = 0.1
MAX_PRICE_SENSITIVITY = 5.0

TARIFF_TYPES = {
    'tou': 'Time-of-Use',
    'cpp': 'Critical Peak Pricing',
    'rtp': 'Real-Time Pricing',
    'edr': 'Emergency Demand Response'
}

# Tariff rates in EUR/kWh for Turin, Italy electricity market
DEFAULT_TARIFF_RATES = {
    'tou': {
        'off_peak': 0.22,      # €0.22/kWh - Italian residential off-peak rate
        'mid_peak': 0.28,      # €0.28/kWh - Italian residential mid-peak rate
        'on_peak': 0.35,       # €0.35/kWh - Italian residential peak rate
        'off_peak_hours': [0, 1, 2, 3, 4, 5, 6, 22, 23],  # Extended off-peak for Italian market
        'on_peak_hours': [18, 19, 20, 21],                 # Italian peak consumption hours
        'weekend_discount': 0.85  # Weekend discount factor
    },
    'cpp': {
        'base_rate': 0.28,           # €0.28/kWh - Italian CPP base rate
        'critical_peak_rate': 0.65,  # €0.65/kWh - Critical peak rate
        'critical_threshold': 0.85,
        'max_events_per_month': 15,  # More frequent events in Italian market
        'event_duration_hours': 3    # Shorter duration events
    },
    'rtp': {
        'base_rate': 0.28,     # €0.28/kWh - Italian RTP base rate
        'min_rate': 0.15,      # €0.15/kWh - Minimum real-time rate
        'max_rate': 0.55,      # €0.55/kWh - Maximum real-time rate
        'volatility': 0.25,    # Lower volatility in European markets
        'demand_elasticity': 0.7
    },
    'edr': {
        'base_rate': 0.28,       # €0.28/kWh - Italian EDR base rate
        'incentive_rate': 0.35,  # €0.35/kWh - Demand response incentive
        'penalty_rate': 0.42,    # €0.42/kWh - Penalty rate for non-compliance
        'baseline_days': 10,
        'min_reduction_percent': 15  # Higher minimum reduction for Italian market
    }
}

PV_SYSTEM_DEFAULTS = {
    'efficiency': 0.20,
    'tilt_angle': 35.0,        # Optimized for Turin latitude (45°N) - typically latitude minus 10°
    'azimuth_angle': 180.0,    # South-facing for Northern Hemisphere
    'temperature_coefficient': -0.004,
    'inverter_efficiency': 0.95,
    'system_losses': 0.15
}

BATTERY_SYSTEM_DEFAULTS = {
    'efficiency': 0.95,
    'max_charge_rate_ratio': 0.5,
    'max_discharge_rate_ratio': 0.5,
    'min_soc': 0.1,
    'max_soc': 0.9,
    'initial_soc': 0.5,
    'cycle_life': 6000,
    'calendar_life_years': 15
}

HVAC_SYSTEM_DEFAULTS = {
    'cop_heating': 3.0,
    'cop_cooling': 4.0,
    'thermal_mass_factor': 10.0,
    'insulation_r_value': 20.0,
    'target_temperature': 22.0,
    'temperature_tolerance': 2.0,
    'setback_temperature': 3.0
}

FLEXIBLE_LOADS_DEFAULTS = {
    'ev_charging': {
        'power_kw': 7.2,
        'duration_hours': 4,
        'earliest_start': 18,
        'latest_end': 6,
        'priority': 2,
        'flexibility_factor': 0.8
    },
    'water_heater': {
        'power_kw': 3.0,
        'duration_hours': 2,
        'earliest_start': 0,
        'latest_end': 23,
        'priority': 1,
        'flexibility_factor': 0.6
    },
    'dishwasher': {
        'power_kw': 1.8,
        'duration_hours': 1.5,
        'earliest_start': 6,
        'latest_end': 22,
        'priority': 3,
        'flexibility_factor': 0.9
    },
    'washing_machine': {
        'power_kw': 2.0,
        'duration_hours': 1,
        'earliest_start': 6,
        'latest_end': 20,
        'priority': 3,
        'flexibility_factor': 0.9
    },
    'dryer': {
        'power_kw': 3.5,
        'duration_hours': 1,
        'earliest_start': 6,
        'latest_end': 22,
        'priority': 4,
        'flexibility_factor': 0.7
    }
}

# Location-specific settings for Turin, Italy
LOCATION_SETTINGS = {
    'city': 'Turin',
    'country': 'Italy',
    'latitude': 45.0703,     # Turin latitude for solar calculations
    'longitude': 7.6869,     # Turin longitude
    'timezone': 'Europe/Rome',
    'currency': 'EUR',
    'currency_symbol': '€'
}

# Weather defaults adjusted for Turin's continental climate
WEATHER_DEFAULTS = {
    'base_temperature_winter': 2.0,    # Turin winter average (°C)
    'base_temperature_summer': 25.0,   # Turin summer average (°C)
    'temperature_variance': 12.0,      # Higher variance for continental climate
    'base_humidity': 65.0,             # Turin average humidity
    'humidity_variance': 25.0,
    'cloud_cover_base': 0.5,           # Turin cloud cover
    'wind_speed_mean': 2.5,            # Turin average wind speed (m/s)
    'wind_speed_std': 1.8,
    'solar_peak_summer': 1100,         # Peak solar irradiance for Turin latitude
    'solar_peak_winter': 600           # Winter peak solar irradiance
}

OPTIMIZATION_SETTINGS = {
    'max_iterations': 50,
    'convergence_tolerance': 1e-3,
    'time_limit_seconds': 300,
    'population_size': 30,
    'crossover_rate': 0.8,
    'mutation_rate': 0.1
}

SIMULATION_SETTINGS = {
    'time_step_minutes': 60,
    'forecast_horizon_hours': 24,
    'rolling_optimization': True,
    'uncertainty_factor': 0.05,
    'demand_response_delay_minutes': 15
}

VISUALIZATION_SETTINGS = {
    'default_chart_height': 400,
    'default_chart_width': 800,
    'color_palette': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ],
    'plot_style': 'plotly_white',
    'font_family': 'Arial, sans-serif',
    'title_font_size': 16,
    'axis_font_size': 12
}

FAIRNESS_METRICS = {
    'gini_coefficient_threshold': 0.3,
    'cv_threshold': 0.25,
    'max_cost_ratio': 2.0,
    'fairness_weights': {
        'gini': 0.4,
        'cv': 0.3,
        'range': 0.3
    }
}

PERFORMANCE_THRESHOLDS = {
    'excellent_savings_percent': 20.0,
    'good_savings_percent': 10.0,
    'excellent_peak_reduction_percent': 15.0,
    'good_peak_reduction_percent': 8.0,
    'excellent_pv_utilization_percent': 80.0,
    'good_pv_utilization_percent': 60.0,
    'excellent_fairness_score': 0.8,
    'good_fairness_score': 0.6
}

API_SETTINGS = {
    'max_requests_per_minute': 60,
    'max_concurrent_simulations': 5,
    'simulation_timeout_minutes': 30,
    'data_retention_days': 90
}

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'simulation.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

DASH_STYLES = {
    'external_stylesheets': [
        'https://codepen.io/chriddyp/pen/bWLwgP.css',
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
    ],
    'custom_css': {
        '.header': {
            'text-align': 'center',
            'color': '#2c3e50',
            'margin-bottom': '30px',
            'font-family': 'Arial, sans-serif'
        },
        '.upload-section': {
            'margin': '20px 0',
            'padding': '20px',
            'border': '1px solid #ddd',
            'border-radius': '5px',
            'background-color': '#f9f9f9'
        },
        '.parameters-section': {
            'margin': '20px 0',
            'padding': '20px',
            'border': '1px solid #ddd',
            'border-radius': '5px'
        },
        '.tariff-section': {
            'margin': '20px 0',
            'padding': '20px',
            'border': '1px solid #ddd',
            'border-radius': '5px'
        },
        '.simulation-controls': {
            'margin': '20px 0',
            'padding': '20px',
            'border': '1px solid #ddd',
            'border-radius': '5px'
        },
        '.results-section': {
            'margin': '20px 0',
            'padding': '20px',
            'border': '1px solid #ddd',
            'border-radius': '5px'
        },
        '.analysis-section': {
            'margin': '20px 0',
            'padding': '20px',
            'border': '1px solid #ddd',
            'border-radius': '5px'
        },
        '.button-primary': {
            'background-color': '#3498db',
            'color': 'white',
            'padding': '10px 20px',
            'border': 'none',
            'border-radius': '5px',
            'cursor': 'pointer',
            'font-size': '16px',
            'margin': '10px 0'
        },
        '.button-primary:hover': {
            'background-color': '#2980b9'
        }
    }
}

DATA_VALIDATION_RULES = {
    'load_profile': {
        'min_hours': 24,
        'max_hours': 8760,
        'min_value_kw': 0,
        'max_value_kw': 100,
        'required_columns': ['electricity'],
        'allowed_file_types': ['.csv', '.xlsx', '.xls']
    },
    'weather_data': {
        'temperature_range': (-40, 50),
        'humidity_range': (0, 100),
        'wind_speed_range': (0, 50),
        'solar_irradiance_range': (0, 1200)
    }
}

ERROR_MESSAGES = {
    'invalid_file_format': 'Unsupported file format. Please upload CSV or Excel files.',
    'missing_data_columns': 'Required data columns are missing from the uploaded file.',
    'invalid_parameter_range': 'Parameter value is outside the allowed range.',
    'simulation_timeout': 'Simulation timed out. Please try with fewer buildings or shorter duration.',
    'optimization_failed': 'Optimization failed to converge. Results may be suboptimal.',
    'insufficient_data': 'Insufficient data for meaningful analysis.',
    'memory_limit_exceeded': 'Memory limit exceeded. Please reduce simulation complexity.'
}

SUCCESS_MESSAGES = {
    'file_uploaded': 'File uploaded successfully.',
    'community_initialized': 'Community initialized successfully.',
    'simulation_completed': 'Simulation completed successfully.',
    'results_exported': 'Results exported successfully.',
    'optimization_converged': 'Optimization converged successfully.'
}

def get_config_value(key: str, default=None):
    
    return globals().get(key, default)

def validate_config():
    
    required_settings = [
        'MAX_BUILDINGS', 'DEFAULT_TARIFF_RATES', 'PV_SYSTEM_DEFAULTS',
        'BATTERY_SYSTEM_DEFAULTS', 'OPTIMIZATION_SETTINGS'
    ]
    
    missing = []
    for setting in required_settings:
        if setting not in globals():
            missing.append(setting)
    
    if missing:
        raise ValueError(f"Missing required configuration settings: {missing}")
    
    if MAX_BUILDINGS <= MIN_BUILDINGS:
        raise ValueError("MAX_BUILDINGS must be greater than MIN_BUILDINGS")
    
    if DEFAULT_BUILDINGS < MIN_BUILDINGS or DEFAULT_BUILDINGS > MAX_BUILDINGS:
        raise ValueError("DEFAULT_BUILDINGS must be within MIN_BUILDINGS and MAX_BUILDINGS range")
    
    return True

def get_tariff_config(tariff_type: str) -> Dict:
    
    return DEFAULT_TARIFF_RATES.get(tariff_type, {})

def get_system_limits() -> Dict:
    
    return {
        'buildings': {'min': MIN_BUILDINGS, 'max': MAX_BUILDINGS, 'default': DEFAULT_BUILDINGS},
        'residents': {'min': MIN_RESIDENTS_PER_BUILDING, 'max': MAX_RESIDENTS_PER_BUILDING, 'default': DEFAULT_RESIDENTS_PER_BUILDING},
        'pv_capacity': {'min': MIN_PV_CAPACITY_KW, 'max': MAX_PV_CAPACITY_KW, 'default': DEFAULT_PV_CAPACITY_KW},
        'battery_capacity': {'min': MIN_BATTERY_CAPACITY_KWH, 'max': MAX_BATTERY_CAPACITY_KWH, 'default': DEFAULT_BATTERY_CAPACITY_KWH},
        'simulation_duration': {'min': MIN_SIMULATION_DURATION_DAYS, 'max': MAX_SIMULATION_DURATION_DAYS, 'default': DEFAULT_SIMULATION_DURATION_DAYS},
        'comfort_threshold': {'min': MIN_COMFORT_THRESHOLD, 'max': MAX_COMFORT_THRESHOLD, 'default': DEFAULT_COMFORT_THRESHOLD},
        'price_sensitivity': {'min': MIN_PRICE_SENSITIVITY, 'max': MAX_PRICE_SENSITIVITY, 'default': DEFAULT_PRICE_SENSITIVITY}
    }

if __name__ == "__main__":
    try:
        validate_config()
        print("Configuration validation passed!")
        print(f"System limits: {get_system_limits()}")
    except ValueError as e:
        print(f"Configuration validation failed: {e}")