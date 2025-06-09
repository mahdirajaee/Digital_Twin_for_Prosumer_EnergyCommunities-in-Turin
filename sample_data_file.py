import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_load_data(duration_days=30, filename="sample_load_data.csv"):
    
    hours = duration_days * 24
    start_date = datetime(2024, 1, 1)
    timestamps = pd.date_range(start=start_date, periods=hours, freq='H')
    
    load_profile = []
    
    for timestamp in timestamps:
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  
        month = timestamp.month
        
        base_load = generate_hourly_base_load(hour, day_of_week)
        seasonal_factor = generate_seasonal_factor(month)
        random_variation = np.random.normal(1.0, 0.1)
        
        final_load = base_load * seasonal_factor * random_variation
        final_load = max(0.5, final_load)  
        
        load_profile.append(final_load)
    
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Electricity_kWh': load_profile
    })
    
    if not os.path.exists('data'):
        os.makedirs('data')
    
    filepath = os.path.join('data', filename)
    df.to_csv(filepath, index=False)
    
    print(f"Sample load data generated: {filepath}")
    print(f"Duration: {duration_days} days ({hours} hours)")
    print(f"Peak load: {max(load_profile):.2f} kWh")
    print(f"Average load: {np.mean(load_profile):.2f} kWh")
    print(f"Load factor: {np.mean(load_profile)/max(load_profile):.2f}")
    
    return df

def generate_hourly_base_load(hour, day_of_week):
    
    if day_of_week >= 5:  
        weekend_factor = 1.1
        morning_shift = 1  
        evening_shift = -1  
    else:  
        weekend_factor = 1.0
        morning_shift = 0
        evening_shift = 0
    
    adjusted_hour = hour
    
    night_consumption = 2.0  
    
    morning_peak_hour = 8 + morning_shift
    morning_peak = 5.0 * np.exp(-0.5 * ((adjusted_hour - morning_peak_hour) / 1.5) ** 2)
    
    midday_consumption = 3.0 + 1.0 * np.sin(np.pi * (adjusted_hour - 12) / 6)
    midday_consumption = max(2.5, midday_consumption)
    
    evening_peak_hour = 19 + evening_shift
    evening_peak = 6.5 * np.exp(-0.5 * ((adjusted_hour - evening_peak_hour) / 2.0) ** 2)
    
    late_evening_decline = 0
    if adjusted_hour >= 22:
        decline_factor = (adjusted_hour - 22) / 2
        late_evening_decline = -2.0 * decline_factor
    
    total_load = night_consumption + morning_peak + midday_consumption + evening_peak + late_evening_decline
    
    total_load = max(1.5, total_load)
    
    total_load *= weekend_factor
    
    return total_load

def generate_seasonal_factor(month):
    
    if month in [6, 7, 8]:  
        return 1.3  
    elif month in [12, 1, 2]:  
        return 1.2  
    elif month in [3, 4, 5, 9, 10, 11]:  
        return 1.0  
    else:
        return 1.0

def generate_multiple_building_data(num_buildings=5, duration_days=30):
    
    all_building_data = {}
    
    for building_id in range(1, num_buildings + 1):
        
        np.random.seed(building_id * 42)  
        
        base_consumption_factor = np.random.uniform(0.8, 1.2)
        peak_time_variance = np.random.randint(-1, 2)
        load_pattern_factor = np.random.uniform(0.9, 1.1)
        
        hours = duration_days * 24
        start_date = datetime(2024, 1, 1)
        timestamps = pd.date_range(start=start_date, periods=hours, freq='H')
        
        load_profile = []
        
        for timestamp in timestamps:
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            month = timestamp.month
            
            base_load = generate_hourly_base_load(hour + peak_time_variance, day_of_week)
            base_load *= base_consumption_factor * load_pattern_factor
            
            seasonal_factor = generate_seasonal_factor(month)
            random_variation = np.random.normal(1.0, 0.08)
            
            final_load = base_load * seasonal_factor * random_variation
            final_load = max(0.5, final_load)
            
            load_profile.append(final_load)
        
        building_data = pd.DataFrame({
            'Timestamp': timestamps,
            f'Building_{building_id}_kWh': load_profile
        })
        
        all_building_data[f'Building_{building_id}'] = building_data
    
    combined_df = all_building_data['Building_1'].copy()
    for building_id in range(2, num_buildings + 1):
        building_key = f'Building_{building_id}'
        combined_df = combined_df.merge(
            all_building_data[building_key], 
            on='Timestamp', 
            how='outer'
        )
    
    combined_df['Total_Community_kWh'] = combined_df.iloc[:, 1:].sum(axis=1)
    
    if not os.path.exists('data'):
        os.makedirs('data')
    
    filepath = os.path.join('data', f'community_{num_buildings}_buildings_data.csv')
    combined_df.to_csv(filepath, index=False)
    
    print(f"Community data generated for {num_buildings} buildings: {filepath}")
    print(f"Total community peak: {combined_df['Total_Community_kWh'].max():.2f} kWh")
    print(f"Average community load: {combined_df['Total_Community_kWh'].mean():.2f} kWh")
    
    return combined_df

def generate_weather_sample_data(duration_days=30):
    
    hours = duration_days * 24
    start_date = datetime(2024, 1, 1)
    timestamps = pd.date_range(start=start_date, periods=hours, freq='H')
    
    weather_data = []
    
    for timestamp in timestamps:
        hour = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        
        daily_temp_cycle = 8 * np.sin(2 * np.pi * (hour - 6) / 24)
        seasonal_temp = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        base_temp = 20
        temp_noise = np.random.normal(0, 2)
        temperature = base_temp + daily_temp_cycle + seasonal_temp + temp_noise
        
        base_humidity = 60
        humidity_daily = 15 * np.sin(2 * np.pi * (hour - 14) / 24)
        humidity_noise = np.random.normal(0, 5)
        humidity = base_humidity + humidity_daily + humidity_noise
        humidity = np.clip(humidity, 20, 90)
        
        solar_angle = max(0, np.sin(np.pi * (hour - 6) / 12))
        cloud_factor = np.random.beta(2, 3)  
        solar_irradiance = solar_angle * (1 - 0.8 * cloud_factor) * 1000
        
        wind_speed = np.random.gamma(2, 2)  
        wind_speed = np.clip(wind_speed, 0, 15)
        
        weather_data.append({
            'Timestamp': timestamp,
            'Temperature_C': round(temperature, 1),
            'Humidity_Percent': round(humidity, 1),
            'Solar_Irradiance_W_m2': round(solar_irradiance, 1),
            'Wind_Speed_m_s': round(wind_speed, 1),
            'Cloud_Cover': round(cloud_factor, 2)
        })
    
    weather_df = pd.DataFrame(weather_data)
    
    if not os.path.exists('data'):
        os.makedirs('data')
    
    filepath = os.path.join('data', 'sample_weather_data.csv')
    weather_df.to_csv(filepath, index=False)
    
    print(f"Sample weather data generated: {filepath}")
    print(f"Temperature range: {weather_df['Temperature_C'].min():.1f}°C to {weather_df['Temperature_C'].max():.1f}°C")
    print(f"Average solar irradiance: {weather_df['Solar_Irradiance_W_m2'].mean():.0f} W/m²")
    
    return weather_df

def create_all_sample_data():
    
    print("Generating comprehensive sample data...")
    print("=" * 50)
    
    print("\n1. Single building load data (30 days):")
    single_building = generate_sample_load_data(duration_days=30)
    
    print("\n2. Single building load data (7 days - for quick testing):")
    quick_test = generate_sample_load_data(duration_days=7, filename="quick_test_load_data.csv")
    
    print("\n3. Community data (5 buildings, 30 days):")
    community_small = generate_multiple_building_data(num_buildings=5, duration_days=30)
    
    print("\n4. Larger community data (20 buildings, 14 days):")
    community_large = generate_multiple_building_data(num_buildings=20, duration_days=14)
    
    print("\n5. Weather data (30 days):")
    weather_data = generate_weather_sample_data(duration_days=30)
    
    print("\n" + "=" * 50)
    print("Sample data generation complete!")
    print("\nFiles created in 'data/' directory:")
    print("- sample_load_data.csv (single building, 30 days)")
    print("- quick_test_load_data.csv (single building, 7 days)")
    print("- community_5_buildings_data.csv (5 buildings, 30 days)")
    print("- community_20_buildings_data.csv (20 buildings, 14 days)")
    print("- sample_weather_data.csv (weather data, 30 days)")
    print("\nThese files can be uploaded through the web interface for testing.")

if __name__ == "__main__":
    create_all_sample_data()