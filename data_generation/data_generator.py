import os
import datetime
import pandas as pd
import numpy as np
from ab_test_advanced_toolkit.analyzer import ABTestAnalyzer
from typing import Tuple
import random
import logging

logger = logging.getLogger(__name__)

# Create folder 'data/{now_timestamp_utc}' if it does not exist
folder_name = os.path.join(
    'data', 
    datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
)
os.makedirs(folder_name, exist_ok=True)

def describe_dataset(df):
    logger.debug("First 5 rows of the dataset:")
    logger.debug(df.head())
    
    logger.debug("\nStatistical summary of the dataset:")
    logger.debug(df.describe())
    
    logger.debug("\nCorrelation between 'value' and 'pre_test_value':")
    logger.debug(df[['value', 'pre_test_value']].corr())
    
    logger.debug("\nAverage 'value' by A/B groups:")
    ab_means = df.groupby('abgroup')['value'].mean()
    logger.debug(ab_means)

def generate_value(pre_test_value, country, platform, user_segment, base_increase_percentage, noise_level):
    # Initial value based on pre_test_value
    value = pre_test_value * (1 + base_increase_percentage)
    
    # Non-linear dependencies
    if country == 'US' and platform == 'iOS':
        value += np.sin(pre_test_value) * 5
    elif country == 'IN' and platform == 'Desktop' and pre_test_value > 0:
        value -= np.log1p(pre_test_value) * 3
    elif user_segment == 'Segment_2' and platform == 'Android' and pre_test_value >= 0:
        value += np.sqrt(pre_test_value) * 2
    else:
        value += np.random.normal(0, noise_level)
    
    # Add random noise
    value += np.random.normal(0, noise_level)
    
    return max(value, 0)  # Ensuring value is non-negative

def generate_synthetic_data(num_users=1000, countries=['US', 'UK', 'DE', 'FR', 'CA', 'AU', 'JP', 'IN'],
                            platforms=['iOS', 'Android', 'Web', 'Desktop'], user_segments=['Segment_1', 'Segment_2', 'Segment_3', 'Segment_4'],
                            ab_groups=['a1', 'a2', 'b'], noise_level=1.0, base_increase_percentage=0.2, seed=40):
    # Set seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Empty DataFrame for data
    data = {
        'userid': [],
        'country': [],
        'platform': [],
        'user_segment': [],
        'abgroup': [],
        # TODO
        # 'age': np.random.randint(18, 65, num_users),  # Age between 18 and 65
        # 'engagement_score': np.random.rand(num_users) * 10,  # Random score between 0 and 10
        'pre_test_value': [],
        'value': []
    }

    # TODO
    # # Calculate the base effect from features
    # base_value = 10 + df['age'] / 10 + df['engagement_score']
    
    for i in range(num_users):
        user_id = i + 1
        country = random.choice(countries)
        platform = random.choice(platforms)
        user_segment = random.choice(user_segments)
        ab_group = random.choice(ab_groups)
        
        # Generate pre_test_value
        pre_test_value = np.random.normal(5, 2)
        
        # Generate in_test_value with non-linear dependency
        in_test_value = generate_value(pre_test_value, country, platform, user_segment, base_increase_percentage, noise_level)
        
        data['userid'].append(user_id)
        data['country'].append(country)
        data['platform'].append(platform)
        data['user_segment'].append(user_segment)
        data['abgroup'].append(ab_group)
        data['pre_test_value'].append(pre_test_value)
        data['value'].append(in_test_value)
    
    df = pd.DataFrame(data)
    return df

def create_dataframes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_data1 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2022-12-10"] * len(df)),
        "userid": df["userid"],
        "event_name": ["purchase"] * len(df),
        "purchase_value": df["value"]
    })

    event_data2 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2022-12-01"] * len(df)),
        "userid": df["userid"],
        "event_name": ["purchase"] * len(df),
        "purchase_value": df["pre_test_value"]
    })

    user_allocations = pd.DataFrame({
        "timestamp": pd.to_datetime(["2022-12-05"] * len(df)),
        "userid": df["userid"],
        "abgroup": df["abgroup"]
    })

    user_properties = pd.DataFrame({
        "userid": df["userid"],
        # TODO

        # "age": df["age"],
        "country": df["country"],
        "device_type": df["platform"],
        "membership_status": ["Free"] * len(df)
    })

    event_data = pd.concat([event_data1, event_data2])
    event_data['timestamp'] = pd.to_datetime(event_data['timestamp'])

    return event_data, user_allocations, user_properties

def run_analysis(df: pd.DataFrame):
    event_data, user_allocations, user_properties = create_dataframes(df)
    
    # Analysis without enhancement
    analyzer_no_enhancement = ABTestAnalyzer(event_data, user_allocations, "a1", user_properties, mode="no_enhancement")
    results_no_enhancement = analyzer_no_enhancement.calculate_event_attribute_sum_per_user('purchase', 'purchase_value')

    # Analysis with CUPED
    analyzer_cuped = ABTestAnalyzer(event_data, user_allocations, "a1", user_properties, mode="cuped")
    results_cuped = analyzer_cuped.calculate_event_attribute_sum_per_user('purchase', 'purchase_value')

    # Analysis with gboost CUPED
    analyzer_gboost_cuped = ABTestAnalyzer(event_data, user_allocations, "a1", user_properties, mode="gboost_cuped")
    results_gboost_cuped = analyzer_gboost_cuped.calculate_event_attribute_sum_per_user('purchase', 'purchase_value')

    return {
        "no_enhancement": results_no_enhancement,
        "cuped": results_cuped,
        "gboost_cuped": results_gboost_cuped
    }

if __name__ == "__main__":
    # Set up logging
    logger.setLevel(logging.DEBUG)

    # Generate synthetic data
    generated_data = generate_synthetic_data(
        num_users=100000,
        countries=['US', 'UK', 'DE', 'FR', 'CA', 'AU', 'JP', 'IN'],
        platforms=['iOS', 'Android', 'Web', 'Desktop'],
        user_segments=['Segment_1', 'Segment_2', 'Segment_3', 'Segment_4'],
        ab_groups=['a1', 'a2', 'b'],
        noise_level=1.0,
        base_increase_percentage=0.1,
    )

    # Save the data to a CSV file
    file_name = '0_generated_data.csv'
    full_path = os.path.join(folder_name, file_name)
    generated_data.to_csv(full_path, index=False)
    logger.info(f"Data saved to {full_path}")

    # Run analysis on the generated data
    results = run_analysis(generated_data)
    logger.info("Analysis results (no enhancement): %s", results["no_enhancement"])
    logger.info("Analysis results (cuped): %s", results["cuped"])
    logger.info("Analysis results (gboost_cuped): %s", results["gboost_cuped"])
