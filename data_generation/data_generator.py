import hashlib
import os
import datetime
import pandas as pd
import numpy as np
from ab_test_advanced_toolkit.analyzer import ABTestAnalyzer
from typing import Tuple, Any
import random
import logging

logger = logging.getLogger(__name__)

# Create folder 'data/{now_timestamp_utc}' if it does not exist
folder_name = os.path.join(
    'data', 
    datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
)
os.makedirs(folder_name, exist_ok=True)

def describe_dataset(df: pd.DataFrame) -> None:
    logger.debug("First 5 rows of the dataset:")
    logger.debug(df.head())
    
    logger.debug("\nStatistical summary of the dataset:")
    logger.debug(df.describe())
    
    logger.debug("\nCorrelation between 'value' and 'pre_test_value':")
    logger.debug(df[['value', 'pre_test_value']].corr())
    
    logger.debug("\nAverage 'value' by A/B groups:")
    ab_means = df.groupby('abgroup')['value'].mean()
    logger.debug(ab_means)


def calculate_hash_1(val: Any) -> float:
    hash_value = int(hashlib.md5(f"{val}".encode()).hexdigest(), 16)
    residual =  (hash_value % 25) / 25
    return residual

def calculate_hash_2(val: Any) -> float:
    hash_value = int(hashlib.sha256(f"{val}".encode()).hexdigest(), 16)
    residual = (hash_value % 25) / 25
    return residual

def generate_pre_test_value(age, engagement_score, country, platform, user_segment, noise_level):
    # Create a hash of the country and platform
    value = 1 + \
            calculate_hash_1(age) + \
            calculate_hash_1(engagement_score) + \
            calculate_hash_1(country) + \
            calculate_hash_1(platform) + \
            calculate_hash_1(user_segment)
    # Add deterministic noise
    # value += np.random.normal(0, noise_level)
    return value



def generate_intermediate_in_test_value(age, engagement_score, country, platform, user_segment, noise_level):
    # Create a hash of the country and platform
    value = 1 + \
            calculate_hash_2(age) + \
            calculate_hash_2(engagement_score) + \
            calculate_hash_2(country) + \
            calculate_hash_2(platform) + \
            calculate_hash_2(user_segment)
    # Add deterministic noise
    # value += np.random.normal(0, noise_level)
    return value



def generate_synthetic_data(
    num_users: int = 1000,
    alpha: float = 0.5,
    countries: list[str] = ['US', 'UK', 'DE', 'FR', 'CA', 'AU', 'JP', 'IN'],
    platforms: list[str] = ['iOS', 'Android', 'Web', 'Desktop'],
    user_segments: list[str] = ['Segment_1', 'Segment_2', 'Segment_3', 'Segment_4'],
    ab_groups: list[str] = ['a1', 'a2', 'b'],
    noise_level: float = 1.0,
    base_increase_percentage: float = 0.2,
    seed: int = 40
) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)
    
    data: dict[str, list[Any]] = {
        'userid': [],
        'country': [],
        'platform': [],
        'user_segment': [],
        'abgroup': [],
        'age': [],
        'engagement_score': [],
        'pre_test_value': [],
        'value': []
    }
    
    for i in range(num_users):
        user_id = i + 1
        country = random.choice(countries)
        platform = random.choice(platforms)
        user_segment = random.choice(user_segments)
        ab_group = random.choice(ab_groups)
        age = np.random.randint(18, 65)
        engagement_score = np.random.randint(1, 11)
        
        pre_test_value = generate_pre_test_value(age, engagement_score, country, platform, user_segment, noise_level)
        intermediate_value = generate_intermediate_in_test_value(age, engagement_score, country, platform, user_segment, noise_level)
        
        in_test_value_alpha = alpha * pre_test_value + (1 - alpha) * intermediate_value
        in_test_value_increased = in_test_value_alpha
        if ab_group.startswith('b'):
            in_test_value_increased = in_test_value_increased * (1 + base_increase_percentage)

        in_test_value_increased = in_test_value_increased + np.random.normal(0, noise_level)
        data['userid'].append(user_id)
        data['country'].append(country)
        data['platform'].append(platform)
        data['user_segment'].append(user_segment)
        data['abgroup'].append(ab_group)
        data['age'].append(age)
        data['engagement_score'].append(engagement_score)
        data['pre_test_value'].append(pre_test_value)
        data['value'].append(in_test_value_increased)
    
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
        "age": df["age"],
        "engagement_score": df["engagement_score"],
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
