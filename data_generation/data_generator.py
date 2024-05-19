import os
import datetime

import pandas as pd
import numpy as np


# Create folder 'data/{now_timestamp_utc}' if it does not exist
folder_name = os.path.join(
    'data', 
    datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d_%H-%M-%S')
)
os.makedirs(folder_name, exist_ok=True)


def describe_dataset(df):
    print("First 5 rows of the dataset:")
    print(df.head())
    
    print("\nStatistical summary of the dataset:")
    print(df.describe())
    
    print("\nCorrelation between 'value' and 'pre_test_value':")
    print(df[['value', 'pre_test_value']].corr())
    
    print("\nAverage 'value' by A/B groups:")
    ab_means = df.groupby('abgroup')['value'].mean()
    print(ab_means)

def generate_synthetic_data(num_users, countries, platforms, user_segments, ab_groups, base_increase_percentage, noise_level=1.0, seed=40):
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Generate synthetic data
    data = {
        'userid': range(1, num_users + 1),
        'country': np.random.choice(countries, num_users),
        'platform': np.random.choice(platforms, num_users),
        'user_segment': np.random.choice(user_segments, num_users),
        'abgroup': np.random.choice(ab_groups, num_users, p=[1/3, 1/3, 1/3]),
        'age': np.random.randint(18, 65, num_users),  # Age between 18 and 65
        'engagement_score': np.random.rand(num_users) * 10,  # Random score between 0 and 10
    }
    df = pd.DataFrame(data)

    # Calculate the base effect from features
    base_value = 10 + df['age'] / 10 + df['engagement_score']

    # Pre-compute indexes for countries, platforms, and user_segments to avoid using apply
    df['country_idx'] = df['country'].apply(lambda x: countries.index(x) % 3)
    df['platform_idx'] = df['platform'].apply(lambda x: platforms.index(x) % 2)
    df['segment_idx'] = df['user_segment'].apply(lambda x: user_segments.index(x) % 4)

    # Vectorized operations for country, platform, and segment effects
    country_effect = np.random.uniform(-1, 5, num_users) * df['country_idx']
    platform_effect = np.random.uniform(-1, 5, num_users) * df['platform_idx']
    segment_effect = np.random.uniform(-1, 5, num_users) * df['segment_idx']

    category_effect = country_effect + platform_effect + segment_effect

    # Group effect with added randomness
    group_effect = np.where(
        df['abgroup'] == 'b', 
        np.random.uniform(-1 * base_increase_percentage, base_increase_percentage * 3, num_users), 
        np.random.uniform(-2 * base_increase_percentage, base_increase_percentage * 2, num_users),
    )

    # Calculate the final value with added nonlinear category effect and noise
    df['value'] = base_value * (1 + category_effect) * (1 + group_effect) + np.random.normal(0, noise_level, num_users)

    # Generating pre-test value with a slightly different formula to introduce non-linearity and noise
    df['pre_test_value'] = base_value * (1 + np.random.normal(1, 0.5, num_users) * category_effect) * (1 + np.random.normal(0, 0.05, num_users)) + \
                        0.5 * np.sin(df['engagement_score'] / 2) * np.random.uniform(-1, 1, num_users) + \
                        np.random.normal(-1, 1, num_users)  # Adding more noise for pre-test value


    describe_dataset(df)

    return df

generated_data = generate_synthetic_data(
    num_users=100000,
    countries=['US', 'UK', 'DE', 'FR', 'CA', 'AU', 'JP', 'IN'],
    platforms=['iOS', 'Android', 'Web', 'Desktop'],
    user_segments=['Segment_1', 'Segment_2', 'Segment_3', 'Segment_4'],
    ab_groups=['a1', 'a2', 'b'],
    noise_level=1.0,
    base_increase_percentage=0.05
)

# Save the data to a CSV file
file_name = '0_generated_data.csv'
full_path = os.path.join(folder_name, file_name)
generated_data.to_csv(full_path, index=False)
print(f"Data saved to {full_path}")
