import numpy as np
import pandas as pd


def generate_user_properties():
    np.random.seed(42)  # For reproducible results

    user_ids = np.arange(1, 101)
    ages = np.random.randint(18, 71, size=100)
    genders = np.random.choice(['Male', 'Female'], size=100, p=[0.5, 0.5])
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan', 'Brazil', 'India', 'China']
    country_distribution = np.random.choice(countries, size=100)
    device_types = np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=100, p=[0.5, 0.3, 0.2])
    membership_status = np.random.choice(['Free', 'Premium'], size=100, p=[0.7, 0.3])
    user_properties = pd.DataFrame({
        'userid': user_ids,
        'age': ages,
        'gender': genders,
        'country': country_distribution,
        'device_type': device_types,
        'membership_status': membership_status
    })
    return user_properties


def generate_event_data():
    np.random.seed(42)  # For reproducible results

    # Expanding the event_list DataFrame
    user_ids = np.random.randint(1, 9, size=100)
    event_names = np.random.choice(['login', 'purchase'], size=100, p=[0.7, 0.3])
    purchase_values = np.where(event_names == 'purchase', np.random.randint(50, 350, size=100), 0)
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq='H')

    event_list = pd.DataFrame({
        'timestamp': timestamps,
        'userid': user_ids,
        'event_name': event_names,
        'purchase_value': purchase_values,
    })
    event_list = event_list[["timestamp", "userid", "event_name", "purchase_value"]]
    return event_list


def generate_user_allocations():
    np.random.seed(42)  # For reproducible results
    user_ids = np.arange(1, 101)
    abgroups = np.random.choice(['A', 'B'], size=100, p=[0.5, 0.5])
    ab_timestamps = pd.date_range(start="2022-12-15", periods=100, freq='H')

    ab_test_allocations = pd.DataFrame({
        'timestamp': ab_timestamps,
        'userid': user_ids,
        'abgroup': abgroups,
    })
    ab_test_allocations = ab_test_allocations[["timestamp", "userid", "abgroup"]]
    return ab_test_allocations
