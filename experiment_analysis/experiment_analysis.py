import numpy as np
from tqdm import tqdm
from data_generation.data_generator import generate_synthetic_data, run_analysis
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import hashlib

from itertools import product

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_seed(params, i):
    # Convert parameters to a sorted string and hash it
    param_str = str(sorted(params.items()))
    param_hash = int(hashlib.md5(param_str.encode()).hexdigest(), 16) % (10 ** 8)  # Limit the size of the hash
    return param_hash * i  # Combine the hash with the iteration number

def display_results(results):
    # Display the results
    for metric_type, sizes in results.items():
        logger.info(f"{metric_type}:")
        for size, median_value in sizes.items():
            logger.info(f"  {size}: Median {median_value}")


# Analyzing p_values vs num_users with fixed other parameters one by one for all combinations

def analyze_feature(values_ranges, fixed_params, feature, num_iterations=50):
    feature_range = values_ranges[feature]
    results = {
        'no_enhancement': [],
        'cuped': [],
        'gboost_cuped': []
    }

    for value in tqdm(feature_range):
        no_enhancement_values = []
        cuped_values = []
        gboost_cuped_values = []

        for i in range(num_iterations):
            params = fixed_params.copy()
            params[feature] = value
            # Generate seed using the hash of parameters and iteration number
            seed = get_seed(params, i)
            generated_data = generate_synthetic_data(**params, seed=seed)
            analysis_results = run_analysis(generated_data)

            no_enhancement_values.append(analysis_results['no_enhancement'].result.stat_significance['b'])
            cuped_values.append(analysis_results['cuped'].result.stat_significance['b'])
            gboost_cuped_values.append(analysis_results['gboost_cuped'].result.stat_significance['b'])

        results['no_enhancement'].append((value, np.median(no_enhancement_values)))
        results['cuped'].append((value, np.median(cuped_values)))
        results['gboost_cuped'].append((value, np.median(gboost_cuped_values)))

    return results

def plot_feature_results(results, feature, fixed_params, show_histogram=False):
    sns.set(style="whitegrid")
    feature_range = [val[0] for val in results['no_enhancement']]

    plt.figure(figsize=(12, 8))
    sns.lineplot(x=feature_range, y=[val[1] for val in results['no_enhancement']], marker='o', label='No Enhancement')
    sns.lineplot(x=feature_range, y=[val[1] for val in results['cuped']], marker='s', label='CUPED')
    sns.lineplot(x=feature_range, y=[val[1] for val in results['gboost_cuped']], marker='^', label='GBoost CUPED')

    plt.xlabel(feature)
    plt.ylabel('Median P-value')
    plt.title(f'Median P-values vs {feature}')
    plt.legend()
    plt.grid(True)
    
    # Format fixed parameters
    fixed_params_str = "\n".join([f"{k}: {v}" for k, v in fixed_params.items()])
    
    # Adding the notes below the plot with left alignment
    plt.figtext(0.1, -0.15, f"Fixed parameters:\n{fixed_params_str}", wrap=True, horizontalalignment='left', fontsize=10)
    plt.subplots_adjust(bottom=0.1)  # Adjust this value to reduce the space
    plt.show()

    # Plotting the histogram of p-values
    if show_histogram:
        plt.figure(figsize=(12, 8))
        sns.histplot([val[1] for val in results['no_enhancement']], bins=50, kde=True, label='P-values', color='blue')
        plt.xlabel('P-value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of P-values for {feature}')
        plt.legend()
        plt.grid(True)
    
    plt.show()


def analyze_and_plot_features(fixed_params, varying_params, x_params, num_iterations=50):
    """
    Analyzes and plots features with given ranges, varying the values of specified features while keeping others fixed.
    
    Args:
        fixed_params (dict): Parameters that remain constant.
            Example:
            {
                'countries': ['US', 'UK', 'DE', 'FR'],
                'platforms': ['iOS', 'Android'],
                'user_segments': ['Segment_1', 'Segment_2'],
                'ab_groups': ['a1', 'a2'],
                'noise_level': 0.5,
                'correlation_level': 0.5
            }
        varying_params (dict): Parameters with ranges used for generating combinations.
            Example:
            {
                'base_increase_percentage': np.arange(0.05, 0.20, 0.10)
            }
        x_params (dict): Parameters with ranges used as X-axis on charts.
            Example:
            {
                'num_users': range(100, 5000, 1000),
                'correlation_level': np.arange(0.0, 1, 0.5)
            }
        num_iterations (int): Number of iterations for analysis.
    """
    
    # Check for errors
    # Ensure no variable is in both fixed_params and varying_params
    for feature in set(fixed_params.keys()).intersection(varying_params.keys()):
        raise ValueError(f"Feature '{feature}' cannot be in both fixed_params and varying_params.")

    # Ensure x_params is not empty
    if not x_params:
        raise ValueError("The x_params dictionary must contain at least one feature.")

    # Ensure at least one feature in fixed_params or varying_params
    if not fixed_params and not varying_params:
        raise ValueError("There must be at least one feature in either fixed_params or varying_params.")

    # Ensure each feature in x_params has at least one value
    for feature, values in x_params.items():
        if len(values) == 0:
            raise ValueError(f"The feature '{feature}' in x_params must have at least one value.")
    
    # Main logic for x_params
    for x_feature, x_values in x_params.items():
        # Create combined combinations of varying_params, excluding the current x_feature
        combined_params = {k: v for k, v in varying_params.items() if k != x_feature}
        all_combinations = list(product(*[[(k, v) for v in values] for k, values in combined_params.items()]))
        
        for combination in all_combinations:
            params = fixed_params.copy()
            for k, v in combination:
                params[k] = v

            # Exclude x_feature from fixed_params
            if x_feature in params:
                del params[x_feature]

            # Ensure the correct value for the x_feature
            logger.info(f"Analyzing feature '{x_feature}' with fixed params: {params}")
            temp_values_ranges = {x_feature: x_values}
            results = analyze_feature(temp_values_ranges, params, x_feature, num_iterations)
            plot_feature_results(results, x_feature, params)
