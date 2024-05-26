import numpy as np
from tqdm import tqdm
from data_generation.data_generator import generate_synthetic_data, run_analysis
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from itertools import product

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def display_results(results):
    # Display the results
    for metric_type, sizes in results.items():
        logger.info(f"{metric_type}:")
        for size, median_value in sizes.items():
            logger.info(f"  {size}: Median {median_value}")


# Analyzing p_values vs num_users with fixed other parameters one by one for all combinations

def analyze_num_users(values_ranges, fixed_params, num_iterations=50):
    num_users_range = values_ranges['num_users']
    results = {
        'no_enhancement': [],
        'cuped': [],
        'gboost_cuped': []
    }
    all_p_values = []

    for num_users in tqdm(num_users_range):
        no_enhancement_values = []
        cuped_values = []
        gboost_cuped_values = []

        for i in range(num_iterations):
            params = fixed_params.copy()
            params['num_users'] = num_users
            generated_data = generate_synthetic_data(**params, seed=i)
            analysis_results = run_analysis(generated_data)

            no_enhancement_values.append(analysis_results['no_enhancement'].result.stat_significance['b'])
            cuped_values.append(analysis_results['cuped'].result.stat_significance['b'])
            gboost_cuped_values.append(analysis_results['gboost_cuped'].result.stat_significance['b'])
            all_p_values.append(analysis_results['no_enhancement'].result.stat_significance['b'])

        results['no_enhancement'].append((num_users, np.median(no_enhancement_values)))
        results['cuped'].append((num_users, np.median(cuped_values)))
        results['gboost_cuped'].append((num_users, np.median(gboost_cuped_values)))

    return results, all_p_values

def plot_num_users_results(results, fixed_params):
    sns.set(style="whitegrid")
    num_users_range = [val[0] for val in results['no_enhancement']]
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=num_users_range, y=[val[1] for val in results['no_enhancement']], marker='o', label='No Enhancement')
    sns.lineplot(x=num_users_range, y=[val[1] for val in results['cuped']], marker='s', label='CUPED')
    sns.lineplot(x=num_users_range, y=[val[1] for val in results['gboost_cuped']], marker='^', label='GBoost CUPED')

    plt.xlabel('num_users')
    plt.ylabel('Median P-value')
    plt.title('Median P-values vs num_users')
    plt.legend()
    plt.grid(True)
    
    # Adding the notes below the plot
    plt.figtext(0.5, -0.1, f"Fixed parameters: {fixed_params}", wrap=True, horizontalalignment='center', fontsize=10)
    plt.show()

    # Plotting the histogram of p-values
    plt.figure(figsize=(12, 8))
    sns.histplot([val[1] for val in results['no_enhancement']], bins=50, kde=True, label='P-values', color='blue')
    plt.xlabel('P-value')
    plt.ylabel('Frequency')
    plt.title('Histogram of P-values')
    plt.legend()
    plt.grid(True)
    
    plt.show()

def analyze_and_plot_num_users(values_ranges, fixed_params, num_iterations=50):
    variable = 'num_users'
    values_range = values_ranges[variable]

    # Ensure only num_users is varied; other parameters are fixed
    fixed_params_combinations = list(product(*[[(k, v) for v in values] for k, values in values_ranges.items() if k != variable]))

    for combination in fixed_params_combinations:
        params = fixed_params.copy()
        for k, v in combination:
            params[k] = v

        logger.info(f"Analyzing with fixed params: {params}")
        results, all_p_values = analyze_num_users({'num_users': values_range}, params, num_iterations)
        plot_num_users_results(results, params)
