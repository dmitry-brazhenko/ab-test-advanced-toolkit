import numpy as np
from tqdm import tqdm
from data_generation.data_generator import generate_synthetic_data, run_analysis
import matplotlib.pyplot as plt
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_variable(variable, values_range, fixed_params, num_iterations=50):
    results = {
        'no_enhancement': {},
        'cuped': {},
        'gboost_cuped': {}
    }
    all_p_values = []

    for value in tqdm(values_range):
        no_enhancement_values = []
        cuped_values = []
        gboost_cuped_values = []

        for i in range(num_iterations):
            params = fixed_params.copy()
            params[variable] = value
            generated_data = generate_synthetic_data(**params, seed=i)
            analysis_results = run_analysis(generated_data)

            no_enhancement_values.append(analysis_results['no_enhancement'].result.stat_significance['b'])
            cuped_values.append(analysis_results['cuped'].result.stat_significance['b'])
            gboost_cuped_values.append(analysis_results['gboost_cuped'].result.stat_significance['b'])
            all_p_values.append(analysis_results['no_enhancement'].result.stat_significance['b'])

        results['no_enhancement'][value] = np.median(no_enhancement_values)
        results['cuped'][value] = np.median(cuped_values)
        results['gboost_cuped'][value] = np.median(gboost_cuped_values)

    return results, all_p_values

def display_results(results):
    # Display the results
    for metric_type, sizes in results.items():
        logger.info(f"{metric_type}:")
        for size, median_value in sizes.items():
            logger.info(f"  {size}: Median {median_value}")

def plot_results(variable, values_range, results, all_p_values):
    # Plotting the results
    plt.figure(figsize=(12, 8))
    plt.plot(values_range, list(results['no_enhancement'].values()), marker='o', label='No Enhancement')
    plt.plot(values_range, list(results['cuped'].values()), marker='s', label='CUPED')
    plt.plot(values_range, list(results['gboost_cuped'].values()), marker='^', label='GBoost CUPED')

    plt.xlabel(variable)
    plt.ylabel('Median P-value')
    plt.title(f'Median P-values vs {variable}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the histogram of p-values
    plt.figure(figsize=(12, 8))
    plt.hist(all_p_values, bins=50, alpha=0.75, label='P-values')
    plt.xlabel('P-value')
    plt.ylabel('Frequency')
    plt.title('Histogram of P-values')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_and_plot(variable, values_range, fixed_params, num_iterations=50):
    results, all_p_values = analyze_variable(variable, values_range, fixed_params, num_iterations)
    display_results(results)
    plot_results(variable, values_range, results, all_p_values)

def analyze_multiple_metrics(values_ranges, fixed_params, num_iterations=50):
    """
    Analyzes and plots graphs for all specified variables.

    :param values_ranges: Dictionary where keys are variables and values are ranges of values for those variables.
    :param fixed_params: Dictionary of fixed parameters for data generation.
    :param num_iterations: Number of iterations for data generation and analysis.
    """
    for variable, values_range in values_ranges.items():
        logger.info(f"Analyzing {variable}...")
        analyze_and_plot(variable, values_range, fixed_params, num_iterations)

def plot_3d_results(var1, var2, results):
    # Plotting the results
    fig = plt.figure(figsize=(14, 10))

    for metric_type, data in results.items():
        ax = fig.add_subplot(111, projection='3d')
        x_vals = [item[0] for item in data]
        y_vals = [item[1] for item in data]
        z_vals = [item[2] for item in data]
        ax.scatter(x_vals, y_vals, z_vals, label=metric_type)

        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_zlabel('Median P-value')
        ax.set_title(f'3D plot of {var1} vs {var2} vs Median P-value')
        ax.legend()
        plt.show()

def analyze_pairwise_metrics(variable_pairs, values_ranges, fixed_params, num_iterations=50):
    """
    Analyzes and plots 3D graphs for all pairs of variables.

    :param variable_pairs: List of tuples of pairs of variables for analysis.
    :param values_ranges: Dictionary where keys are variables and values are ranges of values for those variables.
    :param fixed_params: Dictionary of fixed parameters for data generation.
    :param num_iterations: Number of iterations for data generation and analysis.
    """
    for (var1, var2) in variable_pairs:
        logger.info(f"Analyzing pair ({var1}, {var2})...")
        
        results = {
            'no_enhancement': [],
            'cuped': [],
            'gboost_cuped': []
        }
        
        all_p_values = []
        
        for val1 in tqdm(values_ranges[var1]):
            for val2 in values_ranges[var2]:
                no_enhancement_values = []
                cuped_values = []
                gboost_cuped_values = []

                for i in range(num_iterations):
                    params = fixed_params.copy()
                    params[var1] = val1
                    params[var2] = val2
                    generated_data = generate_synthetic_data(**params, seed=i)
                    analysis_results = run_analysis(generated_data)

                    no_enhancement_values.append(analysis_results['no_enhancement'].result.stat_significance['b'])
                    cuped_values.append(analysis_results['cuped'].result.stat_significance['b'])
                    gboost_cuped_values.append(analysis_results['gboost_cuped'].result.stat_significance['b'])
                    all_p_values.append(analysis_results['no_enhancement'].result.stat_significance['b'])

                results['no_enhancement'].append((val1, val2, np.median(no_enhancement_values)))
                results['cuped'].append((val1, val2, np.median(cuped_values)))
                results['gboost_cuped'].append((val1, val2, np.median(gboost_cuped_values)))

        plot_3d_results(var1, var2, results)
