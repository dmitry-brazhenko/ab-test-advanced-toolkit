import math
from typing import List

from ab_test_advanced_toolkit.metrics import MetricType, Metric


def describe_metric(metric: Metric) -> str:
    metric_type = metric.metrictype
    params = metric.metricparams

    if metric_type == MetricType.EVENT_COUNT_PER_USER:
        return f"Count of '{params.event_name}' events per user."

    elif metric_type == MetricType.EVENT_ATTRIBUTE_SUM_PER_USER:
        return f"Sum of '{params.attribute_name}' for '{params.event_name}' events per user."

    elif metric_type == MetricType.CONVERSION_RATE:
        return f"Conversion rate to '{params.event_name}' event per user."

    else:
        return "Unknown metric type."


def format_metrics_to_html(metrics: List[Metric], control_group: str, test_groups: List[str]):
    # Helper function to determine the color based on p-value and difference
    def get_color(pval, diff):
        if pval > 0.05:
            return 'grey'
        if 0.01 < pval <= 0.05:
            return 'lightgreen' if diff > 0 else 'lightcoral'
        return 'green' if diff > 0 else 'red'

    # Start of the HTML string with enhanced styles for centering and aesthetics
    html_str = '''
    <style>
        body {display: flex; justify-content: center; margin: 0; height: 100vh; align-items: center;}
        table {border-collapse: collapse; width: 80%; margin: auto;}
        td, th {text-align: center; padding: 8px; border: 1px solid #ddd;}
        td, th {min-width: 100px; max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
        table {table-layout: fixed; width: auto;}
        th {background-color: #f2f2f2;}
        .grey {background-color: grey;}
        .lightgreen {background-color: lightgreen;}
        .lightcoral {background-color: lightcoral;}
        .green {background-color: green;}
        .red {background-color: red;}
        tr:hover {background-color: #f5f5f5;}
    </style>
    <table>
        <tr><th>Metric</th><th>''' + control_group + '''</th>'''

    for test_group in test_groups:
        html_str += '<th>' + test_group + '</th>'
    html_str += '</tr>'

    for metric in metrics:
        metric_name = describe_metric(metric)
        html_str += f'<tr><td>{metric_name}</td>'
        control_value = metric.result.data[control_group]
        html_str += f'<td>{control_value:.2f}</td>'

        for test_group in test_groups:
            test_value = metric.result.data[test_group]
            pval = metric.result.stat_significance[test_group]
            diff = ((test_value - control_value) / control_value) * 100 if control_value != 0 else float('inf')
            diff_rounded = int(round(diff, 0)) if control_value != 0 else 0
            color = get_color(pval, diff)
            title_text = f'P-value: {pval:.4f}'
            html_str += f'<td class="{color}" title="{title_text}">{test_value:.2f}<br/>({diff_rounded}%)</td>'

        html_str += '</tr>'

    html_str += '</table>'
    return html_str
