from typing import Tuple, List, Optional

import pandas as pd

from ab_test_advanced_toolkit.data_validation import validate_data
from ab_test_advanced_toolkit.metrics import Metric, MetricType, MetricParams, MetricResult, AggregationOperation
from ab_test_advanced_toolkit.stat_significance import StatTests
from ab_test_advanced_toolkit.vizualizer import format_metrics_to_html


import logging

# Function to set up logging configuration
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

class ABTestAnalyzer:
    def __init__(self, event_data: pd.DataFrame, ab_test_allocations: pd.DataFrame, control_group_name: str,
                 user_properties: Optional[pd.DataFrame] = None, mode="gboost_cuped", logging_level=logging.INFO):
        """
        Initializes the ABTestAnalyzer with event data, AB test allocations, control group name, user properties, and mode.
        :param event_data: DataFrame containing event data. Expected pandas format: |timestamp|userid|event_name|attribute_1|attribute_2|...
        :param ab_test_allocations: DataFrame containing AB test allocations. Expected pandas format: |timestamp|userid|abgroup|
        :param control_group_name: The name of the control group.
        :param user_properties: DataFrame containing user properties. Expected pandas format: |userid|property_1|property_2|...
        :param mode: Mode of enhancement ("no_enhancement", "cuped", "gboost_cuped").
        :param logging_level: The logging level to be used (e.g., logging.INFO, logging.DEBUG).
        """

        self.logger = setup_logging(logging_level)

        assert mode in ["no_enhancement", "cuped", "gboost_cuped"], "Invalid mode"

        self.control_group_name = control_group_name
        self.test_group_names = [group for group in ab_test_allocations['abgroup'].unique() if
                                 group != control_group_name]

        validate_data(event_data, ab_test_allocations, control_group_name)

        self.ab_test_allocations = ab_test_allocations.set_index("userid")
        self.event_data = event_data
        self.user_properties = user_properties
        self.calculated_metrics: List[Metric] = []
        self.mode = mode

    @staticmethod
    def _merge_and_aggregate(event_data: pd.DataFrame, ab_test_allocations: pd.DataFrame, event_name: str,
                             operation: AggregationOperation,
                             attribute_name=None,
                             pretest=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merges event data with AB test allocations and aggregates data based on specified attributes and operation.
        Adds support for conversion operation.

        :param event_data: DataFrame containing event data. Expected pandas format: |timestamp|userid|event_name|attribute_1|attribute_2|...
        :param ab_test_allocations: DataFrame containing AB test allocations. Expected pandas format: |timestamp|userid|abgroup|
        :param event_name: The name of the event to filter on.
        :param attribute_name: The name of the attribute to aggregate sum
        :param operation: The aggregation operation ('sum', 'count', or 'conversion').
        :param pretest: Boolean indicating whether to process pretest (True) or intest (False) data.
        :return: raw merged data, metrics data
        """
        filtered_event_data = event_data[event_data["event_name"] == event_name]

        event_data_with_alloc = pd.merge(filtered_event_data, ab_test_allocations[['timestamp']], left_on='userid',
                                         right_index=True, how='left', suffixes=('_event', '_alloc'))

        # Filter for pretest or intest events
        if pretest:
            filtered_events = event_data_with_alloc[
                event_data_with_alloc['timestamp_event'] < event_data_with_alloc['timestamp_alloc']]
        else:
            filtered_events = event_data_with_alloc[
                event_data_with_alloc['timestamp_event'] >= event_data_with_alloc['timestamp_alloc']]

        # Apply aggregation operation
        if operation == AggregationOperation.CONVERSION:
            # Mark each user as 1 (converted) if the event occurred, else 0
            aggregated_data = filtered_events.groupby("userid").size().gt(0).astype(int).to_frame(
                name='conversion_status')
        elif operation == AggregationOperation.COUNT:
            aggregated_data = filtered_events.groupby("userid").agg({"event_name": "count"})
        elif operation == AggregationOperation.SUM:
            aggregated_data = filtered_events.groupby("userid").agg({attribute_name: "sum"})
        else:
            raise ValueError(f"Unsupported aggregation operation: {operation}")

        # Merge aggregated data with AB test allocations and fill missing values with 0
        merged_data = pd.merge(ab_test_allocations[['abgroup']], aggregated_data, left_index=True, right_index=True,
                               how="left").fillna(0)
        result = merged_data.groupby("abgroup").mean()

        return merged_data, result

    def calculate_event_count_per_user(self, event_name: str) -> pd.DataFrame:
        """
        Calculates the count of events per user for the specified event name.
        :param event_name: The name of the event to calculate the count for.
        :return: DataFrame with the calculated event count per user per group
        """
        merged_pretest, _ = self._merge_and_aggregate(self.event_data, self.ab_test_allocations, event_name,
                                                      AggregationOperation.COUNT, pretest=True)

        merged_intest, result_intest = self._merge_and_aggregate(self.event_data, self.ab_test_allocations, event_name,
                                                                 AggregationOperation.COUNT)

        if self.mode == "gboost_cuped":
            stat_test = StatTests.calculate_gboost_cuped_and_compare(merged_pretest, merged_intest, self.user_properties,
                                                                self.control_group_name, self.test_group_names, True)
        elif self.mode == "cuped":
            stat_test = StatTests.calculate_cuped_and_compare(merged_pretest, merged_intest,
                                                                self.control_group_name, self.test_group_names)
        else:
            stat_test = StatTests.calculate_gboost_cuped_and_compare(merged_pretest, merged_intest,
                                                                    self.user_properties,
                                                                    self.control_group_name, self.test_group_names,
                                                                    False)

        metric_output = Metric(MetricType.EVENT_COUNT_PER_USER, MetricParams(event_name),
                                              MetricResult(result_intest, self.control_group_name,
                                                           self.test_group_names, stat_test))
        self.calculated_metrics.append(metric_output)
        return metric_output

    def calculate_event_attribute_sum_per_user(self, event_name: str, attribute_name: str) -> pd.DataFrame:
        """
        Calculates the sum of a specified attribute per user for the specified event name.
        :param event_name: The event name to calculate the attribute sum for.
        :param attribute_name:  The name of the attribute to calculate the sum for.
        :return: DataFrame with the calculated attribute sum per user per group
        """

        if attribute_name not in self.event_data.columns:
            raise ValueError(f"Attribute {attribute_name} not found in event data.")
        if self.event_data[attribute_name].dtype not in [int, float]:
            raise ValueError(f"Attribute {attribute_name} is not numeric.")

        merged_pretest, _ = self._merge_and_aggregate(self.event_data, self.ab_test_allocations, event_name,
                                                    AggregationOperation.SUM, attribute_name, pretest=True)

        merged_intest, result_intest = self._merge_and_aggregate(self.event_data, self.ab_test_allocations, event_name,
                                                                AggregationOperation.SUM, attribute_name)

        if self.mode == "gboost_cuped":
            stat_test = StatTests.calculate_gboost_cuped_and_compare(merged_pretest, merged_intest, self.user_properties,
                                                                self.control_group_name, self.test_group_names, True)
        elif self.mode == "cuped":
            stat_test = StatTests.calculate_cuped_and_compare(merged_pretest, merged_intest,
                                                                self.control_group_name, self.test_group_names)
        else:
            stat_test = StatTests.calculate_gboost_cuped_and_compare(merged_pretest, merged_intest,
                                                                    self.user_properties,
                                                                    self.control_group_name, self.test_group_names,
                                                                    False)

        metric_output = Metric(MetricType.EVENT_ATTRIBUTE_SUM_PER_USER, MetricParams(event_name, attribute_name),
                MetricResult(result_intest, self.control_group_name, self.test_group_names, stat_test))
        self.calculated_metrics.append(metric_output)
        return metric_output

    def calculate_conversion(self, target_event: str) -> pd.DataFrame:
        """
        Calculates the conversion rate for the specified target event.
        :param target_event: The name of the event to calculate the conversion rate for.
        :return: DataFrame with the calculated conversion rate per user per group
        """
        merged_intest, result_intest = self._merge_and_aggregate(self.event_data, self.ab_test_allocations,
                                                                 target_event,
                                                                 AggregationOperation.CONVERSION)
        stat_test = StatTests.calculate_t_test_for_dataset(merged_intest, self.control_group_name,
                                                           self.test_group_names)

        metric_output = Metric(
            MetricType.CONVERSION_RATE,
            MetricParams(target_event),
            MetricResult(result_intest, self.control_group_name, self.test_group_names, stat_test))
        self.calculated_metrics.append(metric_output)
        return metric_output

    def save_report(self, filename: str):
        res = format_metrics_to_html(self.calculated_metrics, self.control_group_name, self.test_group_names)
        with open(filename, 'w') as f:
            f.write(res)
