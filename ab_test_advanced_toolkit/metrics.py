from typing import List

import pandas as pd

from enum import Enum, auto

from ab_test_advanced_toolkit.stat_significance import StatSignificanceResult


class MetricType(Enum):
    EVENT_COUNT_PER_USER = auto()
    EVENT_ATTRIBUTE_SUM_PER_USER = auto()
    CONVERSION_RATE = auto()


class AggregationOperation(Enum):
    SUM = auto()
    MEAN = auto()
    COUNT = auto()
    CONVERSION = auto()


class MetricParams:
    def __init__(self, event_name, attribute_name=None):
        """
        Initialize with the event name and an optional attribute name.
        :param event_name: event name that is used to calculate the metric. For example, "purchase"
        :param attribute_name: attribute that is used for the metric calculation (optional).
                For example, revenuee sum for event "purchase"
        """
        self.event_name = event_name
        self.attribute_name = attribute_name  # Optional, not all metrics may need this

    def __repr__(self):
        # Provides a readable representation, showing only non-None attributes
        params = {k: v for k, v in vars(self).items() if v is not None}
        return f"<MetricParams({params})>"


class MetricResult:
    def __init__(self, df: pd.DataFrame, control_group: str, test_groups: List[str],
                 stat_significance: StatSignificanceResult):
        """
        Initialize with a pandas DataFrame and specify the control and test groups.
        The DataFrame is expected to have an index named 'abgroup' and a single column with metric values.

        :param df: pandas DataFrame with metrics data.
        :param control_group: The identifier for the control group (e.g., 'A').
        :param test_groups: List of identifiers for the test groups (e.g., ['B', 'C']).
        :param stat_significance: StatSignificanceResult instance with the method used and p-values.
        """
        self.control_group = control_group
        self.test_groups = test_groups
        self.data = self._process_dataframe(df)
        self.stat_significance = self._associate_pvals_with_groups(stat_significance)
        self.stat_significance_method = stat_significance.method_used

    def _process_dataframe(self, df: pd.DataFrame) -> dict:
        """
        Processes the DataFrame to extract metric values for the control and test groups,
        storing the results in a structured dictionary.
        :param df: pandas DataFrame with metrics data.
        """
        data = {self.control_group: df.loc[self.control_group].item() if self.control_group in df.index else None}
        # Store the control group value directly

        # Store test group values directly
        for group in self.test_groups:
            data[group] = df.loc[group].item() if group in df.index else None

        return data

    def _associate_pvals_with_groups(self, stat_significance: StatSignificanceResult) -> dict:
        """
        Associates the provided p-values with the corresponding test groups.
        :param stat_significance: StatSignificanceResult instance with the method used and p-values.
        """
        return {group: pval for group, pval in zip(self.test_groups, stat_significance.p_values)}

    def __repr__(self):
        return f"<MetricResult(control_group={self.control_group}, test_groups={self.test_groups}, data={self.data}, stat_significance={self.stat_significance}, stat_significance_method={self.stat_significance_method})>"


class Metric:
    def __init__(self, metrictype: MetricType, metricparams: MetricParams, metricresult: MetricResult):
        """
        Initialize with a MetricType, MetricParams, and MetricResult.
        :param metrictype: metric type (e.g., MetricType.EVENT_COUNT_PER_USER)
        :param metricparams:  MetricParams instance with event name and optional attribute name
        :param metricresult: Metric result with calculated statistical significance and difference
        """
        self.metrictype = metrictype
        # Ensure metricparams is an instance of MetricParams
        if not isinstance(metricparams, MetricParams):
            raise ValueError("metricparams must be an instance of MetricParams")
        self.metricparams = metricparams
        self.result = metricresult

    def __repr__(self):
        return f"<Metric(metrictype={self.metrictype}, metricparams={self.metricparams}, result={self.result})>"
