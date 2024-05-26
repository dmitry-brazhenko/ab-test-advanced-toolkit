from enum import Enum, auto
from typing import List, Union, Optional

import numpy as np
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

import pandas as pd
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from xgboost import XGBRegressor

import logging

logger = logging.getLogger(__name__)


class StatSignificanceMethod(Enum):
    CHI_SQUARE = auto()
    T_TEST = auto()
    PURE_CUPED_T_TEST = auto()
    GBOOST_CUPED_T_TEST = auto()


class StatSignificanceResult:
    def __init__(self, method_used: StatSignificanceMethod, p_values: List[float]):
        """
        Initializes the StatSignificanceResult with the method used, p-values for each test group.

        :param method_used: The statistical test method used.
        :param p_values: A list of p-values corresponding to each test group comparison with the control group.
        """
        self.method_used = method_used
        self.p_values = p_values

    def __repr__(self):
        return f"StatSignificanceResult(method_used={self.method_used}, p_values={self.p_values})"


class ZeroPredictor:
    def fit(self, X, y):
        # This model doesn't need to learn anything, so `fit` is a no-op
        pass

    def predict(self, X):
        # Return an array of zeros with the same length as the input
        return np.zeros(len(X))


class StatTests:
    @staticmethod
    def calculate_t_test_for_dataset(merged_intest: pd.DataFrame, control_group: str,
                                     test_groups: List[str]) -> StatSignificanceResult:
        """
        Calculates the T-test for the means of two independent samples within a dataset
        for each test group compared to the control group.

        :param merged_intest: DataFrame containing the data. Expected padnas format: | userid (index) | abgroup | value_column |
        :param control_group: Identifier for the control group. For example, 'A'.
        :param test_groups: List of identifiers for the test groups. For example, ['B', 'C'].
        :return: StatSignificanceResult instance with p-values and the method used.
        """
        value_column = merged_intest.columns[-1]
        p_values = []
        control_values = merged_intest[merged_intest['abgroup'] == control_group][value_column]
        for test_group in test_groups:
            test_values = merged_intest[merged_intest['abgroup'] == test_group][value_column]

            _, p_value = stats.ttest_ind(control_values, test_values, equal_var=False, nan_policy='omit')
            p_values.append(p_value)

        return StatSignificanceResult(StatSignificanceMethod.T_TEST, p_values)

    @staticmethod
    def calculate_cuped_and_compare(merged_pretest: pd.DataFrame, merged_intest: pd.DataFrame, control_group: str,
                                    test_groups: List[str]) -> StatSignificanceResult:
        """
        Calculate the CUPED adjustment and compare the adjusted test group values to the control group using T-tests.
        :param merged_pretest: DataFrame containing the pretest data. Expected pandas format: | userid (index) | abgroup | value_column |
        :param merged_intest: DataFrame containing the intest data. Expected pandas format: | userid (index) | abgroup | value_column |
        :param control_group: Identifier for the control group. For example, 'A'.
        :param test_groups: List of identifiers for the test groups. For example, ['B', 'C'].
        :return:
        """
        value_column = merged_pretest.columns[-1]
        # Extract control group pretest and intest values for the regression model
        control_pretest_values = merged_pretest[merged_pretest['abgroup'] == control_group][
            value_column].values.reshape(-1, 1)
        control_intest_values = merged_intest[merged_intest['abgroup'] == control_group][value_column].values

        # Fit linear regression model on control group data
        model = LinearRegression()
        model.fit(control_pretest_values, control_intest_values)
        logger.debug(f"Control pretest values: {control_pretest_values}")
        logger.debug(f"Control intest values: {control_intest_values}")
        logger.debug(f"Model coefficients: {model.coef_}")
        adjusted_control_values = control_intest_values - model.predict(control_pretest_values)

        # Initialize lists for storing p-values and test group names
        p_values = []
        test_group_names = []

        # Iterate over test groups to perform adjustments and T-tests
        for test_group in test_groups:
            test_group_names.append(test_group)

            # Apply model to test group pretest values
            test_pretest_values = merged_pretest[merged_pretest['abgroup'] == test_group][value_column].values.reshape(
                -1, 1)
            test_intest_values = merged_intest[merged_intest['abgroup'] == test_group][value_column].values

            # Calculate the adjusted values for the test group
            predicted_test_intest_values = model.predict(test_pretest_values)
            adjusted_test_values = test_intest_values - predicted_test_intest_values

            # Perform T-test between adjusted control and test group values
            _, p_value = stats.ttest_ind(
                adjusted_control_values,
                adjusted_test_values, equal_var=False)
            p_values.append(p_value)

        return StatSignificanceResult(StatSignificanceMethod.PURE_CUPED_T_TEST, p_values)

    @staticmethod
    def calculate_gboost_cuped_and_compare(merged_pretest: pd.DataFrame, merged_intest: pd.DataFrame,
                                             user_properties: Optional[pd.DataFrame], control_group: str,
                                             test_groups: List[str], use_enhansement: bool = False) -> StatSignificanceResult:
        """
        Calculate the CUPED adjustment and compare the adjusted test group values to the control group using T-tests.
        :param merged_pretest: DataFrame containing the pretest data. Expected pandas format: | userid (index) | abgroup | value_column |
        :param merged_intest: DataFrame containing the intest data. Expected pandas format: | userid (index) | abgroup | value_column |
        :param user_properties: DataFrame containing user properties. Expected pandas format: | userid (index) | property1 | property2 | ...
        :param control_group: Identifier for the control group. For example, 'A'.
        :param test_groups: List of identifiers for the test groups. For example, ['B', 'C'].
        :return:
        """
        # Ensure `value_column` is defined to match your actual data structure
        value_column = merged_intest.columns[-1]  # Assuming last column is the metric of interest

        # Prepare merged dataset with user properties if available
        def prepare_dataset(pretest_data: pd.DataFrame, user_properties: Optional[pd.DataFrame]):
            if user_properties is not None and not user_properties.empty:
                X = pd.merge(pretest_data, user_properties, left_on='userid', right_on='userid', how='left').drop(
                    columns=['abgroup'])
                categorical_features = user_properties.select_dtypes(include=['object', 'category']).columns.tolist()
            else:
                X = pretest_data.drop(columns=['abgroup'])
                categorical_features = []
            return X, categorical_features

        control_pretest_data = merged_pretest[merged_pretest['abgroup'] == control_group]

        X_control, categorical_features = prepare_dataset(control_pretest_data, user_properties)

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', TargetEncoder(smoothing=0.8), categorical_features)
            ],
            remainder='passthrough'
        )

        # Embed the preprocessing step into a pipeline with XGBRegressor
        model: Union[XGBRegressor, ZeroPredictor] = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(verbosity=0))
        ])
        logger.debug(f"use_enhansement: {use_enhansement}")
        if use_enhansement:
            try:
                logger.debug(f"X_control: {X_control}")
                logger.debug(f"merged_intest[merged_intest['abgroup'] == control_group][value_column]: {merged_intest[merged_intest['abgroup'] == control_group][value_column]}")
                x_ = X_control.copy()
                x_.reset_index(drop=True, inplace=True)

                y_ = merged_intest[merged_intest['abgroup'] == control_group][value_column].copy()
                y_.reset_index(drop=True, inplace=True)
                model.fit(x_, y_)
                # model.fit(X_control, merged_intest[merged_intest['abgroup'] == control_group][value_column])
                logger.debug(f"Model was fit")
            except Exception as e:
                logger.error(f"Model was not fit. Error: {e}")
                # this case is equivalent o regular T-test
                model = ZeroPredictor()
        else:
            model = ZeroPredictor()

        adjusted_values = {}
        p_values = []
        test_group_names = []

        # Adjust and calculate deltas for control and test groups
        for group in [control_group] + test_groups:
            group_pretest_data, _ = prepare_dataset(merged_pretest[merged_pretest['abgroup'] == group], user_properties)
            group_intest_data = merged_intest[merged_intest['abgroup'] == group][value_column]

            if not group_pretest_data.empty and model is not None:
                y_predicted = model.predict(group_pretest_data)
                adjusted_values[group] = group_intest_data.values - y_predicted

        # Perform T-tests for statistical significance between control and test group adjustments
        for test_group in test_groups:
            if test_group in adjusted_values and control_group in adjusted_values:
                _, p_value = stats.ttest_ind(adjusted_values[control_group], adjusted_values[test_group],
                                             equal_var=False)
                p_values.append(p_value)
                test_group_names.append(test_group)

        # Assuming StatSignificanceResult is a structure you've defined to store the results
        return StatSignificanceResult(StatSignificanceMethod.GBOOST_CUPED_T_TEST, p_values)
