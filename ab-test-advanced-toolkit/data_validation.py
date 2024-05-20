import pandas as pd


def validate_data(event_data: pd.DataFrame, ab_test_allocations: pd.DataFrame, control_group_name: str) -> None:
    """
    Validates the event data and AB test allocations DataFrame.

    Parameters:
    - event_data: DataFrame containing event data with columns ["timestamp", "userid", "event_name", ...].
    - ab_test_allocations: DataFrame containing AB test allocations with columns ["timestamp", "userid", "abgroup"].
    - control_group_name: The name of the control group to check for its presence in the AB test allocations.
    """

    # Validation for AB test allocations
    required_ab_columns = ["timestamp", "userid", "abgroup"]
    if not all(column in ab_test_allocations.columns for column in required_ab_columns):
        raise ValueError(f"AB test allocations must have the required columns: {required_ab_columns}")

    if not pd.api.types.is_datetime64_any_dtype(ab_test_allocations['timestamp']):
        raise ValueError("The 'timestamp' column in ab_test_allocations must be of datetime type.")

    if control_group_name not in ab_test_allocations['abgroup'].unique():
        raise ValueError(f"Control group '{control_group_name}' not found in ab_test_allocations.")

    if ab_test_allocations['userid'].duplicated().any():
        raise ValueError("Duplicate userids found in ab_test_allocations. Please ensure each userid is unique.")

    # Validation for event data
    required_event_columns = ["timestamp", "userid", "event_name"]
    if not all(column in event_data.columns for column in required_event_columns):
        raise ValueError(f"Event data must have the required columns: {required_event_columns}")

    if not pd.api.types.is_datetime64_any_dtype(event_data['timestamp']):
        raise ValueError("The 'timestamp' column in event_data must be of datetime type.")
