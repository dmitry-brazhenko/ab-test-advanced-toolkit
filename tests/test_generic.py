import pytest
import logging

from ab_test_advanced_toolkit import ABTestAnalyzer
from ab_test_advanced_toolkit.metrics import MetricType
from tests.test_utils import generate_event_data, generate_user_properties, generate_user_allocations

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_library_is_initialized():
    user_properties = generate_user_properties()
    event_data = generate_event_data()
    ab_test_allocations = generate_user_allocations()

    # initializing with user properties
    analyzer = ABTestAnalyzer(event_data, ab_test_allocations, "A", user_properties, mode="gboost_cuped")

    # initializing without user properties
    analyzer = ABTestAnalyzer(event_data, ab_test_allocations, "A", mode="gboost_cuped")


def test_library_is_not_initialized():
    # in case of invalid data input
    event_data = generate_event_data()
    ab_test_allocations = generate_user_allocations()
    event_data['timestamp'] = 0

    # initializing without user properties
    with pytest.raises(Exception) as e_info:
        ABTestAnalyzer(event_data, ab_test_allocations, "A", mode="gboost_cuped")
    assert str("The 'timestamp' column in event_data must be of datetime type.") == str(e_info.value)



# metrics do not differe regardless of user_properties
@pytest.mark.parametrize("user_properties", [generate_user_properties(), None])
def test_calculate_metrics(user_properties):
    event_data = generate_event_data()
    ab_test_allocations = generate_user_allocations()

    analyzer = ABTestAnalyzer(event_data, ab_test_allocations, "A", user_properties, mode="gboost_cuped")

    event_counts = analyzer.calculate_event_count_per_user('purchase')

    assert event_counts.metrictype == MetricType.EVENT_COUNT_PER_USER

    assert event_counts.result.data['A'] == pytest.approx(0.283019, rel=1e-3)
    assert event_counts.result.data['B'] == pytest.approx(0.404255, rel=1e-3)
    assert len(analyzer.calculated_metrics) == 1

    event_attribute_sum = analyzer.calculate_event_attribute_sum_per_user('purchase', 'purchase_value')

    assert event_attribute_sum.result.data['A'] == pytest.approx(53.283019, rel=1e-3)
    assert event_attribute_sum.result.data['B'] == pytest.approx(91.553191, rel=1e-3)
    assert len(analyzer.calculated_metrics) == 2

    conversion_rate = analyzer.calculate_conversion('purchase')
    logger.debug(conversion_rate)
    assert conversion_rate.result.data['A'] == pytest.approx(0.075472, rel=1e-3)
    assert conversion_rate.result.data['B'] == pytest.approx(0.085106, rel=1e-3)
    assert len(analyzer.calculated_metrics) == 3

    # testing that report does not fail and file is saved
    analyzer.save_report("tempfile.html")


# simple check that significance metrics are different due to user properties being used
def test_calculate_significance():
    event_data = generate_event_data()
    ab_test_allocations = generate_user_allocations()

    user_properties = generate_user_properties()
    analyzer_user_properties = ABTestAnalyzer(event_data, ab_test_allocations, "A", user_properties, mode="gboost_cuped")
    analyzer_no_user_properties = ABTestAnalyzer(event_data, ab_test_allocations, "A", mode="gboost_cuped")

    analyzer_user_properties.calculate_event_count_per_user('purchase')
    analyzer_no_user_properties.calculate_event_count_per_user('purchase')

    assert abs(analyzer_user_properties.calculated_metrics[0].result.stat_significance['B'] -
               analyzer_no_user_properties.calculated_metrics[0].result.stat_significance['B']) > 1e-3
