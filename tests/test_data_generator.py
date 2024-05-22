import unittest
import pandas as pd

from ab_test_advanced_toolkit.metrics import Metric
from ab_test_advanced_toolkit.stat_significance import StatSignificanceResult
from data_generation.data_generator import generate_synthetic_data, run_analysis

class TestDataGenerator(unittest.TestCase):
    
    def test_generate_synthetic_data(self):
        num_users = 1000
        countries = ['US', 'UK', 'DE', 'FR', 'CA', 'AU', 'JP', 'IN']
        platforms = ['iOS', 'Android', 'Web', 'Desktop']
        user_segments = ['Segment_1', 'Segment_2', 'Segment_3', 'Segment_4']
        ab_groups = ['a1', 'a2', 'b']
        
        data = generate_synthetic_data(
            num_users=num_users,
            countries=countries,
            platforms=platforms,
            user_segments=user_segments,
            ab_groups=ab_groups,
            noise_level=1.0,
            base_increase_percentage=0.05,
            correlation_level=0.5,
        )
        
        # Check if the data is generated correctly
        self.assertEqual(len(data), num_users)
        self.assertTrue('userid' in data.columns)
        self.assertTrue('country' in data.columns)
        self.assertTrue('platform' in data.columns)
        self.assertTrue('user_segment' in data.columns)
        self.assertTrue('abgroup' in data.columns)
        self.assertTrue('value' in data.columns)
        self.assertTrue('pre_test_value' in data.columns)

    def test_run_analysis(self):
        num_users = 1000
        countries = ['US', 'UK', 'DE', 'FR', 'CA', 'AU', 'JP', 'IN']
        platforms = ['iOS', 'Android', 'Web', 'Desktop']
        user_segments = ['Segment_1', 'Segment_2', 'Segment_3', 'Segment_4']
        ab_groups = ['a1', 'a2', 'b']
        
        data = generate_synthetic_data(
            num_users=num_users,
            countries=countries,
            platforms=platforms,
            user_segments=user_segments,
            ab_groups=ab_groups,
            noise_level=1.0,
            base_increase_percentage=0.05,
            correlation_level=0.5,
        )
        
        results = run_analysis(data)
        
        # Check if the analysis results are not empty
        self.assertTrue("no_enhancement" in results)
        self.assertTrue("cuped" in results)
        self.assertTrue("gboost_cuped" in results)

        # output type is Metric
        self.assertIsInstance(results["no_enhancement"], Metric)
        self.assertIsInstance(results["cuped"], Metric)
        self.assertIsInstance(results["gboost_cuped"], Metric)


if __name__ == '__main__':
    unittest.main()
