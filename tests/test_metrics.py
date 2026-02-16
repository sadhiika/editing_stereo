import unittest
from unittest.mock import patch
from biaswipe import metrics

class TestMetrics(unittest.TestCase):

    # Test methods for mock_stereotype_detector
    def test_mock_detector_stereotype_keyword(self):
        self.assertEqual(metrics.mock_stereotype_detector("This is a stereotype."), (True, 0.8))

    def test_mock_detector_stereotyped_keyword_case_insensitive(self):
        self.assertEqual(metrics.mock_stereotype_detector("This is STEREOTYPED."), (True, 0.8))

    def test_mock_detector_stereotypical_keyword(self):
        self.assertEqual(metrics.mock_stereotype_detector("This is stereotypical."), (True, 0.8))

    def test_mock_detector_stereotype_keyword_with_punctuation(self):
        self.assertEqual(metrics.mock_stereotype_detector("Is this a stereotype? Yes."), (True, 0.8))

    def test_mock_detector_no_keyword(self):
        self.assertEqual(metrics.mock_stereotype_detector("This is neutral."), (False, 0.0))

    def test_mock_detector_empty_string(self):
        self.assertEqual(metrics.mock_stereotype_detector(""), (False, 0.0))

    def test_mock_detector_keyword_substring_should_not_match(self):
        # "stereotypes" is not one of the keywords, nor "astereotype"
        self.assertEqual(metrics.mock_stereotype_detector("This concerns astereotype."), (False, 0.0))
        self.assertEqual(metrics.mock_stereotype_detector("This concerns stereotypes."), (False, 0.0))


    # Test methods for compute_stereotype_rate
    def test_csr_all_true(self):
        self.assertAlmostEqual(metrics.compute_stereotype_rate([True, True, True]), 1.0)

    def test_csr_all_false(self):
        self.assertAlmostEqual(metrics.compute_stereotype_rate([False, False]), 0.0)

    def test_csr_mixed(self):
        self.assertAlmostEqual(metrics.compute_stereotype_rate([True, False, True, False, False]), 0.4)

    def test_csr_empty_list(self):
        self.assertAlmostEqual(metrics.compute_stereotype_rate([]), 0.0)

    # Test methods for compute_stereotype_severity_score
    def test_csss_all_stereotyping(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.7, 0.9], [True, True, True]), 0.7)

    def test_csss_some_stereotyping(self):
        # (0.5 + 0.9 + 0.2) / 3 = 1.6 / 3 = 0.5333...
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.7, 0.9, 0.2], [True, False, True, True]), 1.6 / 3)

    def test_csss_none_stereotyping(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.7], [False, False]), 0.0)

    def test_csss_empty_lists(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([], []), 0.0)

    def test_csss_empty_scores_but_labels_true(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([], [True, True]), 0.0)

    def test_csss_scores_but_empty_labels(self):
         self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.6], []), 0.0)

    def test_csss_mismatched_lengths_ignored_extra_scores(self):
        # Only the first two scores (0.5, 0.6) are considered because labels is shorter.
        # (0.5 + 0.6) / 2 = 0.55
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.6, 0.7], [True, True]), 0.55)

    def test_csss_mismatched_lengths_ignored_extra_labels(self):
        # Only the first two scores (0.5, 0.6) are considered because scores is shorter.
        # (0.5 + 0.6) / 2 = 0.55
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.6], [True, True, False]), 0.55)

    def test_csss_one_item_true(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.6, 0.7], [False, True, False]), 0.6)

    def test_csss_one_item_false(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5], [False]), 0.0)

    def test_csss_one_item_true_single_element_lists(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.9], [True]), 0.9)

    # Test methods for compute_csss
    def setUp_csss_data(self):
        self.sample_prompts = {
            "p1": {"text": "t1", "category": "catA"},
            "p2": {"text": "t2", "category": "catB"},
            "p3": {"text": "t3", "category": "catA"},
            "p4": {"text": "t4", "category": "catC"}
        }
        self.prompt_ids = ["p1", "p2", "p3", "p4"]

    @patch('builtins.print')
    def test_csss_basic(self, mock_print):
        self.setUp_csss_data()
        labels = [True, True, True, False]
        scores = [0.8, 0.6, 0.4, 0.9]
        expected = {"catA": (0.8 + 0.4) / 2, "catB": 0.6} # catC has no stereotyping response

        result = metrics.compute_csss(self.sample_prompts, labels, scores, self.prompt_ids)

        self.assertEqual(len(result), len(expected))
        for category, avg_score in expected.items():
            self.assertIn(category, result)
            self.assertAlmostEqual(result[category], avg_score)

    @patch('builtins.print')
    def test_csss_no_stereotyping(self, mock_print):
        self.setUp_csss_data()
        labels = [False, False, False, False]
        scores = [0.8, 0.6, 0.4, 0.9]
        expected = {} # No categories will have stereotyping responses

        result = metrics.compute_csss(self.sample_prompts, labels, scores, self.prompt_ids)

        self.assertEqual(result, expected)

    @patch('builtins.print')
    def test_csss_all_stereotyping_one_category(self, mock_print):
        sample_prompts_one_cat = {"p1": {"text": "t1", "category": "catA"}, "p2": {"text": "t2", "category": "catA"}}
        prompt_ids_one_cat = ["p1", "p2"]
        labels = [True, True]
        scores = [0.8, 0.4]
        expected = {"catA": 0.6}

        result = metrics.compute_csss(sample_prompts_one_cat, labels, scores, prompt_ids_one_cat)

        self.assertEqual(len(result), len(expected))
        for category, avg_score in expected.items():
            self.assertIn(category, result)
            self.assertAlmostEqual(result[category], avg_score)

    @patch('builtins.print')
    def test_csss_empty_inputs(self, mock_print):
        expected = {}
        result = metrics.compute_csss({}, [], [], [])
        self.assertEqual(result, expected)

    @patch('builtins.print')
    def test_csss_mismatched_input_lengths(self, mock_print):
        self.setUp_csss_data()
        result = metrics.compute_csss(self.sample_prompts, [True], [0.5, 0.6], ["p1"])
        self.assertEqual(result, {})
        mock_print.assert_called_once()
        self.assertIn("must have the same length", mock_print.call_args[0][0])

    @patch('builtins.print')
    def test_csss_prompt_id_not_in_prompts(self, mock_print):
        self.setUp_csss_data()
        labels = [True]
        scores = [0.8]
        prompt_ids_invalid = ["p_non_existent"]
        expected = {}

        result = metrics.compute_csss(self.sample_prompts, labels, scores, prompt_ids_invalid)

        self.assertEqual(result, expected)
        mock_print.assert_called_once()
        self.assertIn("not found", mock_print.call_args[0][0])

    @patch('builtins.print')
    def test_csss_prompt_missing_category_key(self, mock_print):
        sample_prompts_no_cat_key = {"p1": {"text": "t1"}} # No "category" key
        labels = [True]
        scores = [0.8]
        prompt_ids_no_cat_key = ["p1"]
        expected = {}

        result = metrics.compute_csss(sample_prompts_no_cat_key, labels, scores, prompt_ids_no_cat_key)

        self.assertEqual(result, expected)
        mock_print.assert_called_once()
        self.assertIn("'category' key missing", mock_print.call_args[0][0])

    @patch('builtins.print')
    def test_csss_category_not_string(self, mock_print):
        self.setUp_csss_data()
        prompts_cat_not_string = {"p1": {"text": "t1", "category": 123}} # Category is int
        labels = [True]
        scores = [0.8]
        prompt_ids_cat_not_string = ["p1"]
        expected = {}

        result = metrics.compute_csss(prompts_cat_not_string, labels, scores, prompt_ids_cat_not_string)

        self.assertEqual(result, expected)
        mock_print.assert_called_once()
        self.assertIn("not a string", mock_print.call_args[0][0])

    # Test methods for compute_wosi
    @patch('builtins.print')
    def test_wosi_basic(self, mock_print):
        csss = {"catA": 0.8, "catB": 0.6}
        weights = {"catA": 0.7, "catB": 0.3} # Weights sum to 1
        # Expected: (0.8 * 0.7) + (0.6 * 0.3) = 0.56 + 0.18 = 0.74
        # Denominator: 0.7 + 0.3 = 1.0
        # WOSI = 0.74 / 1.0 = 0.74
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.74)

    @patch('builtins.print')
    def test_wosi_weights_do_not_sum_to_one(self, mock_print):
        csss = {"catA": 0.8, "catB": 0.6}
        weights = {"catA": 2.0, "catB": 1.0}
        # Expected: ((0.8 * 2.0) + (0.6 * 1.0)) / (2.0 + 1.0)
        # = (1.6 + 0.6) / 3.0 = 2.2 / 3.0
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 2.2 / 3.0)

    @patch('builtins.print')
    def test_wosi_category_in_csss_not_in_weights(self, mock_print):
        csss = {"catA": 0.8, "catB": 0.6, "catC": 0.9} # catC not in weights
        weights = {"catA": 0.7, "catB": 0.3}
        # Expected: ((0.8 * 0.7) + (0.6 * 0.3)) / (0.7 + 0.3) = 0.74 / 1.0 = 0.74
        # catC is ignored.
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.74)
        mock_print.assert_called_once()
        self.assertIn("not in category weights", mock_print.call_args[0][0])

    @patch('builtins.print')
    def test_wosi_category_in_weights_not_in_csss(self, mock_print):
        csss = {"catA": 0.8} # catB from weights is not here
        weights = {"catA": 0.7, "catB": 0.3}
        # Expected: (0.8 * 0.7) / 0.7 = 0.8
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.8)

    def test_wosi_empty_csss(self):
        csss = {}
        weights = {"catA": 0.7, "catB": 0.3}
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.0)

    @patch('builtins.print')
    def test_wosi_empty_weights(self, mock_print):
        csss = {"catA": 0.8, "catB": 0.6}
        weights = {}
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.0)
        mock_print.assert_called_once()
        self.assertIn("Category weights are empty", mock_print.call_args[0][0])

    @patch('builtins.print')
    def test_wosi_no_matching_categories(self, mock_print):
        csss = {"catA": 0.8, "catB": 0.6}
        weights = {"catC": 0.7, "catD": 0.3} # No common categories
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.0)
        self.assertEqual(mock_print.call_count, 3) # 2 for missing, 1 for zero sum

    @patch('builtins.print')
    def test_wosi_invalid_weight_type(self, mock_print):
        csss = {"catA": 0.8, "catB": 0.5, "catC": 0.9}
        weights = {"catA": "invalid_weight", "catB": 0.5, "catC": -0.1} # catA invalid, catC negative
        # Expected: (0.5 * 0.5) / 0.5 = 0.5
        # catA skipped due to invalid weight type, catC skipped due to negative weight
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.5)
        self.assertEqual(mock_print.call_count, 2)

    @patch('builtins.print')
    def test_wosi_all_weights_zero_or_invalid(self, mock_print):
        csss = {"catA": 0.8, "catB": 0.6}
        weights = {"catA": 0.0, "catB": "invalid"}
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.0)
        self.assertEqual(mock_print.call_count, 2)

    @patch('builtins.print')
    def test_wosi_single_category_match_zero_weight(self, mock_print):
        csss = {"catA": 0.8, "catB": 0.5}
        weights = {"catA": 0.0, "catC": 0.5} # catA has zero weight, catB not in weights
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.0)
        self.assertEqual(mock_print.call_count, 2)


if __name__ == '__main__':
    unittest.main()