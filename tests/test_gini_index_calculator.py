import unittest

import pandas as pd

from gini_index_calculator import GiniCalculator


class TestDecisionTreeClassifier(unittest.TestCase):

    data = {
        #'id': pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        'result': pd.Series(['Pass', 'Fail', 'Fail', 'Pass', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Fail', 'Fail', 'Fail']),
        'other_courses': pd.Series(['Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'N', 'Y', 'N', 'N']),
        'major': pd.Series(['Math', 'Math', 'Math', 'CS', 'Other', 'Other', 'Math', 'CS', 'Math', 'CS', 'CS', 'Math', 'Other', 'Other', 'Math']),
        'working': pd.Series(['N', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y'])
    }
    test_data = pd.DataFrame(data=data)

    gini_calculator = GiniCalculator(x_categories=['other_courses', 'major', 'working'], y_category='result',
                                     training_data=test_data)

    def test_gini_index_calculation(self):
        assert self.gini_calculator._find_possible_outcomes().size == 2  # can only pass or fail the class
        expected_gini_index_for_math_students = 1 - ((4/7) ** 2 + (3/7) ** 2)  # prob(pass | math)^2 + prob(fail | math)^2

        self.gini_calculator._find_input_set_for_all_categories()
        self.gini_calculator._calculate_gini_index_split('major', 'Math')
        assert expected_gini_index_for_math_students == self.gini_calculator.gini_indices['major']['Math']

    def test_all_gini_index_calculations(self):
        expected_gini_index_for_math_students = 1 - ((4/7) ** 2 + (3/7) ** 2)
        expected_gini_index_for_cs_students = 1 - ((4/4) ** 2 + (0/4) ** 2)
        expected_gini_index_for_other_students = 1 - ((0/4) ** 2 + (4/4) ** 2)
        overall_gini_index_for_major_category = expected_gini_index_for_math_students * (7/15) + expected_gini_index_for_cs_students * (4/15) \
            + expected_gini_index_for_other_students * (4/15)

        self.gini_calculator._find_input_set_for_all_categories()
        self.gini_calculator.calculate_gini_indices()

        assert expected_gini_index_for_math_students == self.gini_calculator.gini_indices['major']['Math']
        assert expected_gini_index_for_cs_students == self.gini_calculator.gini_indices['major']['CS']
        assert expected_gini_index_for_other_students == self.gini_calculator.gini_indices['major']['Other']
        assert overall_gini_index_for_major_category == self.gini_calculator.gini_indices['major']['overall']

