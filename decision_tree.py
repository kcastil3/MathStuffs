import pandas as pd

from gini_index_calculator import GiniCalculator


class DecisionNode:

    def __init__(self, criteria_key, choose_left_threshold_value, left_child, right_child):
        self.criteria_key = criteria_key  # the key "path" to the gini index/entropy
        self.choose_left_threshold_value = choose_left_threshold_value
        self.left_child, self.right_child = left_child, right_child


class DecisionTreeClassifier:

    def __init__(self, x_categories, y_category, training_data, use_entropy=False, skip_columns_for_training=None):
        self.x_categories, self.y_category = x_categories, y_category
        self.training_data = training_data[[col for col in training_data if skip_columns_for_training and col not in skip_columns_for_training]]
        self.use_entropy = use_entropy  # by default use Gini index
        self.is_trained = False
        self.gini_indices = pd.DataFrame(columns=['result', 'column_name', 'column_value'])

    def train_classifier(self):
        gini_indices = self._calculate_gini_indices()
        # Now we need to determine the maximum

    def make_prediction(self, x_values):
        pass

    def _calculate_gini_indices(self):
        gini_calculator = GiniCalculator(self.x_categories, self.y_category, self.training_data)
        return gini_calculator.calculate_gini_indices()

    def _cast_series_to_one_hot_encoding(self):
        pass

