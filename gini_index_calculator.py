

class GiniCalculator:

    def __init__(self, x_categories, y_category, training_data):
        self.x_categories, self.y_category = x_categories, y_category
        self.training_data = training_data
        self.category_to_input_set = {}
        self.gini_indices = {}

    def _find_input_set_for_all_categories(self):
        for category in self.x_categories:
            self.category_to_input_set[category] = self._find_input_set_for_category(category)

    def _find_input_set_for_category(self, category):
        return self.training_data[category].unique()

    def _find_possible_outcomes(self):
        return self._find_input_set_for_category(self.y_category)

    def calculate_gini_indices(self):
        for category in self.x_categories:
            if category == self.y_category:
                continue
            for category_value in self._find_input_set_for_category(category):
                self._calculate_gini_index_split(category, category_value)
            self._calculate_overall_gini_index(category)

    def _calculate_gini_index_split(self, category, category_value):
        squared_sum_of_probabilities_given_category = 0
        category_df = self.training_data[[category, self.y_category]].copy()
        category_df = category_df[category_df[category] == category_value]
        for outcome in self._find_possible_outcomes():
            prob_of_outcome = category_df[category_df[self.y_category] == outcome].size / category_df.size
            squared_sum_of_probabilities_given_category += prob_of_outcome ** 2
        # update self.gini_indices dataframe values
        if not self.gini_indices.get(category):
            self.gini_indices[category] = {}
        self.gini_indices[category][category_value] = 1 - squared_sum_of_probabilities_given_category

    def _calculate_overall_gini_index(self, category):
        overall_gini_index = 0
        for key, gini_index_for_value in self.gini_indices[category].items():
            overall_gini_index += self._weighted_avg(category, key) * gini_index_for_value
        self.gini_indices[category]['overall'] = overall_gini_index

    def _weighted_avg(self, column_name, column_value):
        series = self.training_data[column_name]
        return series[series == column_value].size / series.size
