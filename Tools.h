#ifndef TREE_TOOLS_H
#define TREE_TOOLS_H

#include <vector>
#include <string>
#include <map>
#include "Abstract_regressor.h"
#include "Table.h"

/// Mean absolute error calculation function
double mean_absolute_error(
    const std::vector<std::vector<double>> &observation, ///< Actual value matrix
    const std::vector<std::vector<double>> &predictions  ///< Estimated value matrix
);

/// Function of splitting the original set into test and training sets
std::pair<Table, Table> train_test_split(
    const Table &data, ///< Original dataset
    int n_tests        ///< Number of rows in the test set
);

/// Function to check the quality of the model using the walk forward validation method
double walk_forward_validation(
    Abstract_regressor &regressor, ///< Model under test
    const Table &data,             ///< Data set for test
    int n_test,                    ///< Number of tests
    int n_observation              ///< Number of observations on which the model will be trained
);

/// Time series to data transformation function for supervised learning
Table series_to_supervised(
    const Table &data, ///< Original dataset
    int n_in,          ///< Number of features in the new dataset
    int n_out          ///< Number of observation in the new dataset
);

#endif //TREE_TOOLS_H
