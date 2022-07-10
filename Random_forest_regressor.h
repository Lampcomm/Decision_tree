#ifndef TREE_RANDOM_FOREST_REGRESSOR_H
#define TREE_RANDOM_FOREST_REGRESSOR_H

#include <vector>
#include <map>
#include <string>
#include "Random_forest_tree.h"
#include "Abstract_regressor.h"


class Random_forest_regressor : public Abstract_regressor{
public:
    explicit Random_forest_regressor(
        size_t n_trees = 30,              ///< Number of trees
        double X_features_fraction = 1.0, ///< Proportion of features used (Accepts values from 0.0 to 1.0)
        double X_obs_fraction = 1.0,      ///< Proportion of rows used from the training set (Accepts values from 0.0 to 1.0)
        size_t min_samples_split = 20,    ///< Minimum sample size that can be at the tree node
        size_t max_depth = 5              ///< Maximum tree depth
    );

    /// Model training function
    void fit(
        const Table &x,                           ///< Feature set
        const std::vector<std::vector<double>> &y ///< Feature-related observations
    ) override;

    /// Prediction function for one set of features
    std::vector<double> predict(
        const std::vector<double> &values ///< One feature set
    ) const override;

    /// Prediction function for multiple feature sets
    std::vector<std::vector<double>> predict(
        const Table &values ///< Multiple feature sets
    ) const override;

    /// Function to display information about all trees
    void print_trees() const;

private:
    /// Function that creates a bootstrapped sample
    std::pair<Table, std::vector<std::vector<double>>>
            bootstrap_sample(
                const Table &x,                           ///< Feature set
                const std::vector<std::vector<double>> &y ///< Feature-related observations
            ) const;

private:
    size_t min_samples_split;              ///< Minimum sample size that can be at the tree node
    size_t max_depth;                      ///< Maximum tree depth
    size_t y_shape;                        ///< Number of observations
    std::vector<Random_forest_tree> trees; ///< Array of trees
    double X_features_fraction;            ///< Proportion of features used (Accepts values from 0.0 to 1.0)
    double X_obs_fraction;                 ///< Proportion of rows used from the training set (Accepts values from 0.0 to 1.0)
};


#endif //TREE_RANDOM_FOREST_REGRESSOR_H
