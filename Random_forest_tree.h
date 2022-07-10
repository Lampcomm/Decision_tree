#ifndef TREE_RANDOM_FOREST_TREE_H
#define TREE_RANDOM_FOREST_TREE_H

#include <vector>
#include <map>
#include <string>
#include <tuple>
#include <memory>
#include <unordered_set>
#include "Abstract_regressor.h"

class Random_forest_tree : public Abstract_regressor {
public:
    explicit Random_forest_tree(
        double X_features_fraction = 1.0, ///< Proportion of features used
        size_t min_samples_split = 20,    ///< Minimum sample size that can be at the node
        size_t max_depth = 5              ///< Maximum tree depth
    );

    /// Model training function
    void fit(
        const Table &x,                           ///< Feature set
        const std::vector<std::vector<double>> &y ///< Feature-related observations
    ) override;

    /// Tree information output function
    void print_tree() const;

    /// Prediction function for one set of features
    std::vector<double> predict(
        const std::vector<double> &values ///< One feature set
    ) const override;

    /// Prediction function for multiple feature sets
    std::vector<std::vector<double>> predict(
        const Table &values ///< Multiple feature sets
    ) const override;

private:
    /// Function of obtaining the average for each column of the matrix
    static std::vector<double> get_mean(
        const std::vector<std::vector<double>> &arr ///< Matrix
    );

    /// Moving average function
    static std::vector<double> get_ma(
        const std::vector<double> &arr, ///< Array of values
        std::vector<int> indices        ///< Index array for arr sorted in non-descending order
    );

    /// Mean square error calculation function
    static long double get_mse(
        const std::vector<std::vector<double>> &arr, ///< Actual value matrix
        const std::vector<double> &value,            ///< Estimated values array
        double n                                     ///< Mean square error denominator
    );

    /// Function of calculating the best value and the best feature number for splitting samples
    std::pair<int, double> get_best_split(
        const Table &x,                           ///< Feature set
        const std::vector<std::vector<double>> &y ///< Feature-related observations
    ) const;

    /// Function of calculating a set of random non-repeating feature numbers
    std::unordered_set<size_t> get_features(
        size_t n_features ///< Number of features
    ) const;

    /// Function of splitting a set of features and related observations into two parts
    std::tuple<Table, std::vector<std::vector<double>>, Table, std::vector<std::vector<double>>>
    split(
        const Table &x,                           ///< Feature set
        const std::vector<std::vector<double>> &y ///< Feature-related observations
    ) const;

    /// Node information output function
    void print_info(size_t width = 4) const;

private:
    constexpr static int window = 2;           ///< Window size
    char node_type;                            ///< Node type (0 - Root node, 1 - Left node, 2 - Right node)
    int best_feature;                          ///< Number of the best feature to split samples
    size_t min_samples_split;                  ///< Minimum sample size that can be at the node
    size_t max_depth;                          ///< Maximum tree depth
    size_t depth;                              ///< Current tree depth
    size_t samples_size;                       ///< Current sample size in node
    double best_value;                         ///< Best value to split samples
    double X_features_fraction;                ///< Proportion of features used (Accepts values from 0.0 to 1.0)
    std::vector<double> ymean;                 ///< Node prediction
    std::unique_ptr<Random_forest_tree> left;  ///< Pointer to the left child of the node
    std::unique_ptr<Random_forest_tree> right; ///< Pointer to the right child of the node

    long double mse;                           ///< Node mean square error
};


#endif //TREE_RANDOM_FOREST_TREE_H
