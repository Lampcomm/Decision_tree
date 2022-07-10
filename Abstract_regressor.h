#ifndef TREE_ABSTRACT_REGRESSOR_H
#define TREE_ABSTRACT_REGRESSOR_H

#include <vector>
#include "Table.h"

class Abstract_regressor {
public:
    /// Model training function
    virtual void fit(
        const Table &x,                           ///< Feature set
        const std::vector<std::vector<double>> &y ///< Feature-related observations
    ) = 0;

    /// Prediction function for one set of features
    virtual std::vector<double> predict(
        const std::vector<double> &values ///< One feature sets
    ) const = 0;

    /// Prediction function for multiple feature sets
    virtual std::vector<std::vector<double>> predict(
        const Table &values ///< Multiple feature sets
    ) const = 0;
};


#endif
