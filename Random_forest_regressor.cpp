#include "Random_forest_regressor.h"

#include <random>
#include <future>
#include <iostream>
#include "omp.h"

Random_forest_regressor::Random_forest_regressor(size_t n_trees, double X_features_fraction, double X_obs_fraction,
                                                 size_t min_samples_split, size_t max_depth) : X_features_fraction(X_features_fraction),
                                                                                               X_obs_fraction(X_obs_fraction), min_samples_split(min_samples_split),
                                                                                               max_depth(max_depth), y_shape(0)
{
    if (this->X_obs_fraction > 1.0 || this->X_obs_fraction < std::numeric_limits<double>::epsilon()) {
        throw std::invalid_argument("X_obs_fraction must be in the interval (0.0, 1.0] ");
    }

    this->trees.reserve(n_trees);
    for (size_t i = 0; i < n_trees; ++i) {
        this->trees.emplace_back(this->X_features_fraction, this->min_samples_split, this->max_depth);
    }
}

std::pair<Table, std::vector<std::vector<double>>>
Random_forest_regressor::bootstrap_sample(const Table &x,
                                          const std::vector<std::vector<double>> &y) const
{
    std::pair<Table, std::vector<std::vector<double>>> ans;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> distribution(0, y.size() - 1);

    auto n = static_cast<size_t>(static_cast<double>(y.size()) * this->X_obs_fraction);

    if (!n) {
        n = 1;
    }

    ans.second.reserve(n);
    ans.first.set_column_count(x.get_columns_count());

    for (int i = 0; i < n; ++i) {
        auto index = distribution(gen);
        ans.second.push_back(y.at(index));
        ans.first.push_back_row(x.get_row(index));
    }

    return ans;
}

std::vector<double> Random_forest_regressor::predict(const std::vector<double> &values) const {

    if (!this->y_shape) {
        return {0};
    }

    std::vector<double> ans(this->y_shape, 0);

    for (const auto &i : this->trees) {
        auto vec = i.predict(values);

        for (size_t j = 0; j < vec.size(); ++j) {
            ans[j] += vec[j] / static_cast<double>(this->trees.size());
        }
    }

    return ans;
}

std::vector<std::vector<double>> Random_forest_regressor::predict(const Table &values) const {
    if (!this->y_shape) {
        return {values.get_rows_count(), std::vector<double>(1, 0)};
    }


    std::vector<std::vector<double>> ans(values.get_rows_count(), std::vector<double>(this->y_shape, 0));

    std::vector<std::vector<std::vector<double>>> partial_sums(omp_get_max_threads(),
                                                           std::vector<std::vector<double>>(values.get_rows_count(),
                                                                   std::vector<double>(this->y_shape, 0)));

#pragma omp parallel for default(none) shared(values, ans, partial_sums)
    for (const auto &i : this->trees) {
        auto vec = i.predict(values);

        int thread_id = omp_get_thread_num();
            for (size_t j = 0; j < vec.size(); ++j) {
                for (size_t k = 0; k < vec[j].size(); ++k) {
                    partial_sums[thread_id][j][k] += vec[j][k] / static_cast<double>(this->trees.size());
                }
            }
    }

    for (auto &sum : partial_sums) {
        for (size_t j = 0; j < sum.size(); ++j) {
            for (size_t k = 0; k < sum[j].size(); ++k) {
                ans[j][k] += sum[j][k];
            }
        }
    }

    return ans;
}

void Random_forest_regressor::print_trees() const {
    for (auto it = this->trees.cbegin(); it != this->trees.cend(); ++it) {
        std::cout << "------ \n" << "Tree number: " << it - this->trees.cbegin() + 1 << std::endl;
        it->print_tree();
        std::cout << "------ \n";
    }
}

void Random_forest_regressor::fit(const Table &x,
                                  const std::vector<std::vector<double>> &y)
{
    this->y_shape = y.front().size();

#pragma omp parallel for shared(x, y) default(none)
    for (auto &i : this->trees) {
        auto new_data = bootstrap_sample(x, y);
        i.fit(new_data.first, new_data.second);
    }
}
