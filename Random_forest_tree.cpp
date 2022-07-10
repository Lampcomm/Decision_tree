#include "Random_forest_tree.h"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <utility>
#include <random>
#include <unordered_set>
#include <limits>

Random_forest_tree::Random_forest_tree(double X_features_fraction, size_t min_samples_split, size_t max_depth) :
                                       X_features_fraction(X_features_fraction),
                                       min_samples_split(min_samples_split), max_depth(max_depth), depth(0),
                                       best_value(0.0),
                                       samples_size(0), ymean{0}, mse(0), best_feature(-1), node_type(0)
{
    if (this->X_features_fraction > 1.0 || this->X_features_fraction < std::numeric_limits<double>::epsilon()) {
        throw std::invalid_argument("X_features_fraction must be in the interval (0.0, 1.0] ");
    }

    if (this->min_samples_split < window) {
        throw std::invalid_argument("min_samples_split must be greater than or equal to " + std::to_string(window));
    }
}

std::vector<double> Random_forest_tree::get_mean(const std::vector<std::vector<double>> &arr) {
    std::vector<double> ans(arr.front().size(), 0);

    for (const auto &i : arr) {
        for (size_t j = 0; j < i.size(); ++j) {
            ans[j] += i[j] / static_cast<double>(arr.size());
        }
    }

    return ans;
}

std::vector<double> Random_forest_tree::get_ma(const std::vector<double> &arr, std::vector<int> indices) {
    std::vector<double> ans;
    indices.erase(std::unique(indices.begin(), indices.end(), [&arr](int a, int b){return arr[a] == arr[b];}), indices.end());
    ans.reserve(indices.size() - window + 1);

    for (int i = window - 1; i < indices.size(); ++i) {
        double temp = 0;
        for (int j = 0; j < window; ++j) {
            temp += arr[indices[i - j]] / window;
        }

        ans.push_back(temp);
    }

    return ans;
}

long double
Random_forest_tree::get_mse(const std::vector<std::vector<double>> &arr, const std::vector<double> &value, double n) {
    long double ans = 0;

    for (const auto &i : arr) {
        for (size_t j = 0; j < i.size(); ++j) {
            ans += (i[j] / n) * i[j] - 2 * (i[j] / n) * value[j] + (value[j] / n) * value[j];
        }
    }

    return ans;
}

std::pair<int, double> Random_forest_tree::get_best_split(const Table &x, const std::vector<std::vector<double>> &y) const {
    long double mse_base = this->mse;

    std::vector<long double> sum(y.front().size(), 0),
                             sum2(y.front().size(), 0);

    for (size_t i = 0; i < y.size(); ++i) {
        for (size_t j = 0; j < y[i].size(); ++j) {
            sum[j] += y[i][j] / static_cast<long double>(y.size() * y.front().size());
            sum2[j] += y[i][j] / static_cast<long double>(y.size() * y.front().size()) * y[i][j];
        }
    }

    std::pair<int, double> ans;

    for (const auto &feature : get_features(x.get_columns_count())) {
        const auto& arr = x.get_column(feature);
        std::vector<int> indices(arr.size());
        size_t index = 0;
        std::generate(indices.begin(), indices.end(), [&index](){return index++;});
        std::sort(indices.begin(), indices.end(), [&arr](int a, int b){return arr[a] < arr[b];});

        auto n = static_cast<long double>(y.size() * y.front().size());
        std::vector<long double> leftSum(sum.size(), 0),
                rightSum(sum),
                leftSum2(sum.size(), 0),
                rightSum2(sum2);
        size_t NLeft = 0, NRight = y.size();

        for (const auto &value : get_ma(arr, indices)) {
            while (NLeft < indices.size() - 1 && arr[indices[NLeft]] < value) {
                for (size_t i = 0; i < leftSum.size(); ++i) {
                    const double& temp = y[indices[NLeft]][i];
                    leftSum[i] += temp / n;
                    leftSum2[i] += temp / n * temp;
                    rightSum[i] -= temp / n;
                    rightSum2[i] -= temp / n * temp;
                }

                NLeft++;
                NRight--;
            }

            long double mse_split = 0;
            for (size_t i = 0; i < sum.size(); ++i) {
                mse_split += leftSum2[i] - (n / static_cast<long double>(NLeft)) * leftSum[i] * leftSum[i];
                mse_split += rightSum2[i] - (n / static_cast<long double>(NRight)) * rightSum[i] * rightSum[i];
            }

            if (mse_split < mse_base) {
                ans.first = feature;
                ans.second = value;
                mse_base = mse_split;
            }
        }
    }

    return ans;
}

std::unordered_set<size_t> Random_forest_tree::get_features(size_t n_features) const {
    std::unordered_set<size_t> indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> distribution(0, n_features - 1);

    auto n_ft = static_cast<size_t>(static_cast<double>(n_features) * this->X_features_fraction);

    if (!n_ft) {
        n_ft = 1;
    }

    while (indices.size() < n_ft) {
        indices.insert(distribution(gen));
    }

    return indices;
}

std::tuple<Table, std::vector<std::vector<double>>, Table, std::vector<std::vector<double>>>
Random_forest_tree::split(const Table &x, const std::vector<std::vector<double>> &y) const
{
    std::tuple<Table, std::vector<std::vector<double>>, Table, std::vector<std::vector<double>>> ans;
    Table &left_x = std::get<0>(ans);
    std::vector<std::vector<double>> &left_y = std::get<1>(ans);
    Table &right_x = std::get<2>(ans);
    std::vector<std::vector<double>> &right_y = std::get<3>(ans);

    left_x.set_column_count(x.get_columns_count());
    right_x.set_column_count(x.get_columns_count());

    for (size_t i = 0; i < x.get_rows_count(); ++i) {
        if (x.at(i, this->best_feature) > this->best_value) {
            right_x.push_back_row(x.get_row(i));
            right_y.push_back(y[i]);
        }
        else {
            left_x.push_back_row(x.get_row(i));
            left_y.push_back(y[i]);
        }
    }

    return ans;
}

void Random_forest_tree::print_info(size_t width) const {
    size_t coef = this->depth * static_cast<size_t>(std::sqrt(width * width * width));
    if (this->node_type == 0) {
        std::cout << "Root\n";
    }
    else if (this->best_feature != -1) {
        std::cout << std::string(coef, ' ') << (this->node_type == 1 ? "Left_node\n" : "Right_node\n");
        std::cout << std::string(coef, ' ') << "  | Best value to split " << this->best_value << std::endl;
        std::cout << std::string(coef, ' ') << "  | Best feature to split " << this->best_feature << std::endl;
    }
    else {
        std::cout << std::string(coef, ' ') << (this->node_type == 1 ? "Left_node" : "Right_node") << " (leaf)\n";
    }

    std::cout << std::string(coef, ' ') << "  | MSE of the node: " << this->mse << std::endl;
    std::cout << std::string(coef, ' ') << "  | Count of observations in node: " << this->samples_size << std::endl;
    std::cout << std::string(coef, ' ') << "  | Prediction of node: ";

    for (size_t i = 0; i + 1 < this->ymean.size(); i++) {
        std::cout << this->ymean[i] << ", ";
    }
    std::cout << this->ymean.back() << std::endl;
}

void Random_forest_tree::print_tree() const {
    this->print_info();

    if (this->left) {
        left->print_tree();
    }

    if (this->right) {
        right->print_tree();
    }
}

std::vector<double> Random_forest_tree::predict(const std::vector<double> &values) const {
    const Random_forest_tree *cur_node = this;
    while (true) {
        const int &best_feature = cur_node->best_feature;
        const double &best_value = cur_node->best_value;

        if (best_feature == -1) {
            return cur_node->ymean;
        }

        if (values.at(best_feature) > best_value) {
            cur_node = cur_node->right.get();
        }
        else {
            cur_node = cur_node->left.get();
        }
    }
}

std::vector<std::vector<double>> Random_forest_tree::predict(const Table &values) const {
    std::vector<std::vector<double>> ans;
    ans.reserve(values.get_rows_count());

    for (size_t i = 0; i < values.get_rows_count(); ++i) {
        ans.push_back(this->predict(values.get_row(i)));
    }

    return ans;
}

void Random_forest_tree::fit(const Table &x,
                             const std::vector<std::vector<double>> &y)
{
    this->ymean = get_mean(y);
    this->mse = get_mse(y, this->ymean, static_cast<double>(y.size() * this->ymean.size()));
    this->samples_size = y.size();

    if (this->depth < this->max_depth && y.size() >= this->min_samples_split) {
        auto best_split_values = this->get_best_split(x, y);

        if (best_split_values.first != -1) {
            this->best_feature = best_split_values.first;
            this->best_value = best_split_values.second;

            auto split_data = this->split(x, y);
            Table &left_x = std::get<0>(split_data);
            std::vector<std::vector<double>> &left_y = std::get<1>(split_data);
            Table &right_x = std::get<2>(split_data);
            std::vector<std::vector<double>> &right_y = std::get<3>(split_data);;

            if (!left_y.empty()){
                this->left = std::unique_ptr<Random_forest_tree>(new Random_forest_tree(this->X_features_fraction,
                                                                  this->min_samples_split,
                                                                  this->max_depth));
                this->left->depth = this->depth + 1;
                this->left->node_type = 1;
                this->left->fit(left_x, left_y);
            }

            if (!right_y.empty()) {
                this->right =  std::unique_ptr<Random_forest_tree>(new Random_forest_tree(this->X_features_fraction,
                                                                   this->min_samples_split,
                                                                   this->max_depth));
                this->right->depth = this->depth + 1;
                this->right->node_type = 2;
                this->right->fit(right_x, right_y);
            }
        }
    }
}
