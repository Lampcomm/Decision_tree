#include "Tools.h"

#include <iostream>
#include "Regression_tree.h"

#define TEST 1
#if TEST
#include <chrono>
#endif

double mean_absolute_error(const std::vector<std::vector<double>> &observation, const std::vector<std::vector<double>> &predictions) {
    double ans = 0;

    for (size_t i = 0; i < observation.size(); ++i) {
        for (size_t j = 0; j < observation[i].size(); ++j) {
            ans += std::abs(predictions[i][j] - observation[i][j]) /
                   static_cast<double>(observation.size() * observation.front().size());
        }
    }

    return ans;
}

#if TEST
std::chrono::microseconds res_time, one_fit_time;
#endif

std::vector<double> abstract_regressor_forecast(Abstract_regressor &regressor,
                                                const Table &train,
                                                const std::vector<double> &test,
                                                int n_observation)
{
    Table x;
    std::vector<std::vector<double>> y(train.get_rows_count());

    x.set_column_count(train.get_columns_count() - n_observation);
    for (size_t i = 0; i < train.get_rows_count(); ++i) {
        for (size_t j = train.get_columns_count() - n_observation; j < train.get_columns_count(); ++j) {
            y[i].push_back(train.at(i, j));
        }
        auto r = train.get_row(i);
        r.erase(std::prev(r.end(), n_observation), r.end());
        x.push_back_row(r);
    }

#if TEST
    auto start = std::chrono::high_resolution_clock::now();
    regressor.fit(x, y);
    auto stop = std::chrono::high_resolution_clock::now();
    one_fit_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    res_time += one_fit_time;
#else
    regressor.fit(x, y);
#endif

    return regressor.predict(test);
}

double walk_forward_validation(Abstract_regressor &regressor, const Table &data, int n_test, int n_observation) {
    auto data_split = train_test_split(data, n_test);
    std::vector<std::vector<double>> predictions, observation;

    for (int i = 0; i < n_test; ++i) {

        std::vector<double> testY = data_split.second.get_row(i);
        observation.emplace_back();
        observation.back().reserve(n_observation);
        for (size_t j = testY.size() - n_observation; j < testY.size(); ++j) {
            observation.back().push_back(testY[j]);
        }
        testY.erase(std::prev(testY.end(), n_observation), testY.end());

        predictions.emplace_back(abstract_regressor_forecast(regressor, data_split.first, testY, n_observation));
        data_split.first.push_back_row(data_split.second.get_row(i));

        std::cout << ">expected=";

        for (size_t j = 0; j + 1 < observation.back().size(); ++j) {
            std::cout << observation.back()[j] << ", ";
        }
        std::cout << observation.back().back() << ", predicted=";

        for (size_t j = 0; j + 1 < predictions.back().size(); ++j) {
            std::cout << predictions.back()[j] << ", ";
        }

        std::cout << predictions.back().back() << std::endl;
#if TEST
        std::cout << "One fit time: " << static_cast<double>(one_fit_time.count()) * 1e-6 << " s.\n\n";
#endif
    }

#if TEST
    std::cout << "Total fit time: " << static_cast<double>(res_time.count()) * 1e-6 << " s.\n";
    std::cout << "Mean fit time: " << static_cast<double>(res_time.count()) / static_cast<double>(n_test) * 1e-6 << " s.\n";
#endif

    return mean_absolute_error(observation, predictions);
}

std::pair<Table, Table> train_test_split(const Table &data, int n_tests)
{
    std::pair<Table, Table> ans;

    ans.first.set_column_count(data.get_columns_count());
    ans.second.set_column_count(data.get_columns_count());

    for (size_t i = 0; i < data.get_rows_count() - n_tests; ++i) {
        ans.first.push_back_row(data.get_row(i));
    }

    for (size_t i = data.get_rows_count() - n_tests; i < data.get_rows_count(); ++i) {
        ans.second.push_back_row(data.get_row(i));
    }

    return ans;
}

Table series_to_supervised(const Table &data, int n_in, int n_out) {
    int new_rows = static_cast<int>(data.rows) - n_in - n_out + 1;
    int new_columns = (n_in + n_out) * static_cast<int>(data.columns);

    if (new_rows <= 0 || new_columns <= 0) {
        return {};
    }

    Table ans;
    ans.rows = new_rows;
    ans.columns = new_columns;

    for (size_t i = 0, shift = 0; i < new_rows; ++i, shift += data.columns) {
        auto it = data.data.begin() + shift;
        std::copy(it, it + new_columns, std::back_inserter(ans.data));
    }

    return ans;
}
