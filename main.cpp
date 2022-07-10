#include <iostream>
#include "Regression_tree.h"
#include "Random_forest_tree.h"
#include "Random_forest_regressor.h"
#include "Tools.h"

int main() {
    constexpr int n_in = 7, n_out = 1;

    Table t;
//    t.load_from_file("../dataset.csv", {"Day"});
//    t.load_from_file("../daily-total-female-births.csv", {"Date"});
    t.load_from_file("../daily-min-temperatures.csv", {"Date"});
    t = series_to_supervised(t, n_in, n_out);

//    auto regressor = Regression_tree(3, 3);
//    auto regressor = Random_forest_tree(0, 3, 5);
    auto regressor = Random_forest_regressor(1000, 0.75, 1.0, 3, 5);

    double mae = walk_forward_validation(regressor, t, 12, n_out);

    std::cout << "MAE: " << mae << std::endl;

//    auto start = std::chrono::high_resolution_clock::now();
//    double mae = walk_forward_validation(regressor, t, 12, n_out);
//    auto stop = std::chrono::high_resolution_clock::now();
//
//    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms\n";
//    std::cout << "MAE: " << mae << std::endl;

    return 0;
}