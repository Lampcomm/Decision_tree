#ifndef TREE_TABLE_H
#define TREE_TABLE_H

#include <vector>
#include <string>
#include <unordered_set>
#include <deque>

class Table {
public:
    Table() = default;

    /// Function to load a table from a file
    void load_from_file(
        const std::string &file_name,                                ///< The path to the file
        const std::unordered_set<std::string> &ignored_columns = {}, ///< Ignored column names
        char delim = ','                                             ///< Separator between columns data
    );

    /// Function of loading data from the matrix
    void load_from_array(
        double *arr, ///< Matrix pointer
        size_t n,    ///< Number of rows in matrix
        size_t m     ///< Number of columns in matrix
    );

    /// Function that returns the number of rows in a table
    size_t get_rows_count() const;

    /// Function that returns the number of columns in a table
    size_t get_columns_count() const;

    /// Function that changes the number of rows in a table
    void set_rows_count(
        size_t rows ///< New number of rows
    );

    /// Function that changes the number of columns in a table
    void set_column_count(
        size_t columns ///< New number of columns
    );

    /// Function returning a table column by its index
    std::vector<double> get_column(
        size_t column ///< Column index
    ) const;

    /// Function that returns a table row by its index
    std::vector<double> get_row(
        size_t row ///< Row index
    ) const;

    /// Function that inserts a row at the end of a table
    void push_back_row(
        const std::vector<double> &row ///< New row
    );

    /// Function that inserts a column at the end of a table
    void push_back_column(
        const std::vector<double> &column ///< New column
    );

    /// Table data access function by row and column indexes
    double& at(
        size_t row,   ///< Row index
        size_t column ///< Column index
    );

    /// Constant table data access function by row and column indexes
    const double& at(
        size_t row,   ///< Row index
        size_t column ///< Column index
    ) const;

    friend std::ostream& operator<<(std::ostream &out, const Table &a);
    friend Table series_to_supervised(const Table &data, int n_in, int n_out);

private:
    static std::vector<std::string> split(const std::string &str, char delim);

private:
    size_t rows = 0;         ///< Number of rows in the table
    size_t columns = 0;      ///< Number of columns in the table
    std::deque<double> data; ///< Table data
};


#endif //TREE_TABLE_H
