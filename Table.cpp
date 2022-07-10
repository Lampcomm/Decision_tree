#include "Table.h"

#include <fstream>
#include <sstream>

void Table::load_from_file(const std::string &file_name, const std::unordered_set<std::string> &ignored_columns, char delim) {
    std::ifstream inp(file_name);

    if (!inp.is_open()) {
        throw std::invalid_argument("Failed to open file");
    }

    std::string line;
    std::getline(inp, line);

    std::vector<std::string> column_names = split(line, delim);
    this->columns = column_names.size();

    if (ignored_columns.size() >= this->columns) {
        throw std::invalid_argument("The number of columns to be ignored is greater or equal than the total number of columns");
    }

    this->columns -= ignored_columns.size();

    while (std::getline(inp, line)) {
        ++this->rows;
        std::vector<std::string> temp = split(line, delim);
        for (size_t i = 0; i < temp.size(); ++i) {
            if (ignored_columns.find(column_names[i]) != ignored_columns.end()) {
                continue;
            }
            this->data.push_back(std::stod(temp[i]));
        }
    }
}

std::vector<std::string> Table::split(const std::string &str, char delim) {
    std::vector<std::string> ans;
    std::istringstream inp(str);

    for (std::string word; std::getline(inp, word, delim);) {
        ans.push_back(word);
    }

    return ans;
}

std::ostream& operator<<(std::ostream &out, const Table &a) {
    if (!a.rows || !a.columns) {
        return out;
    }

    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.columns; ++j) {
            out << a.data[i * a.columns + j] << "\t";
        }
        out << std::endl;
    }

    return out;
}

size_t Table::get_rows_count() const {
    return this->rows;
}

size_t Table::get_columns_count() const {
    return this->columns;
}

double &Table::at(size_t row, size_t column) {
    if (row >= this->rows || column >= this->columns) {
        throw std::out_of_range("Out of range");
    }

    return this->data.at(row * this->columns + column);
}

const double &Table::at(size_t row, size_t column) const {
    if (row >= this->rows || column >= this->columns) {
        throw std::out_of_range("Out of range");
    }

    return this->data[row * this->columns + column];
}

void Table::set_rows_count(size_t rows) {
    if (this->rows > rows) {
        this->data.erase(std::prev(this->data.end(), this->columns * (this->rows - rows)), this->data.end());
    }
    else if (this->rows < rows) {
        this->data.resize(rows * this->columns);
    }

    this->rows = rows;
}

void Table::set_column_count(size_t columns) {
    if (this->columns > columns) {
        auto it = std::next(this->data.begin(), columns);
        for (size_t i = 0; i < this->rows; it += columns, ++i) {
            it = this->data.erase(it, std::next(it, this->columns - columns));
        }
    }
    else if (this->columns < columns) {
        auto it = std::next(this->data.begin(), this->columns);
        for (size_t i = 0; i < this->rows; ++i, it += columns) {
            it = this->data.insert(it, columns - this->columns, 0.0);
        }
    }

    this->columns = columns;
}

std::vector<double> Table::get_column(size_t column) const {
    if (column >= this->columns) {
        throw std::out_of_range("Out of range");
    }

    std::vector<double> ans;
    ans.reserve(this->rows);

    for (size_t i = 0; i < this->rows; ++i) {
        ans.push_back(this->data[i * this->columns + column]);
    }

    return ans;
}

std::vector<double> Table::get_row(size_t row) const {
    if (row >= this->rows) {
        throw std::out_of_range("Out of range");
    }

    std::vector<double> ans;
    ans.reserve(this->columns);

    for (size_t i = 0; i < this->columns; ++i) {
        ans.push_back(this->data[row * this->columns + i]);
    }

    return ans;
}

void Table::push_back_row(const std::vector<double> &row) {
    if (row.size() != this->columns) {
        throw std::invalid_argument("Wrong number of columns");
    }

    ++this->rows;
    for (const auto &i : row) {
        this->data.push_back(i);
    }
}

void Table::push_back_column(const std::vector<double> &column) {
    if (column.size() != this->rows) {
        throw std::invalid_argument("Wrong number of rows");
    }

    auto it = std::next(this->data.begin(), this->columns);

    for (const auto &i : column) {
        it = this->data.insert(it, i);
        it += this->columns + 1;
    }

    ++this->columns;
}

void Table::load_from_array(double *arr, size_t n, size_t m) {
    this->rows = n;
    this->columns = m;

    for (size_t i = 0; i < n * m; ++i) {
        this->data.push_back(arr[i]);
    }
}
