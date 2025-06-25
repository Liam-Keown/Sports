#ifndef ARSENALDATA_H
#define ARSENALDATA_H

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>


std::vector<std::string> parseCsvLine(const std::string& line, char delimiter);

class DataFrame {
public:

    DataFrame();

    bool read_csv(const std::string& filename, char delimiter = ',');

    size_t num_rows() const { return data.size(); }

    size_t num_cols() const { return headers.size(); }

    const std::vector<std::string>& get_headers() const { return headers; }

    std::vector<std::string> get_column(const std::string& column_name) const;

    const std::vector<std::string>& get_row(size_t index) const;

    std::string get_value(size_t row_index, const std::string& column_name) const;

    void head(size_t n = 5) const;

    void print() const;

private:
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> data; 
    std::map<std::string, int> header_to_index; 
    std::map<std::string, std::vector<size_t>> date_to_row_indices;

    void build_header_map();
    void build_date_index();
};

#endif 

