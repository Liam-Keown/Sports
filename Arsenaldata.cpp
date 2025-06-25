#include "Arsenaldata.h"

// This handles quoted fields (commas inside quotes) and escaped quotes ("")
std::vector<std::string> parseCsvLine(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    std::string current_field;
    bool in_quote = false;

    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];

        if (c == '"') {
            if (in_quote && (i + 1 < line.length()) && (line[i+1] == '"')) {
                current_field += '"';
                i++; // Skip the next quote
            } else {
                in_quote = !in_quote;
            }
        } else if (c == delimiter && !in_quote) {
            fields.push_back(current_field);
            current_field.clear();
        } else {
            current_field += c;
        }
    }
    // Add the last field after the loop finishes
    fields.push_back(current_field);

    return fields;
}

DataFrame::DataFrame() {

}

bool DataFrame::read_csv(const std::string& filename, char delimiter) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    // read headers    
    std::string line;
    if (std::getline(file, line)) {
        headers = parseCsvLine(line, delimiter);
        build_header_map();
    } else {
        std::cerr << "Error: CSV file is empty or header could not be read." << std::endl;
        file.close();
        return false;
    }

    // read data lines
    data.clear(); 
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> row_data = parseCsvLine(line, delimiter);
        if (row_data.size() == headers.size()) {
            data.push_back(row_data);
        } else {
            std::cerr << "Warning: Skipping malformed row (column count mismatch): " << line << std::endl;
        }
    }

    file.close();
    return true;
}

void DataFrame::build_header_map() {
    header_to_index.clear();
    for (int i = 0; i < headers.size(); ++i) {
        header_to_index[headers[i]] = i;
    }
}

std::vector<std::string> DataFrame::get_column(const std::string& column_name) const {
    std::vector<std::string> column_data;
    auto it = header_to_index.find(column_name);

    if (it == header_to_index.end()) {
        std::cerr << "Error: Column '" << column_name << "' not found." << std::endl;
        return column_data; 
    }

    int col_index = it->second;
    for (const auto& row : data) {
        if (col_index < row.size()) { 
            column_data.push_back(row[col_index]);
        }
    }
    return column_data;
}

const std::vector<std::string>& DataFrame::get_row(size_t index) const {
    if (index >= data.size()) {
        std::cerr << "Error: Row index " << index << " out of bounds." << std::endl;
        // Return a static empty vector to avoid crashing.
        // In real code, you might throw an exception.
        static const std::vector<std::string> empty_row;
        return empty_row;
    }
    return data[index];
}

std::string DataFrame::get_value(size_t row_index, const std::string& column_name) const {
    auto it = header_to_index.find(column_name);
    if (it == header_to_index.end()) {
        std::cerr << "Error: Column '" << column_name << "' not found." << std::endl;
        return ""; 
    }
    int col_index = it->second;

    if (row_index >= data.size()) {
        std::cerr << "Error: Row index " << row_index << " out of bounds." << std::endl;
        return "";
    }

    if (col_index >= data[row_index].size()) {
        std::cerr << "Error: Column index " << col_index << " out of bounds for row " << row_index << std::endl;
        return "";
    }

    return data[row_index][col_index];
}


void DataFrame::head(size_t n) const {
    std::cout << "--- DataFrame Head (" << std::min(n, num_rows()) << "/" << num_rows() << " rows) ---" << std::endl;

    // Print headers
    for (const auto& header : headers) {
        std::cout << header << "\t";
    }
    std::cout << std::endl;

    // Print data rows
    for (size_t i = 0; i < std::min(n, num_rows()); ++i) {
        for (const auto& field : data[i]) {
            std::cout << field << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------------------------" << std::endl;
}

void DataFrame::print() const {
    std::cout << "--- Full DataFrame (" << num_rows() << " rows) ---" << std::endl;

    // Print headers
    for (const auto& header : headers) {
        std::cout << header << "\t";
    }
    std::cout << std::endl;

    // Print all data rows
    for (const auto& row : data) {
        for (const auto& field : row) {
            std::cout << field << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------------------------" << std::endl;
}