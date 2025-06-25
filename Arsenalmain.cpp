#include "Arsenaldata.h" 
#include "Arsenaldata.cpp"
#include <iostream>
#include <iomanip>     
#include <limits>     
int main() {
    DataFrame df;

    std::cout << "Attempting to read 'Arsenal19-20.csv'..." << std::endl;
    if (df.read_csv("Arsenal19-20.csv")) {
        std::cout << "File read successfully!" << std::endl;

        // --- DataFrame operations ---
        std::cout << "\nDataFrame Headers:\n";
        for (const auto& header : df.get_headers()) {
            std::cout << header << " ";
        }
        std::cout << std::endl;

        std::cout << "\nDataFrame dimensions: " << df.num_rows() << " rows, " << df.num_cols() << " columns." << std::endl;

        std::cout << "\n--- Displaying first 5 rows (df.head()) ---" << std::endl;
        df.head();

        std::cout << "\n--- Getting 'Player' column data ---" << std::endl;
        std::vector<std::string> players = df.get_column("LastName");
        if (!players.empty()) {
            for (size_t i = 0; i < players.size()/200; ++i) {
                std::cout << "Row " << i << ": " << players[i] << std::endl;
            }
        }

        std::cout << "\n--- Getting 'Goals' column data ---" << std::endl;
        std::vector<std::string> goals_str = df.get_column("G");
        if (!goals_str.empty()) {
            // Example of converting string data to int
            int total_goals = 0;
            std::cout << "Goals Scored per record:\n";
            for (const auto& goal_str : goals_str) {
                try {
                    int goals = std::stoi(goal_str);
                    total_goals += goals;
                    std::cout << goals << " ";
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Warning: Could not convert '" << goal_str << "' to int." << std::endl;
                } catch (const std::out_of_range& e) {
                    std::cerr << "Warning: Value '" << goal_str << "' out of range for int." << std::endl;
                }
            }
            std::cout << "\nTotal goals over these games: " << total_goals << std::endl;
        }

        std::cout << "\n--- Accessing specific value (Row 2, Column 'Line') ---" << std::endl;
        // Index 2 refers to the 3rd data row (M002, second record for Aubameyang)
        std::cout << "Value: " << df.get_value(2, "Line") << std::endl;

        std::cout << "\n--- Printing a specific row (Row 0) ---" << std::endl;
        const auto& first_row = df.get_row(0);
        for (const auto& field : first_row) {
            std::cout << field << " | ";
        }
        std::cout << std::endl;

        std::cout << "\n--- Displaying all data (df.print()) ---" << std::endl;
        //df.print();

    } else {
        std::cout << "Failed to read the CSV file." << std::endl;
    }

    // Exit console 
    std::cout << "\nPress Enter to exit.";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
    std::cin.get();
    return 0;
}