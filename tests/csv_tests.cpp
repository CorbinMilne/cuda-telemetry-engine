#include "csv.hpp"
#include "validation.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_rejected(const std::string& text, std::size_t line) {
    std::istringstream input(text);
    try {
        telemetry::read_csv(input);
    } catch (const std::invalid_argument& error) {
        const std::string prefix = "CSV line " + std::to_string(line) + ": ";
        require(std::string(error.what()).starts_with(prefix), "CSV error has the wrong line number");
        return;
    }
    throw std::runtime_error("Invalid CSV was accepted");
}

} // namespace

int main() {
    try {
        std::istringstream example("measurement,reference\n10,10\n10.5,10\n8,10\n14,10\n");
        const auto data = telemetry::read_csv(example);
        require(data.measurements == std::vector<float>{10, 10.5F, 8, 14}, "Wrong measurement values");
        require(data.references == std::vector<float>{10, 10, 10, 10}, "Wrong reference values");
        require(data.anomaly_flags.empty(), "File input must not invent anomaly labels");
        const auto result = telemetry::validate(data.measurements, data.references, 1.0F);
        require(result.sample_count == 4 && result.violation_count == 2 &&
                result.max_absolute_error == 4.0 && result.mean_squared_error == 5.0625,
                "CSV data produced the wrong validation summary");
        std::cout << "PASS: CSV values and validation summary\n";

        std::istringstream formatted("measurement,reference\r\n\r\n +1.25e1 ,\t-2.5 \r\n0, -0");
        const auto parsed = telemetry::read_csv(formatted);
        require(parsed.measurements == std::vector<float>{12.5F, 0} &&
                parsed.references == std::vector<float>{-2.5F, 0},
                "Spaces, signed numbers, scientific notation, CRLF, or final line failed");
        std::istringstream header_only("measurement,reference\n \t\r\n");
        const auto empty = telemetry::read_csv(header_only);
        require(empty.measurements.empty() && empty.references.empty() && empty.anomaly_flags.empty(),
                "Header-only CSV must produce empty data");
        std::istringstream tiny("measurement,reference\n1.401298464324817e-45,0\n");
        require(telemetry::read_csv(tiny).measurements[0] == std::numeric_limits<float>::denorm_min(),
                "Representable subnormal input must be accepted");
        std::cout << "PASS: CSV formatting, empty data, and subnormal input\n";

        for (const std::string text : {"", "\n", "reference,measurement\n", "measurement\n",
                                       "measurement,reference,extra\n"}) {
            require_rejected(text, 1);
        }
        for (const std::string row : {"1", "1,2,3", "1,2,", ",2", "1,", ",", "\"1\",2"}) {
            require_rejected("measurement,reference\n1,1\n" + row + "\n", 3);
        }
        for (const std::string value : {"nan", "inf", "-inf", "word", "1junk", "1 2",
                                        "1e100", "1e-100", "+", "+-1", "++1", "0x1p0"}) {
            require_rejected("measurement,reference\n" + value + ",0\n", 2);
            require_rejected("measurement,reference\n0," + value + "\n", 2);
        }
        require_rejected("measurement,reference\n\n1,1\n\nwrong\n", 5);
        std::cout << "PASS: CSV headers, columns, numeric errors, and line numbers\n";

        std::istringstream broken;
        broken.setstate(std::ios::badbit);
        bool io_error = false;
        try {
            telemetry::read_csv(broken);
        } catch (const std::runtime_error&) {
            io_error = true;
        }
        require(io_error, "An unreadable stream must fail instead of returning empty data");
        std::cout << "PASS: CSV stream read failure\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
