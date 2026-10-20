#include "csv.hpp"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace telemetry {
namespace {

std::string_view trim(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r");
    if (first == std::string_view::npos) return {};
    const auto last = text.find_last_not_of(" \t\r");
    return text.substr(first, last - first + 1);
}

[[noreturn]] void invalid_csv(std::size_t line, const char* reason) {
    throw std::invalid_argument("CSV line " + std::to_string(line) + ": " + reason);
}

float parse_value(std::string_view text, std::size_t line) {
    text = trim(text);
    if (text.empty()) invalid_csv(line, "Missing numeric value");

    // from_chars accepts a leading minus, but requires us to handle a leading plus.
    if (text.front() == '+') {
        text.remove_prefix(1);
        if (text.empty() || text.front() == '-') {
            invalid_csv(line, "Expected a finite float value");
        }
    }

    float value{};
    const char* end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value, std::chars_format::general);
    if (result.ec != std::errc{} || result.ptr != end || !std::isfinite(value)) {
        invalid_csv(line, "Expected a finite float value");
    }
    return value;
}

} // namespace

TelemetryData read_csv(std::istream& input) {
    std::string line;
    if (!std::getline(input, line)) {
        if (input.bad() || !input.eof()) {
            throw std::runtime_error("Could not read CSV header");
        }
        invalid_csv(1, "Expected header measurement,reference");
    }
    if (trim(line) != "measurement,reference") {
        invalid_csv(1, "Expected header measurement,reference");
    }

    TelemetryData data;
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        const auto row = trim(line);
        if (row.empty()) continue;

        const auto comma = row.find(',');
        if (comma == std::string_view::npos || row.find(',', comma + 1) != std::string_view::npos) {
            invalid_csv(line_number, "Expected exactly two columns");
        }
        const float measurement = parse_value(row.substr(0, comma), line_number);
        const float reference = parse_value(row.substr(comma + 1), line_number);
        data.measurements.push_back(measurement);
        data.references.push_back(reference);
    }
    // A failed read must not silently return a partial dataset as a successful load.
    if (input.bad() || !input.eof()) {
        throw std::runtime_error("Could not read CSV input after line " + std::to_string(line_number));
    }
    return data;
}

} // namespace telemetry
