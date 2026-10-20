#pragma once

#include "generator.hpp"

#include <istream>

namespace telemetry {

// Read an unquoted, two-column numeric CSV headed measurement,reference.
// Invalid rows throw std::invalid_argument with a line number; I/O errors throw
// std::runtime_error. The caller owns the stream. Anomaly flags remain empty.
TelemetryData read_csv(std::istream& input);

} // namespace telemetry
