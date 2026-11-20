#pragma once

#include "simulation.h"

#include <cstdint>
#include <string>
#include <vector>

// Summary statistics of a set of samples. Percentiles use the nearest-rank
// method: the p-th percentile is the value at rank ceil(p/100 * n).
struct Stats {
    double mean = 0.0;
    double min = 0.0;
    double p50 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};
Stats computeStats(std::vector<double> values);

// One row of the per-stage table. Every value is in nanoseconds per step.
struct StageRow {
    std::string name;
    Stats stats;
    double percentOfFrame = 0.0;  // Stage's share of total frame time.
    size_t samples = 0;
};

// Splits measured batches into per-stage, per-step timings.
std::vector<StageRow> buildStageRows(const std::vector<BatchSample>& batches);

// Finds a row by name, or throws if it does not exist.
const StageRow& findRow(const std::vector<StageRow>& rows, const std::string& name);

// Settings describing a run, written above every table and CSV.
struct RunInfo {
    std::string device;
    std::string driver;
    uint32_t particles = 0;
    uint32_t batch = 1;
    std::string kernel;
    uint32_t warmupSteps = 0;
    uint32_t measuredSteps = 0;
    bool validation = false;
};

void printReport(const RunInfo& info, const std::vector<StageRow>& rows);
void writeReportCsv(const std::string& path, const RunInfo& info,
                    const std::vector<StageRow>& rows);

// Comment lines starting with '#' that record the GPU, driver and settings.
std::string csvMetadata(const RunInfo& info);
