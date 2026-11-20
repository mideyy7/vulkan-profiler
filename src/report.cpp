#include "report.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>

Stats computeStats(std::vector<double> values) {
    Stats stats;
    if (values.empty()) {
        return stats;
    }
    std::sort(values.begin(), values.end());
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    const size_t n = values.size();
    auto percentile = [&](double p) {
        const size_t rank = static_cast<size_t>(std::ceil(p / 100.0 * static_cast<double>(n)));
        return values[std::max<size_t>(rank, 1) - 1];
    };
    stats.mean = sum / static_cast<double>(n);
    stats.min = values.front();
    stats.p50 = percentile(50.0);
    stats.p99 = percentile(99.0);
    stats.max = values.back();
    return stats;
}

namespace {

// Samples for one stage plus its total time, used for the % of frame column.
struct StageSamples {
    explicit StageSamples(std::string stageName) : name(std::move(stageName)) {}
    std::string name;
    std::vector<double> perStep;
    double total = 0.0;
};

}  // namespace

std::vector<StageRow> buildStageRows(const std::vector<BatchSample>& batches) {
    StageSamples force("force"), forceBarrier("force->integrate barrier"),
        integrate("integrate"), stepBarrier("step barrier"),
        readbackBarrier("readback barrier"), copy("readback copy"), gpu("gpu span"),
        submit("cpu submit"), wait("cpu fence wait"), frame("frame"),
        overhead("overhead (frame - gpu)");

    // A stage that happens once per step contributes one sample per step.
    auto addEach = [](StageSamples& s, double ns) {
        s.perStep.push_back(ns);
        s.total += ns;
    };
    // A stage that happens once per batch is spread over the batch's steps.
    auto addPerBatch = [](StageSamples& s, double ns, uint32_t steps) {
        s.perStep.push_back(ns / steps);
        s.total += ns;
    };

    for (const BatchSample& b : batches) {
        const uint32_t K = b.steps;
        auto ns = [&](uint32_t from, uint32_t to) {
            return static_cast<double>(b.ticks[to] - b.ticks[from]) * b.nsPerTick;
        };
        for (uint32_t k = 0; k < K; ++k) {
            addEach(force, ns(4 * k, 4 * k + 1));
            addEach(forceBarrier, ns(4 * k + 1, 4 * k + 2));
            addEach(integrate, ns(4 * k + 2, 4 * k + 3));
            if (k + 1 < K) {
                addEach(stepBarrier, ns(4 * k + 3, 4 * k + 4));
            }
        }
        const double gpuNs = ns(0, 4 * K + 1);
        addPerBatch(readbackBarrier, ns(4 * K - 1, 4 * K), K);
        addPerBatch(copy, ns(4 * K, 4 * K + 1), K);
        addPerBatch(gpu, gpuNs, K);
        addPerBatch(submit, b.submitNs, K);
        addPerBatch(wait, b.waitNs, K);
        addPerBatch(frame, b.frameNs, K);
        addPerBatch(overhead, b.frameNs - gpuNs, K);
    }

    std::vector<StageRow> rows;
    for (const StageSamples* s : {&force, &forceBarrier, &integrate, &stepBarrier,
                                  &readbackBarrier, &copy, &gpu, &submit, &wait, &frame,
                                  &overhead}) {
        if (s->perStep.empty()) {
            continue;  // No step barrier when every batch is a single step.
        }
        StageRow row;
        row.name = s->name;
        row.stats = computeStats(s->perStep);
        row.percentOfFrame = frame.total > 0.0 ? 100.0 * s->total / frame.total : 0.0;
        row.samples = s->perStep.size();
        rows.push_back(row);
    }
    return rows;
}

const StageRow& findRow(const std::vector<StageRow>& rows, const std::string& name) {
    for (const StageRow& row : rows) {
        if (row.name == name) {
            return row;
        }
    }
    throw std::runtime_error("no stage named " + name);
}

std::string csvMetadata(const RunInfo& info) {
    std::string text;
    text += "# gpu: " + info.device + "\n";
    text += "# driver: " + info.driver + "\n";
    text += "# particles: " + std::to_string(info.particles) + "\n";
    text += "# batch: " + std::to_string(info.batch) + "\n";
    text += "# kernel: " + info.kernel + "\n";
    text += "# warmup_steps: " + std::to_string(info.warmupSteps) + "\n";
    text += "# measured_steps: " + std::to_string(info.measuredSteps) + "\n";
    text += std::string("# validation: ") + (info.validation ? "on" : "off") + "\n";
    return text;
}

void printReport(const RunInfo& info, const std::vector<StageRow>& rows) {
    std::printf("GPU: %s (%s)\n", info.device.c_str(), info.driver.c_str());
    std::printf("N = %u, batch = %u, kernel = %s, %u measured steps after %u warm-up steps",
                info.particles, info.batch, info.kernel.c_str(), info.measuredSteps,
                info.warmupSteps);
    std::printf(", validation %s\n\n", info.validation ? "ON (timings inflated)" : "off");
    std::printf("Per-step times in microseconds\n");
    std::printf("%-26s %10s %10s %10s %10s %10s %8s\n", "stage", "mean", "min", "p50", "p99",
                "max", "% frame");
    for (const StageRow& row : rows) {
        std::printf("%-26s %10.2f %10.2f %10.2f %10.2f %10.2f %7.1f%%\n", row.name.c_str(),
                    row.stats.mean / 1e3, row.stats.min / 1e3, row.stats.p50 / 1e3,
                    row.stats.p99 / 1e3, row.stats.max / 1e3, row.percentOfFrame);
    }
}

void writeReportCsv(const std::string& path, const RunInfo& info,
                    const std::vector<StageRow>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    out << csvMetadata(info);
    out << "stage,mean_ns,min_ns,p50_ns,p99_ns,max_ns,percent_of_frame,samples\n";
    for (const StageRow& row : rows) {
        out << row.name << ',' << row.stats.mean << ',' << row.stats.min << ','
            << row.stats.p50 << ',' << row.stats.p99 << ',' << row.stats.max << ','
            << row.percentOfFrame << ',' << row.samples << '\n';
    }
}
