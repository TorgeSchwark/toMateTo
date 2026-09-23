#pragma once
// A single-line, in-place-updating progress bar (uses '\r', not '\n' - only
// the final call should end the line). Deliberately plain stdout + \r, not
// stderr: everything else in these tools already prints via std::printf on
// stdout, and mixing streams risks the progress bar and the surrounding log
// lines interleaving out of order when output is piped/redirected to a file
// (as e.g. a background training run typically is).

#include <cstdio>
#include <string>

inline void print_progress_bar(long current, long total, double rate_ms_per_item, double eta_s,
                                const std::string& suffix = "") {
    constexpr int kWidth = 30;
    const double frac = total > 0 ? double(current) / double(total) : 1.0;
    const int filled = int(frac * kWidth);

    std::string bar(std::size_t(kWidth), ' ');
    for (int i = 0; i < filled && i < kWidth; ++i) bar[std::size_t(i)] = '=';
    if (filled < kWidth) bar[std::size_t(filled)] = '>';

    std::printf("\r[%s] %3d%%  (%ld/%ld)  %.1fms/item  ETA %.0fs%s%s", bar.c_str(), int(frac * 100.0), current,
                total, rate_ms_per_item, eta_s, suffix.empty() ? "" : "  ", suffix.c_str());
    std::fflush(stdout);
    if (current >= total) std::printf("\n");
}
