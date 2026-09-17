#pragma once

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <x86intrin.h>

static inline uint64_t cpu_cycles()
{
    return __rdtsc();
}

class Profiler
{
public:

    using Clock = std::chrono::high_resolution_clock;

    struct Stats
    {
        uint64_t cycles = 0;
        uint64_t calls = 0;
    };

    inline static std::map<std::string, Stats> stats;

    // ---------------------------------------------------------
    // Chess-engine counters
    // ---------------------------------------------------------

    inline static uint64_t nodes = 0;
    inline static uint64_t q_nodes = 0;

    inline static uint64_t tt_lookups = 0;
    inline static uint64_t tt_hits = 0;
    inline static uint64_t tt_cutoffs = 0;

    inline static uint64_t tt_exact = 0;
    inline static uint64_t tt_lowerbound = 0;
    inline static uint64_t tt_upperbound = 0;

    // ---------------------------------------------------------
    // CPU frequency
    // ---------------------------------------------------------

    inline static double cpu_frequency_ghz = 0.0;

    // ---------------------------------------------------------
    // Measure TSC frequency
    // ---------------------------------------------------------

    static void calibrate_cpu_frequency()
    {
        using namespace std::chrono;

        uint64_t start_cycles = cpu_cycles();
        auto start_time = steady_clock::now();

        // 100 ms measurement
        std::this_thread::sleep_for(milliseconds(100));

        uint64_t end_cycles = cpu_cycles();
        auto end_time = steady_clock::now();

        double seconds =
            duration<double>(end_time - start_time).count();

        uint64_t cycles = end_cycles - start_cycles;

        cpu_frequency_ghz =
            static_cast<double>(cycles) /
            seconds /
            1'000'000'000.0;
    }

    // ---------------------------------------------------------
    // Convert cycles -> microseconds
    // ---------------------------------------------------------

    static double cycles_to_us(uint64_t cycles)
    {
        if (cpu_frequency_ghz <= 0.0)
            return 0.0;

        return static_cast<double>(cycles) /
               (cpu_frequency_ghz * 1000.0);
    }

    // ---------------------------------------------------------
    // Scope
    // ---------------------------------------------------------

    class Scope
    {
    public:

        Scope(const char* name)
            : name(name),
              start_time(cpu_cycles()),
              running(true)
        {
            Profiler::stats[name].calls++;
        }

        ~Scope()
        {
            stop();
        }

        void stop()
        {
            if (!running)
                return;

            uint64_t end = cpu_cycles();

            Profiler::stats[name].cycles +=
                end - start_time;

            running = false;
        }

        void start()
        {
            if (running)
                return;

            start_time = cpu_cycles();
            running = true;
        }

    private:

        const char* name;
        uint64_t start_time;
        bool running;
    };

    // ---------------------------------------------------------
    // Reset
    // ---------------------------------------------------------

    static void reset()
    {
        stats.clear();

        nodes = 0;
        q_nodes = 0;

        tt_lookups = 0;
        tt_hits = 0;
        tt_cutoffs = 0;

        tt_exact = 0;
        tt_lowerbound = 0;
        tt_upperbound = 0;
    }

    // ---------------------------------------------------------
    // Print
    // ---------------------------------------------------------

    static void print()
    {
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "PROFILER\n";
        std::cout << "========================================\n";

        std::cout
            << std::left
            << std::setw(25) << "Function"
            << std::right
            << std::setw(15) << "Time [ms]"
            << std::setw(15) << "Calls"
            << std::setw(15) << "us/call"
            << "\n";

        std::cout << "----------------------------------------\n";

        for (const auto& [name, stat] : stats)
        {
            double us =
                cycles_to_us(stat.cycles);

            double ms =
                us / 1000.0;

            double us_per_call =
                stat.calls > 0
                    ? us /
                      static_cast<double>(stat.calls)
                    : 0.0;

            std::cout
                << std::left
                << std::setw(25) << name
                << std::right
                << std::setw(15) << std::fixed
                << std::setprecision(3) << ms
                << std::setw(15) << stat.calls
                << std::setw(15) << std::setprecision(3)
                << us_per_call
                << "\n";
        }

        std::cout << "\n";

        std::cout << "CPU frequency: "
                  << std::fixed
                  << std::setprecision(3)
                  << cpu_frequency_ghz
                  << " GHz\n";

        std::cout << "\n";

        std::cout << "========================================\n";
        std::cout << "SEARCH STATISTICS\n";
        std::cout << "========================================\n";

        std::cout
            << "Nodes:          "
            << nodes << "\n";

        std::cout
            << "Q-Nodes:        "
            << q_nodes << "\n";

        std::cout
            << "TT lookups:     "
            << tt_lookups << "\n";

        std::cout
            << "TT hits:        "
            << tt_hits << "\n";

        std::cout
            << "TT cutoffs:     "
            << tt_cutoffs << "\n";

        std::cout
            << "TT EXACT:       "
            << tt_exact << "\n";

        std::cout
            << "TT LOWERBOUND:  "
            << tt_lowerbound << "\n";

        std::cout
            << "TT UPPERBOUND:  "
            << tt_upperbound << "\n";

        if (tt_lookups > 0)
        {
            double hit_rate =
                100.0 *
                static_cast<double>(tt_hits) /
                static_cast<double>(tt_lookups);

            std::cout
                << "TT hit rate:    "
                << std::fixed
                << std::setprecision(2)
                << hit_rate
                << "%\n";
        }
    }
};