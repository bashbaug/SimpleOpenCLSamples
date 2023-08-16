/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

//#define USE_STRING_KEYS

#include <chrono>
class TimerChrono {
public:
    using clock = std::chrono::steady_clock;

    TimerChrono() {}
    inline uint64_t ticks() {
        return std::chrono::time_point_cast<std::chrono::nanoseconds>(clock::now())
            .time_since_epoch()
            .count();
    }

    inline double ticks_to_usf(uint64_t tick_delta) {
        return (double)tick_delta / 1000.0;
    }
    inline double ticks_to_nsf(uint64_t tick_delta) {
        return (double)tick_delta;
    }
    inline uint64_t ticks_to_us(uint64_t tick_delta) {
        return tick_delta / 1000;
    }
    inline uint64_t ticks_to_ns(uint64_t tick_delta) {
        return tick_delta;
    }
};

#if defined(_WIN32) || defined(_WIN64)

#include "windows.h"
#include <intrin.h>
class TimerWindows {
public:
    TimerWindows() {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        m_frequency = freq.QuadPart;
    }

    inline uint64_t ticks() {
        LARGE_INTEGER qpcnt;
        int rval = QueryPerformanceCounter(&qpcnt);
        return qpcnt.QuadPart;
    }

    inline double ticks_to_usf(uint64_t tick_delta) {
        return (double)tick_delta * 1000000 / m_frequency;
    }
    inline double ticks_to_nsf(uint64_t tick_delta) {
        return (double)tick_delta * 1000000000 / m_frequency;
    }
    inline uint64_t ticks_to_us(uint64_t tick_delta) {
        return tick_delta * 1000000 / m_frequency;
    }
    inline uint64_t ticks_to_ns(uint64_t tick_delta) {
        return tick_delta * 1000000000 / m_frequency;
    }

private:
    uint64_t m_frequency;
};

using Timer = TimerWindows;

#elif defined(__linux__)

#include <sched.h>
// https://stackoverflow.com/questions/42189976/calculate-system-time-using-rdtsc
// Discussion describes how clock_gettime() costs about 4 ns per call
class TimerLinux {
public:
    TimerLinux() {}

    inline uint64_t ticks() {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return (static_cast<uint64_t>(1000000000UL) *
                static_cast<uint64_t>(ts.tv_sec) +
                static_cast<uint64_t>(ts.tv_nsec));
    }

    inline double ticks_to_usf(uint64_t tick_delta) {
        return (double)tick_delta / 1000.0;
    }
    inline double ticks_to_nsf(uint64_t tick_delta) {
        return (double)tick_delta;
    }
    inline uint64_t ticks_to_us(uint64_t tick_delta) {
        return tick_delta / 1000;
    }
    inline uint64_t ticks_to_ns(uint64_t tick_delta) {
        return tick_delta;
    }
};

using Timer = TimerLinux;

#else

using Timer = TimerChrono;

#endif

inline Timer& getTimer()
{
    static Timer timer;
    return timer;
}

class StatsAggregator
{
public:
    StatsAggregator() = default;
    ~StatsAggregator()
    {
        //std::lock_guard<std::mutex> lock(m_Mutex);
        auto& os = std::cout;

        if( !m_HostTimingStatsMap.empty() )
        {
            uint64_t    totalTotalNS = 0;
            size_t      longestName = 32;

            os << std::endl << "Call Profiling Results:" << std::endl;

            // Move data from the unordered map to an ordered map for better reporting.
            typedef std::map<std::string, SHostTimingStats>COrderedHostTimingStatsMap;
            COrderedHostTimingStatsMap stats;

            for (const auto& i : m_HostTimingStatsMap)
            {
                const std::string& name = i.first;
                const SHostTimingStats& hostTimingStats = i.second;

                if( !name.empty() )
                {
                    totalTotalNS += hostTimingStats.TotalNS;
                    longestName = std::max< size_t >( name.length(), longestName );

                    stats[name] = hostTimingStats;
                }
            }

            os << std::endl << "Total Time (ns): " << totalTotalNS << std::endl;
            os << std::endl
                << std::right << std::setw(longestName) << "Function Name" << ", "
                << std::right << std::setw( 6) << "Calls" << ", "
                << std::right << std::setw(13) << "Time (ns)" << ", "
                << std::right << std::setw( 8) << "Time (%)" << ", "
                << std::right << std::setw(13) << "Average (ns)" << ", "
                << std::right << std::setw(13) << "Min (ns)" << ", "
                << std::right << std::setw(13) << "Max (ns)" << std::endl;

            // Now report the data from the ordered map.
            for (const auto& i : stats)
            {
                const std::string& name = i.first;
                const SHostTimingStats& hostTimingStats = i.second;

                os << std::right << std::setw(longestName) << name << ", "
                    << std::right << std::setw( 6) << hostTimingStats.NumberOfCalls << ", "
                    << std::right << std::setw(13) << hostTimingStats.TotalNS << ", "
                    << std::right << std::setw( 7) << std::fixed << std::setprecision(2) << hostTimingStats.TotalNS * 100.0f / totalTotalNS << "%, "
                    << std::right << std::setw(13) << hostTimingStats.TotalNS / hostTimingStats.NumberOfCalls << ", "
                    << std::right << std::setw(13) << hostTimingStats.MinNS << ", "
                    << std::right << std::setw(13) << hostTimingStats.MaxNS << std::endl;
            }
        }
    }

    void addRecord(const char* label, uint64_t delta)
    {
        //std::lock_guard<std::mutex> lock(m_Mutex);
        SHostTimingStats&   stats = m_HostTimingStatsMap[label];

        stats.NumberOfCalls++;
        stats.TotalNS += delta;
        stats.MinNS = std::min<uint64_t>(stats.MinNS, delta);
        stats.MaxNS = std::max<uint64_t>(stats.MaxNS, delta);
    }

private:
    //std::mutex m_Mutex;

    struct SHostTimingStats
    {
        SHostTimingStats() :
            NumberOfCalls(0),
            MinNS(UINT64_MAX),
            MaxNS(0),
            TotalNS(0) {}

        uint64_t    NumberOfCalls;
        uint64_t    MinNS;
        uint64_t    MaxNS;
        uint64_t    TotalNS;
    };

#if defined(USE_STRING_KEYS)
    typedef std::unordered_map<std::string, SHostTimingStats>   CHostTimingStatsMap;
#else
    typedef std::unordered_map<const char*, SHostTimingStats>   CHostTimingStatsMap;
#endif // defined(USE_STRING_KEYS)
    CHostTimingStatsMap  m_HostTimingStatsMap;
};

inline StatsAggregator& getAggregator()
{
    static StatsAggregator aggregator;
    return aggregator;
}

class ScopeProfiler
{
public:
    ScopeProfiler(const char* label) :
        m_Label(label), m_StartTicks(getTimer().ticks()) {}
    ~ScopeProfiler()
    {
        Timer& timer = getTimer();
        uint64_t tick_delta = timer.ticks() - m_StartTicks;
        uint64_t ns_delta = timer.ticks_to_ns(tick_delta);

        getAggregator().addRecord(m_Label, ns_delta);
    }

private:
    const char* m_Label;
    uint64_t m_StartTicks;
};

#define PROFILE_SCOPE(_label)  \
    ScopeProfiler _prof(_label)
