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
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

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
            os << std::endl << "Call Profiling Results:" << std::endl;

            std::vector<std::string> keys;
            keys.reserve(m_HostTimingStatsMap.size());

            uint64_t    totalTotalNS = 0;
            size_t      longestName = 32;

            CHostTimingStatsMap::const_iterator i = m_HostTimingStatsMap.begin();
            while( i != m_HostTimingStatsMap.end() )
            {
                const std::string& name = (*i).first;
                const SHostTimingStats& hostTimingStats = (*i).second;

                if( !name.empty() )
                {
                    keys.push_back(name);
                    totalTotalNS += hostTimingStats.TotalNS;
                    longestName = std::max< size_t >( name.length(), longestName );
                }

                ++i;
            }

            std::sort(keys.begin(), keys.end());

            os << std::endl << "Total Time (ns): " << totalTotalNS << std::endl;

            os << std::endl
                << std::right << std::setw(longestName) << "Function Name" << ", "
                << std::right << std::setw( 6) << "Calls" << ", "
                << std::right << std::setw(13) << "Time (ns)" << ", "
                << std::right << std::setw( 8) << "Time (%)" << ", "
                << std::right << std::setw(13) << "Average (ns)" << ", "
                << std::right << std::setw(13) << "Min (ns)" << ", "
                << std::right << std::setw(13) << "Max (ns)" << std::endl;

            for( const auto& name : keys )
            {
                const SHostTimingStats& hostTimingStats = m_HostTimingStatsMap.at(name);

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

    void addRecord(const std::string& func, uint64_t delta)
    {
        //std::lock_guard<std::mutex> lock(m_Mutex);
        SHostTimingStats&   stats = m_HostTimingStatsMap[func];

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

    typedef std::unordered_map<std::string, SHostTimingStats>   CHostTimingStatsMap;
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
    using clock = std::chrono::steady_clock;

    ScopeProfiler(const char* func) :
        m_FuncName(func), m_Start(clock::now()) {}
    ~ScopeProfiler(void)
    {
        clock::time_point   end = clock::now();

        using ns = std::chrono::nanoseconds;
        uint64_t    delta = std::chrono::duration_cast<ns>(end - m_Start).count();

        getAggregator().addRecord(m_FuncName, delta);
    }

private:
    const char* m_FuncName;
    clock::time_point   m_Start;
};

#define PROFILE_SCOPE(_label)  \
    ScopeProfiler _prof(_label)
