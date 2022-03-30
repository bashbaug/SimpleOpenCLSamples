/*
// Copyright (c) 2019-2021 Ben Ashbaugh
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
*/

#include <benchmark/benchmark.h>

#include <chrono>
#include <fstream>
#include <sstream>

#include <cinttypes>
#include <cstdio>

#if defined(_WIN32)

#include <Windows.h>

#define GET_PROCESS_ID()    GetCurrentProcessId()
#define GET_THREAD_ID()     GetCurrentThreadId()

#else

#include <pthread.h>

#define GET_PROCESS_ID()    getpid()
#define GET_THREAD_ID()     pthread_self()

#endif

using test_clock = std::chrono::steady_clock;

struct ofstream_fixture : public benchmark::Fixture
{
    std::ofstream   trace;
    test_clock::time_point   global_start;
    test_clock::time_point   start;
    test_clock::time_point   end;

    virtual void SetUp(benchmark::State& state) override {
        global_start = test_clock::now();

        trace.open( "dummy_trace_ofstream.json", std::ios::out );
        trace << "[\n";

        start = test_clock::now();

        uint64_t    processId = GET_PROCESS_ID();
        uint64_t    threadId = GET_THREAD_ID();
        std::string processName = "dummy_process";
        trace
            << "{\"ph\":\"M\", \"name\":\"process_name\", \"pid\":" << processId
            << ", \"tid\":" << threadId
            << ", \"args\":{\"name\":\"" << processName
            << "\"}},\n";

        end = test_clock::now();
    }
    virtual void TearDown(benchmark::State& state) override {
        trace.close();
    }
};

BENCHMARK_DEFINE_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += __FUNCTION__;

        std::ostringstream args;

        uint64_t    processId = GET_PROCESS_ID();
        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        trace
            << "{\"ph\":\"X\", \"pid\":" << processId
            << ", \"tid\":" << threadId
            << ", \"name\":\"" << name
            << "\", \"ts\":" << usStart
            << ", \"dur\":" << usDelta
            << args.str()
            << "},\n";
    }
}
BENCHMARK_REGISTER_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs);

BENCHMARK_DEFINE_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_init)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name(__FUNCTION__);

        std::ostringstream args;

        uint64_t    processId = GET_PROCESS_ID();
        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        trace
            << "{\"ph\":\"X\", \"pid\":" << processId
            << ", \"tid\":" << threadId
            << ", \"name\":\"" << name
            << "\", \"ts\":" << usStart
            << ", \"dur\":" << usDelta
            << args.str()
            << "},\n";
    }
}
BENCHMARK_REGISTER_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_init);

BENCHMARK_DEFINE_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name(__FUNCTION__);

        std::ostringstream args;

        uint64_t    processId = GET_PROCESS_ID();
        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        trace
            << "{\"ph\":\"X\",\"pid\":" << processId
            << ",\"tid\":" << threadId
            << ",\"name\":\"" << name
            << "\",\"ts\":" << usStart
            << ",\"dur\":" << usDelta
            << args.str()
            << "},\n";
    }
}
BENCHMARK_REGISTER_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces);

struct FILE_fixture : public benchmark::Fixture
{
    FILE*   trace;
    test_clock::time_point   global_start;
    test_clock::time_point   start;
    test_clock::time_point   end;

    virtual void SetUp(benchmark::State& state) override {
        global_start = test_clock::now();

        trace = fopen( "dummy_trace_FILE.json", "wb" );
        fprintf(trace, "[\n");

        start = test_clock::now();

        uint64_t    processId = GET_PROCESS_ID();
        uint64_t    threadId = GET_THREAD_ID();
        std::string processName = "dummy_process";
        fprintf(trace, "{\"ph\":\"M\", \"name\":\"process_name\", \"pid\":%" PRIu64 ", \"tid\":%" PRIu64 ", \"args\":{\"name\":\"%s\"}},\n",
            processId,
            threadId,
            processName.c_str() );

        end = test_clock::now();
    }
    virtual void TearDown(benchmark::State& state) override {
        fclose(trace);;
    }
};

BENCHMARK_DEFINE_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += __FUNCTION__;

        std::ostringstream args;

        uint64_t    processId = GET_PROCESS_ID();
        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        fprintf(trace, "{\"ph\":\"X\", \"pid\":%" PRIu64 ", \"tid\":%" PRIu64 ", \"name\":\"%s\", \"ts\":%" PRIu64 ", \"dur\":%" PRIu64 "%s},\n",
            processId,
            threadId,
            name.c_str(),
            usStart,
            usDelta,
            args.str().c_str() );
    }
}
BENCHMARK_REGISTER_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs);

BENCHMARK_DEFINE_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += __FUNCTION__;

        std::ostringstream args;

        uint64_t    processId = GET_PROCESS_ID();
        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        fprintf(trace, "{\"ph\":\"X\",\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"name\":\"%s\",\"ts\":%" PRIu64 ",\"dur\":%" PRIu64 "%s},\n",
            processId,
            threadId,
            name.c_str(),
            usStart,
            usDelta,
            args.str().c_str() );
    }
}
BENCHMARK_REGISTER_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces);


BENCHMARK_MAIN();