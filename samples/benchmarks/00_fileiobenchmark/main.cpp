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

#define CLI_SPRINTF(_s, _sz, _f, ...)   sprintf_s(_s, _TRUNCATE, _f, ##__VA_ARGS__)
#define CLI_VSPRINTF(_s, _sz, _f, _a)   vsnprintf_s(_s, _TRUNCATE, _f, _a)

#else

#include <pthread.h>

#define GET_PROCESS_ID()    getpid()
#define GET_THREAD_ID()     pthread_self()

#define CLI_SPRINTF(_s, _sz, _f, ...)   snprintf(_s, _sz, _f, ##__VA_ARGS__)
#define CLI_VSPRINTF(_s, _sz, _f, _a)   vsnprintf(_s, _sz, _f, _a)

#endif

using test_clock = std::chrono::steady_clock;






struct ofstream_fixture : public benchmark::Fixture
{
    std::ofstream   trace;
    uint64_t    fixture_processId;
    uint64_t    fixture_threadId;
    test_clock::time_point   global_start;
    test_clock::time_point   start;
    test_clock::time_point   end;

    static const size_t BUFFER_SIZE = 16 * 1024;
    char buffer[BUFFER_SIZE];

    virtual void SetUp(benchmark::State& state) override {
        global_start = test_clock::now();

        trace.open( "dummy_trace_ofstream.json", std::ios::out | std::ios::binary );
        trace << "[ \n";

        start = test_clock::now();

        fixture_processId = GET_PROCESS_ID();
        fixture_threadId = GET_THREAD_ID();
        std::string processName = "dummy_process";
        trace
            << "{\"ph\":\"M\", \"name\":\"process_name\", \"pid\":" << fixture_processId
            << ", \"tid\":" << fixture_threadId
            << ", \"args\":{\"name\":\"" << processName
            << "\"}},\n";

        end = test_clock::now();
    }
    virtual void TearDown(benchmark::State& state) override {
        trace.close();
    }
};

BENCHMARK_DEFINE_F(ofstream_fixture, ChromeCallLogging_StringLiteral)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        trace << "{\"ph\":\"X\",\"pid\":33692,\"tid\":33752,\"name\":\"test_function\",\"ts\":9191,\"dur\":8},\n";
    }
}
BENCHMARK_REGISTER_F(ofstream_fixture, ChromeCallLogging_StringLiteral);

BENCHMARK_DEFINE_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += "test_function";

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
        std::string name("test_function");

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
        std::string name("test_function");

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

BENCHMARK_DEFINE_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_fixturepid)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name("test_function");

        std::ostringstream args;

        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        trace
            << "{\"ph\":\"X\",\"pid\":" << fixture_processId
            << ",\"tid\":" << threadId
            << ",\"name\":\"" << name
            << "\",\"ts\":" << usStart
            << ",\"dur\":" << usDelta
            << args.str()
            << "},\n";
    }
}
BENCHMARK_REGISTER_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_fixturepid);

BENCHMARK_DEFINE_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_sprintf_insertion)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += "test_function";

        std::ostringstream args;

        uint64_t    processId = GET_PROCESS_ID();
        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        int size = CLI_SPRINTF(buffer, BUFFER_SIZE,
            "{\"ph\":\"X\",\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"name\":\"%s\",\"ts\":%" PRIu64 ",\"dur\":%" PRIu64 "%s},\n",
            processId,
            threadId,
            name.c_str(),
            usStart,
            usDelta,
            args.str().c_str() );
        trace << buffer;
    }
}
BENCHMARK_REGISTER_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_sprintf_insertion);

BENCHMARK_DEFINE_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_sprintf_write)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += "test_function";

        std::ostringstream args;

        uint64_t    processId = GET_PROCESS_ID();
        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        int size = CLI_SPRINTF(buffer, BUFFER_SIZE,
            "{\"ph\":\"X\",\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"name\":\"%s\",\"ts\":%" PRIu64 ",\"dur\":%" PRIu64 "%s},\n",
            processId,
            threadId,
            name.c_str(),
            usStart,
            usDelta,
            args.str().c_str() );
        trace.write(buffer, size);
    }
}
BENCHMARK_REGISTER_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_sprintf_write);

BENCHMARK_DEFINE_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_fixturepid_sprintf_insertion)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += "test_function";

        std::ostringstream args;

        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        int size = CLI_SPRINTF(buffer, BUFFER_SIZE,
            "{\"ph\":\"X\",\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"name\":\"%s\",\"ts\":%" PRIu64 ",\"dur\":%" PRIu64 "%s},\n",
            fixture_processId,
            threadId,
            name.c_str(),
            usStart,
            usDelta,
            args.str().c_str() );
        trace << buffer;
    }
}
BENCHMARK_REGISTER_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_fixturepid_sprintf_insertion);

BENCHMARK_DEFINE_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_fixturepid_sprintf_write)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += "test_function";

        std::ostringstream args;

        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        int size = CLI_SPRINTF(buffer, BUFFER_SIZE,
            "{\"ph\":\"X\",\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"name\":\"%s\",\"ts\":%" PRIu64 ",\"dur\":%" PRIu64 "%s},\n",
            fixture_processId,
            threadId,
            name.c_str(),
            usStart,
            usDelta,
            args.str().c_str() );
        trace.write(buffer, size);
    }
}
BENCHMARK_REGISTER_F(ofstream_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_fixturepid_sprintf_write);






struct ofstream_text_fixture : public benchmark::Fixture
{
    std::ofstream   trace;
    uint64_t    fixture_processId;
    uint64_t    fixture_threadId;
    test_clock::time_point   global_start;
    test_clock::time_point   start;
    test_clock::time_point   end;

    static const size_t BUFFER_SIZE = 16 * 1024;
    char buffer[BUFFER_SIZE];

    virtual void SetUp(benchmark::State& state) override {
        global_start = test_clock::now();

        trace.open( "dummy_trace_ofstream.json", std::ios::out );
        trace << "[ \n";

        start = test_clock::now();

        fixture_processId = GET_PROCESS_ID();
        fixture_threadId = GET_THREAD_ID();
        std::string processName = "dummy_process";
        trace
            << "{\"ph\":\"M\", \"name\":\"process_name\", \"pid\":" << fixture_processId
            << ", \"tid\":" << fixture_threadId
            << ", \"args\":{\"name\":\"" << processName
            << "\"}},\n";

        end = test_clock::now();
    }
    virtual void TearDown(benchmark::State& state) override {
        trace.close();
    }
};

BENCHMARK_DEFINE_F(ofstream_text_fixture, ChromeCallLogging_StringLiteral)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        trace << "{\"ph\":\"X\",\"pid\":33692,\"tid\":33752,\"name\":\"test_function\",\"ts\":9191,\"dur\":8},\n";
    }
}
BENCHMARK_REGISTER_F(ofstream_text_fixture, ChromeCallLogging_StringLiteral);

BENCHMARK_DEFINE_F(ofstream_text_fixture, ChromeCallLogging_WithTimes_WithArgs)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += "test_function";

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
BENCHMARK_REGISTER_F(ofstream_text_fixture, ChromeCallLogging_WithTimes_WithArgs);






struct FILE_fixture : public benchmark::Fixture
{
    FILE*   trace;
    uint64_t    fixture_processId;
    uint64_t    fixture_threadId;
    test_clock::time_point   global_start;
    test_clock::time_point   start;
    test_clock::time_point   end;

    static const size_t BUFFER_SIZE = 16 * 1024;
    char buffer[BUFFER_SIZE];

    virtual void SetUp(benchmark::State& state) override {
        global_start = test_clock::now();

        trace = fopen( "dummy_trace_FILE.json", "wb" );
        fprintf(trace, "[ \n");

        start = test_clock::now();

        fixture_processId = GET_PROCESS_ID();
        fixture_threadId = GET_THREAD_ID();
        std::string processName = "dummy_process";
        fprintf(trace, "{\"ph\":\"M\", \"name\":\"process_name\", \"pid\":%" PRIu64 ", \"tid\":%" PRIu64 ", \"args\":{\"name\":\"%s\"}},\n",
            fixture_processId,
            fixture_threadId,
            processName.c_str() );

        end = test_clock::now();
    }
    virtual void TearDown(benchmark::State& state) override {
        fclose(trace);;
    }
};

BENCHMARK_DEFINE_F(FILE_fixture, ChromeCallLogging_StringLiteral)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        fprintf(trace, "{\"ph\":\"X\",\"pid\":33692,\"tid\":33752,\"name\":\"test_function\",\"ts\":9191,\"dur\":8},\n");
    }
}
BENCHMARK_REGISTER_F(FILE_fixture, ChromeCallLogging_StringLiteral);

BENCHMARK_DEFINE_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += "test_function";

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
        name += "test_function";

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

BENCHMARK_DEFINE_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_fixturepid)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += "test_function";

        std::ostringstream args;

        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        fprintf(trace, "{\"ph\":\"X\",\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"name\":\"%s\",\"ts\":%" PRIu64 ",\"dur\":%" PRIu64 "%s},\n",
            fixture_processId,
            threadId,
            name.c_str(),
            usStart,
            usDelta,
            args.str().c_str() );
    }
}
BENCHMARK_REGISTER_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_fixturepid);

BENCHMARK_DEFINE_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_sprintf_fprintf)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += "test_function";

        std::ostringstream args;

        uint64_t    processId = GET_PROCESS_ID();
        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        int size = CLI_SPRINTF(buffer, BUFFER_SIZE,
            "{\"ph\":\"X\",\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"name\":\"%s\",\"ts\":%" PRIu64 ",\"dur\":%" PRIu64 "%s},\n",
            processId,
            threadId,
            name.c_str(),
            usStart,
            usDelta,
            args.str().c_str() );
        fprintf(trace, "%s", buffer);
    }
}
BENCHMARK_REGISTER_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_sprintf_fprintf);

BENCHMARK_DEFINE_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_sprintf_fwrite)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += "test_function";

        std::ostringstream args;

        uint64_t    processId = GET_PROCESS_ID();
        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        int size = CLI_SPRINTF(buffer, BUFFER_SIZE,
            "{\"ph\":\"X\",\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"name\":\"%s\",\"ts\":%" PRIu64 ",\"dur\":%" PRIu64 "%s},\n",
            processId,
            threadId,
            name.c_str(),
            usStart,
            usDelta,
            args.str().c_str() );
        fwrite(buffer, size, 1, trace);
    }
}
BENCHMARK_REGISTER_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_sprintf_fwrite);

BENCHMARK_DEFINE_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_fixturepid_sprintf_fprintf)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += "test_function";

        std::ostringstream args;

        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        int size = CLI_SPRINTF(buffer, BUFFER_SIZE,
            "{\"ph\":\"X\",\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"name\":\"%s\",\"ts\":%" PRIu64 ",\"dur\":%" PRIu64 "%s},\n",
            fixture_processId,
            threadId,
            name.c_str(),
            usStart,
            usDelta,
            args.str().c_str() );
        fprintf(trace, "%s", buffer);
    }
}
BENCHMARK_REGISTER_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_fixturepid_sprintf_fprintf);

BENCHMARK_DEFINE_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_fixturepid_sprintf_fwrite)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        std::string name;
        name += "test_function";

        std::ostringstream args;

        uint64_t    threadId = GET_THREAD_ID();

        using us = std::chrono::microseconds;
        uint64_t    usStart =
            std::chrono::duration_cast<us>(start - global_start).count();
        uint64_t    usDelta =
            std::chrono::duration_cast<us>(end - start).count();

        int size = CLI_SPRINTF(buffer, BUFFER_SIZE,
            "{\"ph\":\"X\",\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"name\":\"%s\",\"ts\":%" PRIu64 ",\"dur\":%" PRIu64 "%s},\n",
            fixture_processId,
            threadId,
            name.c_str(),
            usStart,
            usDelta,
            args.str().c_str() );
        fwrite(buffer, size, 1, trace);
    }
}
BENCHMARK_REGISTER_F(FILE_fixture, ChromeCallLogging_WithTimes_WithArgs_nospaces_fixturepid_sprintf_fwrite);


BENCHMARK_MAIN();