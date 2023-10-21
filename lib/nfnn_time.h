#ifndef NFNN_TIME_H
#define NFNN_TIME_H

#if defined(_WIN32)
#include <Windows.h>
#include <stdint.h>
#else
#include <sys/time.h>
#endif

#include <time.h>
#include <stdio.h>
#include "nfnn_types.h"

typedef struct nfnn_time nfnn_time;
struct nfnn_time
{
    s64 Seconds;
    s64 Microseconds;
#if 0
  time_t Seconds;
  suseconds_t Microseconds;
#endif
};

typedef struct nfnn_time_diff nfnn_time_diff;
struct nfnn_time_diff
{
    s64 Seconds;
    s64 Microseconds;
};

static nfnn_time
NfNN_Time_CurrentTime()
{
    nfnn_time Result = {0};
#if defined(WIN32)
    static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);
    SYSTEMTIME system_time;
    FILETIME file_time;
    u64 time;
    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);
    time = ((u64)file_time.dwLowDateTime);
    time += ((u64)file_time.dwHighDateTime) << 32;
    Result.Seconds = (s64)((time - EPOCH) / 10000000L);
    Result.Microseconds = (s64)(system_time.wMilliseconds * 1000);
#else
    struct timeval TimeVal = {0};
    gettimeofday(&TimeVal, 0);
    Result.Seconds = TimeVal.tv_sec;
    Result.Microseconds = TimeVal.tv_usec;
#endif
    return Result;
}

static nfnn_time_diff
NfNN_Time_Diff(nfnn_time Start, nfnn_time End)
{
    nfnn_time_diff Result = {0};
    Result.Seconds = End.Seconds - Start.Seconds;
    Result.Microseconds = End.Microseconds - Start.Microseconds;
    if (Result.Microseconds < 0)
    {
        Result.Seconds -= 1;
        Result.Microseconds += 1000000;
    }
    return Result;
}

static void
NfNN_Time_Format(nfnn_time TimeInfo, char *Buffer, u32 BufferSize)
{
    struct tm *Time = localtime((time_t *)&TimeInfo.Seconds);
#if 1
    strftime(Buffer, BufferSize, "%Y-%m-%d %H:%M:%S", Time);
#else
    strftime(Buffer, BufferSize, "[%d %b %Y %H:%M:%S]", Time);
#endif
}

#endif // NFNN_TIME_H 
