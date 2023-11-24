#ifndef NFNN_RANDOM_H
#define NFNN_RANDOM_H

#include "nfnn_macro.h"
#include "nfnn_types.h"

#include <stdlib.h>

#define NFNN_RANDOM_U32_MAX 0xFFFFFFFF

typedef struct nfnn_random_state nfnn_random_state;
struct nfnn_random_state
{
    u32 State;
};

static void NfNN_Random_Init(nfnn_random_state *RandomState, u32 Seed)
{
    NFNN_ASSERT(RandomState != 0, "RandomState is null");
    RandomState->State = Seed;
}

static nfnn_random_state NfNN_Random_Seed(u32 Seed)
{
    nfnn_random_state Result = {0};
    NfNN_Random_Init(&Result, Seed);
    return Result;
}

static u32 NfNN_Random_Next(nfnn_random_state *RandomState)
{
    NFNN_ASSERT(RandomState != 0, "RandomState is null");
    // NOTE(luatil): https://en.wikipedia.org/wiki/Xorshift 32bit implementation
    u32 Result = RandomState->State;
    Result ^= Result << 13;
    Result ^= Result >> 17;
    Result ^= Result << 5;
    RandomState->State = Result;
    return Result;
}

static f32 NfNN_Random_ZeroToOne(nfnn_random_state *RandomState)
{
    f32 Divisor = 1.0f / (f32)NFNN_RANDOM_U32_MAX;
    f32 Result = Divisor * NfNN_Random_Next(RandomState);
    return Result;
}

static f32 NfNN_Random_Range_f32(nfnn_random_state *RandomState, f32 Min, f32 Max)
{
    f32 Result = Min + (Max - Min) * NfNN_Random_ZeroToOne(RandomState);
    return Result;
}

static u32 NfNN_Random_Range_u32(nfnn_random_state *RandomState, u32 Min, u32 Max)
{
    NFNN_ASSERT(Min < Max, "Min must be less than Max");
    NFNN_ASSERT(Min >= 0, "Min must be non-negative");
    u32 Result = Min + (u32)((Max - Min) * NfNN_Random_ZeroToOne(RandomState));
    return Result;
}

static void NfNN_Random_ShuffleInPlace_u32(nfnn_random_state *RandomState, u32 *Array, u32 N)
{
    for (u32 I = 0; I < N; I++)
    {
        u32 Index = NfNN_Random_Range_u32(RandomState, 0, N - 1);
        u32 Temp = Array[I];
        Array[I] = Array[Index];
        Array[Index] = Temp;
    }
}

static void NfNN_Random_UniformArrayInRange_f32(nfnn_random_state *RandomState, f32 *Array, u32 N, f32 Min, f32 Max)
{
    for (u32 I = 0; I < N; I++)
    {
        Array[I] = NfNN_Random_Range_f32(RandomState, Min, Max);
    }
}

static void NfNN_Random_UniformArrayInRange_u32(nfnn_random_state *RandomState, u32 *Array, u32 N, u32 Min, u32 Max)
{
    for (u32 I = 0; I < N; I++)
    {
        Array[I] = NfNN_Random_Range_u32(RandomState, Min, Max);
    }
}

#endif // NFNN_RANDOM_H
