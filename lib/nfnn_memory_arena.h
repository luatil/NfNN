#ifndef NFNN_MEMORY_ARENA_H 
#define NFNN_MEMORY_ARENA_H 

#include <stdlib.h>
#include <memory.h>

#include "nfnn_types.h"
#include "nfnn_macro.h"

typedef struct nfnn_memory_arena nfnn_memory_arena;
struct nfnn_memory_arena
{
    u8 *Base;
    u64 Size;
    u64 Used;
    u64 TempCount;
};

static void NfNN_MemoryArena_Init(nfnn_memory_arena *Arena, u64 Size)
{
    Arena->Base = (u8 *)calloc(Size, 1);
    Arena->Size = Size;
    Arena->Used = 0;
}

static u8 *NfNN_MemoryArena_Alloc(nfnn_memory_arena *Arena, u64 Size)
{
    u8 *Result = Arena->Base + Arena->Used;
    Arena->Used += Size;
    memset(Result, 0, Size);
    return Result;
}

static void NfNN_MemoryArena_Clear(nfnn_memory_arena *Arena)
{
    memset(Arena->Base, 0, Arena->Size);
    Arena->Used = 0;
}

static void NfNN_MemoryArena_TempInit(nfnn_memory_arena *Arena)
{
    // TODO(luatil): Allow this to be nested
    Arena->TempCount = Arena->Used;
}

static void NfNN_MemoryArena_TempClear(nfnn_memory_arena *Arena)
{
    memset(Arena->Base + Arena->TempCount, 0, Arena->Used - Arena->TempCount);
    Arena->Used = Arena->TempCount;
}

static void *NfNN__PushSize(nfnn_memory_arena *Arena, u32 Size)
{
    NFNN_ASSERT((Arena->Used + Size) <= Arena->Size, "Memory arena overflow.");
    void *Result = Arena->Base + Arena->Used;
    Arena->Used += Size;
    return Result;
}

static void NfNN__MemoryCopy(void *Dest, void *Source, u64 Size)
{
    memcpy(Dest, Source, Size);
}

#define NfNN_MemoryCopy(_Dest, _Source, _Size)                                      \
    NfNN__MemoryCopy((void *)(_Dest), (void *)(_Source), (_Size))

#define NfNN_PushArray(_Arena, _Type, _Count)                                       \
    (_Type *)NfNN__PushSize(_Arena, sizeof(_Type) * (_Count))

#define NfNN_PushStruct(_Arena, _Type) (_Type *)NfNN__PushSize(_Arena, sizeof(_Type))

#define NfNN_PushTensor(_Arena, _Dim) NfNN_PushArray(_Arena, f32, NfNN_DimSize(_Dim))


#endif // NFNN_MEMORY_ARENA_H 
