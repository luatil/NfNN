#ifndef NFNN_TYPES_H
#define NFNN_TYPES_H

#include <stdbool.h>
#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef float f32;
typedef double f64;

#define KB(_X) (_X * 1024)
#define MB(_X) (KB(_X) * 1024)
#define GB(_X) (MB(_X) * 1024)

#endif // NFNN_TYPES_H