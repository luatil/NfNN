#ifndef NFNN_MACRO_H
#define NFNN_MACRO_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NFNN_EPS_FOR_EQUAL 0.0001f
#define NFNN_MINUS_INF_F32 -1.0e+30f
#define NFNN_EPS_FOR_LOG 1.0e-10f
#define NFNN_IS_NAN(_f) ((_f) != (_f))
#define NFNN_ATOI(_str) (atoi(_str))
#define NFNN_STRLEN(_str) (strlen(_str))
#define NFNN_U32TOSTR(_u32, _str) (sprintf(_str, "%u", _u32))
#define NFNN_STREQUAL(_strA, _strB) (strcmp(_strA, _strB) == 0)
#define NFNN_STRCPY(_strA, _strB) (strcpy(_strA, _strB))
#define NFNN_ATOF(_str) (atof(_str))

#if defined(_WIN32)
#define NFNN_DEBUG_BREAK() __debugbreak();
#define NFNN_FILE __FILE__
#define NFNN_FUNCTION __FUNCTION__
#define NFNN_PRINT_GLOBAL_ERROR()                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        DWORD errorMessageID = WSAGetLastError();                                                                      \
        LPSTR errorMessage = NULL;                                                                                     \
                                                                                                                       \
        size_t size = FormatMessageA(                                                                                  \
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,         \
            errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&errorMessage, 0, NULL);                 \
                                                                                                                       \
        fprintf(stderr, "Connect failed with error: %s\n", errorMessage);                                              \
                                                                                                                       \
        LocalFree(errorMessage);                                                                                       \
    } while (0)
#else
#include <signal.h>
#define NFNN_DEBUG_BREAK() raise(SIGTRAP);
#define NFNN_PRINT_GLOBAL_ERROR() fprintf(stderr, "Connect failed with error: %s\n", strerror(errno));
#define NFNN_FILE __FILE__
#define NFNN_FUNCTION __func__
#endif

// From beej's guide to c programming
#define NFNN_ASSERT(_expr, _msg)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(_expr))                                                                                                  \
        {                                                                                                              \
            fprintf(stderr, "%s--%s:%d: ASSERTION: %s\n", NFNN_FILE, NFNN_FUNCTION, __LINE__, #_msg);                  \
            NFNN_DEBUG_BREAK();                                                                                        \
        }                                                                                                              \
    } while (0)

#define NFNN_NOT_IMPLEMENTED()                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        fprintf(stderr, "%s--%s:%d: NOT IMPLEMENTED\n", NFNN_FILE, NFNN_FUNCTION, __LINE__);                           \
        NFNN_DEBUG_BREAK();                                                                                            \
    } while (0)

#define NFNN_TEST(_expr, _msg)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(_expr))                                                                                                  \
        {                                                                                                              \
            fprintf(stderr, "%s--%s:%d: FAIL: %s - : %s\n", NFNN_FILE, NFNN_FUNCTION, __LINE__, #_msg, _msg);          \
            NFNN_DEBUG_BREAK();                                                                                        \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            fprintf(stderr, "PASS: %s\n", _msg);                                                                       \
        }                                                                                                              \
    } while (0)

#define NFNN_ARRAY_COUNT(_array) (sizeof(_array) / sizeof(_array[0]))

#define NFNN_ERROR() NFNN_ASSERT(false, "Error")
#define NFNN_NOT_USED()

#define NFNN_SLL_PushBack(f, l, n)                                                                                     \
    ((f) == (0) ? ((f) = (l) = (n), (n)->Next = 0) : ((l)->Next = (n), (n)->Next = 0), (l) = (n))

#define NFNN_DLL_PushBack(f, l, n)                                                                                     \
    ((f) == (0) ? ((f) = (l) = (n), (n)->Next = 0, (n)->Prev = 0)                                                      \
                : ((l)->Next = (n), (n)->Next = 0, (n)->Prev = (l), (l) = (n)))

#define NFNN_DLL_Pop(f, l) (((f) == (0) || (f)->Next == (0)) ? ((f) = (l) = (0)) : ((l) = (l)->Prev, (l)->Next = 0))

#endif // NFNN_MACRO_H