#ifndef NFNN_PLATFORM_H
#define NFNN_PLATFORM_H

#if defined(_WIN32)
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#else
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#endif

#if defined(_WIN32)
#define NFNN_ISVALIDSOCKET(s) ((s) != INVALID_SOCKET)
#define NFNN_CLOSESOCKET(s) closesocket(s)
#define NFNN_GETSOCKETERRNO() (WSAGetLastError())
typedef SOCKET nfnn_platform_socket;
#define NFNN_PLATFORM_SLEEP(_seconds) Sleep(_seconds * 1000)
#define NFNN_PRINT_WORKING_DIR()                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        char Buffer[1024];                                                                                             \
        GetCurrentDirectoryA(1024, Buffer);                                                                            \
        printf("Working dir: %s\n", Buffer);                                                                           \
    } while (0)
#else
#define NFNN_ISVALIDSOCKET(s) ((s) >= 0)
#define NFNN_CLOSESOCKET(s) close(s)
typedef int nfnn_platform_socket;
#define NFNN_GETSOCKETERRNO() (errno)
#define NFNN_PLATFORM_SLEEP(_seconds) usleep((_seconds * 1000) * 1000)
#define NFNN_PRINT_WORKING_DIR()                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        char Buffer[1024];                                                                                             \
        getcwd(Buffer, 1024);                                                                                          \
        printf("Working dir: %s\n", Buffer);                                                                           \
    } while (0)
#endif

#define NFNN_MAXHOST NI_MAXHOST

typedef struct addrinfo nfnn_addrinfo;

#define NFNN_AI_PASSIVE AI_PASSIVE
#define NFNN_SOCK_STREAM SOCK_STREAM
#define NFNN_IPPROTO_TCP IPPROTO_TCP
#define NFNN_AF_UNSPEC AF_UNSPEC

static void NfNN_Network_GetAddrinfo(nfnn_addrinfo *Hints, nfnn_addrinfo **Result, char *Host, char *Port)
{
    int R = getaddrinfo(Host, Port, Hints, Result);
    NFNN_ASSERT(R == 0, "getaddrinfo failed");
}

typedef fd_set nfnn_fd_set;

#define NFNN_FD_ZERO(set) FD_ZERO(set)
#define NFNN_FD_SET(socket, set) FD_SET(socket, set)
#define NFNN_FD_ISSET(socket, set) FD_ISSET(socket, set)

#endif // NFNN_PLATFORM_H
