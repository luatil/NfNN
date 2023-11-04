#ifndef NFNN_NETWORK_H
#define NFNN_NETWORK_H

#include "nfnn_types.h"
#include "nfnn_memory_arena.h"
#include "nfnn_macro.h"
#include "nfnn_tensor.h"
#include "nfnn_platform.h"

typedef struct nfnn_socket nfnn_socket;
struct nfnn_socket
{
    nfnn_platform_socket Handle;
};

typedef struct nfnn_parameter_server nfnn_parameter_server;
struct nfnn_parameter_server
{
    char Host[NFNN_MAXHOST];
    u32 Port;
    nfnn_socket *Socket;
    nfnn_fd_set *Master;
    nfnn_fd_set *Working;
    nfnn_platform_socket MaxSocket;
};

typedef struct nfnn_network_interface nfnn_network_interface;
struct nfnn_network_interface
{
    #if defined(_WIN32)
    WSADATA WsaData;
    #endif
};

static nfnn_socket *
NfNN_Network_TCPConnect(nfnn_memory_arena *Mem, char *Host, u32 Port)
{
    nfnn_socket *Result = NfNN_PushStruct(Mem, nfnn_socket);

    nfnn_addrinfo Hints = {0};
    nfnn_addrinfo *R = 0;

    // Hints.ai_flags = NFNN_AI_PASSIVE;
    Hints.ai_socktype = NFNN_SOCK_STREAM;
    Hints.ai_protocol = NFNN_IPPROTO_TCP;
    Hints.ai_family = NFNN_AF_UNSPEC;

    char PortString[16];
    NFNN_U32TOSTR(Port, PortString);

    NfNN_Network_GetAddrinfo(&Hints, &R, Host, PortString);

    Result->Handle = socket(R->ai_family, R->ai_socktype, R->ai_protocol);

    NFNN_ASSERT(NFNN_ISVALIDSOCKET(Result->Handle), "socket failed");

    if (connect(Result->Handle, R->ai_addr, R->ai_addrlen))
    {
        NFNN_PRINT_GLOBAL_ERROR();
        NFNN_ASSERT(0, "connect failed");
        NFNN_CLOSESOCKET(Result->Handle);
    }

    freeaddrinfo(R);

    return Result;
}

static nfnn_socket *
NfNN_Network_TCPListeningSocket(nfnn_memory_arena *Mem, char *Host, u32 Port)
{
    nfnn_socket *Result = NfNN_PushStruct(Mem, nfnn_socket);

    nfnn_addrinfo Hints = {0};
    nfnn_addrinfo *R = 0;

    Hints.ai_flags = NFNN_AI_PASSIVE;
    Hints.ai_socktype = NFNN_SOCK_STREAM;
    Hints.ai_protocol = NFNN_IPPROTO_TCP;
    Hints.ai_family = NFNN_AF_UNSPEC;

    char PortString[16];
    NFNN_U32TOSTR(Port, PortString);

    NfNN_Network_GetAddrinfo(&Hints, &R, Host, PortString);

    Result->Handle = socket(R->ai_family, R->ai_socktype, R->ai_protocol);

    NFNN_ASSERT(NFNN_ISVALIDSOCKET(Result->Handle), "socket failed");

    int ReuseAddr = 1;
    if (setsockopt(Result->Handle, SOL_SOCKET, SO_REUSEADDR, (const char *)&ReuseAddr, sizeof(ReuseAddr)) < 0)
    {
        NFNN_ASSERT(0, "setsockopt failed");
    }

    if (bind(Result->Handle, R->ai_addr, R->ai_addrlen) < 0)
    {
        NFNN_ASSERT(0, "bind failed");
    }

    if (listen(Result->Handle, 10) < 0)
    {
        NFNN_ASSERT(0, "listen failed");
    }

    freeaddrinfo(R);

    return Result;
}

static nfnn_network_interface *
NfNN_Network_CreateInterface(nfnn_memory_arena *Mem)
{
    nfnn_network_interface *Result = NfNN_PushStruct(Mem, nfnn_network_interface);

    #if defined(_WIN32)
    int WSAResult = WSAStartup(MAKEWORD(2, 2), &Result->WsaData);
    NFNN_ASSERT(WSAResult == 0, "WSAStartup failed");
    // TODO(luatil): Do better error handling
    // printf("WSAStartup failed: %d\n", WSAGetLastError());
    #endif

    return Result;
}

static void
NfNN_Network_DestroyInterface(nfnn_network_interface *Interface)
{
    #if defined(_WIN32)
    WSACleanup();
    #endif
}

static nfnn_parameter_server *
NfNN_ParameterServer_Create(nfnn_memory_arena *Mem, char *Host, u32 Port)
{
    nfnn_parameter_server *Result = NfNN_PushStruct(Mem, nfnn_parameter_server);

    u32 HostLength = NFNN_STRLEN(Host);
    NfNN_MemoryCopy(Result->Host, Host, HostLength);

    Result->Port = Port;
    Result->Socket = NfNN_Network_TCPListeningSocket(Mem, Host, Port); 

    Result->Master = NfNN_PushStruct(Mem, nfnn_fd_set);
    Result->Working = NfNN_PushStruct(Mem, nfnn_fd_set);

    NFNN_FD_ZERO(Result->Master);
    NFNN_FD_ZERO(Result->Working);
    NFNN_FD_SET(Result->Socket->Handle, Result->Master);
    Result->MaxSocket = Result->Socket->Handle;

    return Result;
}

static void
NfNN_ParameterServer_AddWorker(nfnn_memory_arena *Mem, nfnn_parameter_server *Server)
{
    NfNN_MemoryCopy(Server->Working, Server->Master, sizeof(nfnn_fd_set));

    int SelectResult = select(Server->MaxSocket + 1, Server->Working, 0, 0, 0);

    NFNN_ASSERT(SelectResult >= 0, "select failed");

    for (nfnn_platform_socket SockIt = 1; SockIt <= Server->MaxSocket; SockIt++)
    {
        if (NFNN_FD_ISSET(SockIt, Server->Working))
        {
            if (SockIt == Server->Socket->Handle)
            {
                // New connection (accept
                struct sockaddr PeerAddr = {0};
                socklen_t PeerAddrlen = sizeof(struct sockaddr);

                nfnn_platform_socket Peer = accept(Server->Socket->Handle, &PeerAddr, &PeerAddrlen);

                if (Peer == 0)
                {
                    printf("accept failed with error: %d\n", NFNN_GETSOCKETERRNO());
                    NFNN_CLOSESOCKET(Server->Socket->Handle);
                    return;
                }

                char NodeBuffer[NI_MAXHOST] = {0};
                char ServiceBuffer[NI_MAXSERV] = {0};

                getnameinfo(&PeerAddr,
                            PeerAddrlen,
                            NodeBuffer,
                            sizeof(NodeBuffer),
                            ServiceBuffer,
                            sizeof(ServiceBuffer),
                            NI_NUMERICHOST | NI_NUMERICSERV);

                printf("New connection from %s:%s\n", NodeBuffer, ServiceBuffer);

                NFNN_FD_SET(Peer, Server->Master);
                Server->MaxSocket = (Peer > Server->MaxSocket) ? Peer : Server->MaxSocket;
            }
            else
            {
                NFNN_ERROR();
            }
        }
        else
        {
            NFNN_NOT_USED();
        }
    }
}

static void 
NfNN_Network_RecvSize(nfnn_platform_socket Sock, u32 SizeInBytes, char *Dst)
{
    int BytesReceived = 0;
    int RemainingBytes = SizeInBytes;

    while(RemainingBytes > 0)
    {
        int Recv = recv(Sock, Dst + BytesReceived, RemainingBytes, 0);
        if (Recv > 0 && Recv <= RemainingBytes)
        {
            BytesReceived += Recv;
            RemainingBytes -= Recv;
        }
        else if (Recv == 0)
        {
            NFNN_ERROR();
        }
        else
        {
            NFNN_ERROR();
        }
    }
}

static void
NfNN_Network_SendAll(nfnn_platform_socket Sock, u32 SizeInBytes, char *Src)
{
    int BytesSent = 0;
    int RemainingBytes = SizeInBytes;

    while (RemainingBytes > 0)
    {
        int Sent = send(Sock, Src + BytesSent, RemainingBytes, 0);
        if (Sent > 0 && Sent <= RemainingBytes)
        {
            BytesSent += Sent;
            RemainingBytes -= Sent ;
        }
        else if (Sent == 0)
        {
            NFNN_ERROR();
        }
        else
        {
            NFNN_ERROR();
        }
    }
}

static void
NfNN_Network_RecvGradient(nfnn_memory_arena *Mem, nfnn_platform_socket Sock, nfnn_tensor *T)
{
    NfNN_Network_RecvSize(Sock, NfNN_Size(T), (char*)T->Gradient);
}

static void
NfNN_Network_SendGradient(nfnn_memory_arena *Mem, nfnn_platform_socket Sock, nfnn_tensor *T)
{
    NfNN_Network_SendAll(Sock, NfNN_Size(T), (char*)T->Gradient);
}

static void
NfNN_Network_SendTensor(nfnn_memory_arena *Mem, nfnn_platform_socket Sock, nfnn_tensor *T)
{
    NfNN_Network_SendAll(Sock, NfNN_Size(T), (char*)T->Data);
}

static void
NfNN_Network_RecvTensor(nfnn_memory_arena *Mem, nfnn_platform_socket Sock, nfnn_tensor *T)
{
    NfNN_Network_RecvSize(Sock, NfNN_Size(T), (char*)T->Data);
}

static void
NfNN_ParameterServer_BroadcastWeights(nfnn_memory_arena *Mem, nfnn_parameter_server *Server, nfnn_tensor *T)
{
    for (nfnn_platform_socket Sock = 1; Sock <= Server->MaxSocket; Sock++)
    {
        if (NFNN_FD_ISSET(Sock, Server->Master))
        {
            if (Sock != Server->Socket->Handle)
            {
                NfNN_Network_SendTensor(Mem, Sock, T);
            }
            else
            {
                NFNN_NOT_USED();
            }
        }
        else
        {
            NFNN_NOT_USED();
        }
    }
}

static void
NfNN_Network_RecvAddGradient(nfnn_memory_arena *Mem, nfnn_platform_socket Sock, nfnn_tensor *T)
{
    nfnn_tensor *Temp = NfNN_TensorLike(Mem, T);
    NfNN_Network_RecvGradient(Mem, Sock, Temp);
    NfNN_Math_Add_f32(Temp->Gradient, T->Gradient, NfNN_Length(T), T->Gradient);
}

static void
NfNN_ParameterServer_AwaitGradient(nfnn_memory_arena *Mem, nfnn_parameter_server *Server, nfnn_tensor *T)
{
    for (nfnn_platform_socket Sock = 1; Sock <= Server->MaxSocket; Sock++)
    {
        if (NFNN_FD_ISSET(Sock, Server->Master))
        {
            if (Sock != Server->Socket->Handle)
            {
                nfnn_tensor *Temp = NfNN_TensorLike(Mem, T);
                NfNN_Network_RecvGradient(Mem, Sock, Temp);
                NfNN_Math_Add_f32(Temp->Gradient, T->Gradient, NfNN_Length(T), T->Gradient);
            }
            else
            {
                NFNN_NOT_USED();
            }
        }
        else
        {
            NFNN_NOT_USED();
        }
    }
}

static nfnn_platform_socket 
NfNN_ParameterServer_AwaitForWorker(nfnn_memory_arena *Mem, nfnn_parameter_server *Server)
{
    NfNN_MemoryCopy(Server->Working, Server->Master, sizeof(nfnn_fd_set));

    int SelectResult = select(Server->MaxSocket + 1, Server->Working, 0, 0, 0);

    NFNN_ASSERT(SelectResult >= 0, "select failed");

    for (nfnn_platform_socket Sock = 1; Sock <= Server->MaxSocket; Sock++)
    {
        if (NFNN_FD_ISSET(Sock, Server->Working))
        {
            if (Sock != Server->Socket->Handle)
            {
                return Sock;
            }
            else
            {
                NFNN_NOT_USED();
            }
        }
        else
        {
            NFNN_NOT_USED();
        }
    }
    return -1;
}

static void
NfNN_ParameterServer_WaitWorkers(nfnn_memory_arena *Mem, nfnn_parameter_server *Server, u32 NumberOfWorkers)
{
    u32 ReadyToRead = 0;
    for(;;)
    {
        for (nfnn_platform_socket Sock = 1; Sock <= Server->MaxSocket; Sock++)
        {
            if (NFNN_FD_ISSET(Sock, Server->Master) && Sock != Server->Socket->Handle)
            {
                ReadyToRead++;
            }
        }
        if (ReadyToRead == NumberOfWorkers)
        {
            break;
        }
        else
        {
            ReadyToRead = 0;
            NFNN_PLATFORM_SLEEP(0.01);
        }
    }
}

#endif // NFNN_NETWORK_H
