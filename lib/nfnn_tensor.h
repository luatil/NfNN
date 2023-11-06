#ifndef NFNN_TENSOR_H
#define NFNN_TENSOR_H

#include "nfnn_types.h"
#include "nfnn_memory_arena.h"

#define NFNN_MAX_DIMENSIONS 2
typedef struct nfnn_dim nfnn_dim;
struct nfnn_dim
{
    u32 Dimensions[NFNN_MAX_DIMENSIONS];
    u32 UsedDimensions;
};

static nfnn_dim NfNN_Dim2(u32 D0, u32 D1)
{
    nfnn_dim Result = {0};
    Result.Dimensions[0] = D0;
    Result.Dimensions[1] = D1;
    Result.UsedDimensions = 2;
    return Result;
}

static u32 NfNN_DimSize(nfnn_dim Dim)
{
    return Dim.Dimensions[0] * Dim.Dimensions[1];
}

static bool NfNN_Dim_Equal(nfnn_dim A, nfnn_dim B)
{
    bool Result = true;
    if (A.UsedDimensions != B.UsedDimensions)
    {
        Result = false;
    }
    else
    {
        for (u32 DimIndex = 0; DimIndex < A.UsedDimensions; DimIndex++)
        {
            if (A.Dimensions[DimIndex] != B.Dimensions[DimIndex])
            {
                Result = false;
                break;
            }
        }
    }
    return Result;
}

static bool NfNN_Dim_Broadcastable(nfnn_dim A, nfnn_dim B)
{
    /**
     * Copied from pytorch guide
     * General semantics
     * Two tensors are “broadcastable” if the following rules hold:
     *  - Each tensor has at least one dimension.
     *  - When iterating over the dimension sizes, 
     *    starting at the trailing dimension, the dimension 
     *    sizes must either be equal, one of them is 1, or one of 
     *    them does not exist.
     * 
     * Examples:
     * 
     * (1)
     * A      (2d tensor):  5 x 4
     * B      (2d tensor):  5 x 1
     * A + B is broadcastable
     * 
     * (2)
     * A      (2d tensor):  5 x 4
     * B      (2d tensor):  3 x 1
     * A + B is not broadcastable
     * 
     * (3)
     * A      (2d tensor):  5 x 4
     * B      (2d tensor):  1 x 4
     * A + B is broadcastable
     * 
     **/
    
    bool Result = true;
    // NOTE(luatil): Hardcoded for 2 dimension tensors it for now on 2
    // NOTE(luatil): Only supports broadcasting the B tensor

    if (A.UsedDimensions != 2 || B.UsedDimensions != 2)
    {
        NFNN_NOT_IMPLEMENTED();
    }

    if (A.UsedDimensions != B.UsedDimensions)
    {
        Result = false;
    }
    else
    {
        for (u32 DimIndex = 0; DimIndex < A.UsedDimensions; DimIndex++)
        {
            if (A.Dimensions[DimIndex] != B.Dimensions[DimIndex] &&
                B.Dimensions[DimIndex] != 1)
            {
                Result = false;
                break;
            }
        }
    }

    return Result;
}

typedef enum nfnn_op_type nfnn_op_type;
enum nfnn_op_type
{
    NFNN_OP_TYPE_LEAF,
    NFNN_OP_TYPE_ADD,
    NFNN_OP_TYPE_BROADCAST_ADD,
    NFNN_OP_TYPE_SUB,
    NFNN_OP_TYPE_MUL,
    NFNN_OP_TYPE_MATMUL,
    NFNN_OP_TYPE_COPY,
    NFNN_OP_TYPE_SIGMOID,
    NFNN_OP_TYPE_RELU,
    NFNN_OP_TYPE_SQUARE,
    NFNN_OP_TYPE_LOG_SOFTMAX,
    NFNN_OP_TYPE_NLL_LOSS,
    NFNN_OP_TYPE_TANH,
    NFNN_OP_TYPE_MUL_CONST,
    NFNN_OP_TYPE_RESHAPE,
    NFNN_OP_TYPE_COUNT
};

typedef struct nfnn_tensor nfnn_tensor;
struct nfnn_tensor;

#define NFNN_MAX_INPUTS 2
typedef struct nfnn_op nfnn_op;
struct nfnn_op
{
    nfnn_op_type Type;
    union {
        nfnn_tensor *Inputs[NFNN_MAX_INPUTS];
        struct {
            nfnn_tensor *Left;
            nfnn_tensor *Right;
        } Binary;
        struct {
            nfnn_tensor *Input;
            u32 Dim;
        } Dimensional;
        struct {
            nfnn_tensor *Input;
        } Unary;
        struct {
            f32 ConstantInputf32;
        } Constant;
    };
    nfnn_op *Next;
    nfnn_op *Prev;
};

typedef struct nfnn_op_list nfnn_op_list;
struct nfnn_op_list
{
    nfnn_op *First;
    nfnn_op *Last;
};

struct nfnn_tensor
{
    nfnn_dim Dimensions;
    f32 *Data;
    f32 *Gradient; 
    bool RequiresGrad;
    bool Visited;
    nfnn_op Op;
    nfnn_tensor *Next;
    nfnn_tensor *Prev;
};

typedef struct nfnn_tensor_list nfnn_tensor_list;
struct nfnn_tensor_list
{
    nfnn_tensor *First;
    nfnn_tensor *Last;
};

static u32 NfNN_Length(nfnn_tensor *T)
{
    return NfNN_DimSize(T->Dimensions);
}

static u32 NfNN_Size(nfnn_tensor *X)
{
    u32 Result = 0;

    Result = NfNN_DimSize(X->Dimensions) * sizeof(f32);

    return Result;
}

static nfnn_tensor *NfNN_CreateTensor(nfnn_memory_arena *Mem, nfnn_dim Dim, bool RequiresGrad)
{
    nfnn_tensor *Result = NfNN_PushStruct(Mem, nfnn_tensor);

    Result->Dimensions = Dim;
    Result->Data = NfNN_PushTensor(Mem, Dim);
    Result->Gradient = NfNN_PushTensor(Mem, Dim);

    Result->RequiresGrad = RequiresGrad;
    Result->Visited = false;

    Result->Next = 0;
    Result->Prev = 0;

    return Result;
}

static nfnn_tensor *NfNN_TensorLike(nfnn_memory_arena *Mem, nfnn_tensor *X)
{
    nfnn_tensor *Result = NfNN_CreateTensor(Mem, X->Dimensions, X->RequiresGrad);
    return Result;
}

static nfnn_tensor *NfNN_ZeroesLike(nfnn_memory_arena *Mem, nfnn_tensor *X)
{
    nfnn_tensor *Result = NfNN_CreateTensor(Mem, X->Dimensions, X->RequiresGrad);
    return Result;
}


static void NfNN_Print_(nfnn_tensor *T)
{
    for (u32 Row = 0; Row < T->Dimensions.Dimensions[0]; Row++)
    {
        for (u32 Col = 0; Col < T->Dimensions.Dimensions[1]; Col++)
        {
            printf("%.4f ", T->Data[Row * T->Dimensions.Dimensions[1] + Col]);
        }
        printf("\n");
    }
}

static void NfNN_PrintGrad_(nfnn_tensor *T)
{
    for (u32 Row = 0; Row < T->Dimensions.Dimensions[0]; Row++)
    {
        for (u32 Col = 0; Col < T->Dimensions.Dimensions[1]; Col++)
        {
            printf("%.4f ", T->Gradient[Row * T->Dimensions.Dimensions[1] + Col]);
        }
        printf("\n");
    }
}

#define NfNN_Print(_T) printf("%s:\n", #_T); NfNN_Print_(_T); printf("\n");
#define NfNN_PrintGrad(_T) printf("d%s:\n", #_T); NfNN_PrintGrad_(_T); printf("\n");


#endif // NFNN_TENSOR_H
