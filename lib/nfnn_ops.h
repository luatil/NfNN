#ifndef NFNN_OPS_H
#define NFNN_OPS_H

#include "nfnn_math.h"
#include "nfnn_memory_arena.h"
#include "nfnn_random.h"
#include "nfnn_tensor.h"
#include "nfnn_types.h"

static nfnn_op NfNN_Op_Binary(nfnn_op_type Type, nfnn_tensor *Left, nfnn_tensor *Right)
{
    nfnn_op Result = {0};
    Result.Type = Type;
    Result.Binary.Left = Left;
    Result.Binary.Right = Right;
    return Result;
}

static nfnn_op NfNN_Op_Dimensional(nfnn_op_type Type, nfnn_tensor *Input, u32 Dim)
{
    nfnn_op Result = {0};
    Result.Type = Type;
    Result.Dimensional.Input = Input;
    Result.Dimensional.Dim = Dim;
    return Result;
}

static nfnn_op NfNN_Op_Unary(nfnn_op_type Type, nfnn_tensor *Input)
{
    nfnn_op Result = {0};
    Result.Type = Type;
    Result.Unary.Input = Input;
    return Result;
}

static f32 NfNN_Item(nfnn_tensor *T)
{
    NFNN_ASSERT(T->Dimensions.Dimensions[0] == 1 && T->Dimensions.Dimensions[1] == 1,
                "NfNN_Item: Tensor is not a scalar");
    return T->Data[0];
}

static void NfNN_Update(nfnn_tensor *T, f32 LearningRate)
{
    // TODO(luatil): This should be a backend function
    if (T->RequiresGrad)
    {
        for (u32 I = 0; I < NfNN_Length(T); I++)
        {
            T->Data[I] -= LearningRate * T->Gradient[I];
        }
    }
}

static nfnn_tensor *NfNN_Select(nfnn_memory_arena *Mem, nfnn_tensor *X, u32 *Indexes, u32 N)
{
    nfnn_tensor *Result = NfNN_CreateTensor(Mem, NfNN_Dim2(N, X->Dimensions.Dimensions[1]), false);

    Result->Op.Type = NFNN_OP_TYPE_LEAF;

    // NOTE(luatil): Don't know how to do this in a better way

    for (u32 I = 0; I < N; I++)
    {
        for (u32 J = 0; J < X->Dimensions.Dimensions[1]; J++)
        {
            Result->Data[I * X->Dimensions.Dimensions[1] + J] = X->Data[Indexes[I] * X->Dimensions.Dimensions[1] + J];
        }
    }

    return Result;
}

static nfnn_tensor *NfNN_Matrix(nfnn_memory_arena *Mem, nfnn_random_state *Random, u32 InputSize, u32 OutputSize)
{
    nfnn_tensor *Result = NfNN_PushStruct(Mem, nfnn_tensor);
    Result->Dimensions = NfNN_Dim2(InputSize, OutputSize);
    Result->Data = NfNN_PushArray(Mem, f32, InputSize * OutputSize);
    Result->Gradient = NfNN_PushArray(Mem, f32, InputSize * OutputSize);
    Result->RequiresGrad = true;
    Result->Visited = false;
    Result->Op.Type = NFNN_OP_TYPE_LEAF;

    // TODO(luatil): Initialize weights
    NfNN_Random_UniformArrayInRange_f32(Random, Result->Data, NfNN_Length(Result), -1.0f, 1.0f);

    return Result;
}

static nfnn_tensor *NfNN_Copy(nfnn_memory_arena *Mem, nfnn_tensor *X)
{
    nfnn_tensor *Result = NfNN_TensorLike(Mem, X);

    Result->Op.Type = NFNN_OP_TYPE_COPY;
    Result->Op.Unary.Input = X;

    // Operation
    NfNN_MemoryCopy(Result->Data, X->Data, NfNN_Size(X));

    return Result;
}

static nfnn_tensor *NfNN_Sigmoid(nfnn_memory_arena *Mem, nfnn_tensor *X)
{
    nfnn_tensor *Result = NfNN_TensorLike(Mem, X);

    Result->Op.Type = NFNN_OP_TYPE_SIGMOID;
    Result->Op.Unary.Input = X;

    // Operation
    NfNN_Math_Sigmoid_f32(X->Data, NfNN_Length(X), Result->Data);

    return Result;
}

static nfnn_tensor *NfNN_Const(nfnn_memory_arena *Mem, nfnn_dim Dim, f32 Const)
{
    nfnn_tensor *Result = NfNN_CreateTensor(Mem, Dim, false);

    Result->Op.Type = NFNN_OP_TYPE_LEAF;
    Result->RequiresGrad = false;

    NfNN_Math_FillConstant_f32(Result->Data, NfNN_Length(Result), Const);

    return Result;
}

static nfnn_tensor *NfNN_LinSpace(nfnn_memory_arena *Mem, nfnn_dim Dim, f32 Lower, f32 Upper)
{
    nfnn_tensor *Result = NfNN_CreateTensor(Mem, Dim, false);
    Result->Op.Type = NFNN_OP_TYPE_LEAF;
    NfNN_Math_LinSpace_f32(Result->Data, NfNN_Length(Result), Lower, Upper);
    return Result;
}

static nfnn_tensor *NfNN_Ones(nfnn_memory_arena *Mem, nfnn_dim Dim)
{
    return NfNN_Const(Mem, Dim, 1.0f);
}

static nfnn_tensor *NfNN_MultiplyByConstant(nfnn_memory_arena *Mem, nfnn_tensor *X, f32 Constant)
{
    nfnn_tensor *Result = NfNN_TensorLike(Mem, X);

    Result->Op.Type = NFNN_OP_TYPE_MUL_CONST;
    Result->Op.Constant.ConstantInputf32 = Constant;

    NfNN_Math_MultiplyByConstant_f32(X->Data, NfNN_Length(Result), Constant, Result->Data);

    return Result;
}

static nfnn_tensor *NfNN_Add(nfnn_memory_arena *Mem, nfnn_tensor *X, nfnn_tensor *Y)
{

    bool EqualDimensions = NfNN_Dim_Equal(X->Dimensions, Y->Dimensions);
    bool Broadcastable = NfNN_Dim_Broadcastable(X->Dimensions, Y->Dimensions);

    NFNN_ASSERT((EqualDimensions || Broadcastable), "NfNN_Add: Dimensions must be equal or broadcastable");

    nfnn_tensor *Result = NfNN_TensorLike(Mem, X);

    if (EqualDimensions)
    {
        Result->Op = NfNN_Op_Binary(NFNN_OP_TYPE_ADD, X, Y);
        NfNN_Math_Add_f32(X->Data, Y->Data, NfNN_Length(X), Result->Data);
    }
    else if (Broadcastable)
    {
        Result->Op = NfNN_Op_Binary(NFNN_OP_TYPE_BROADCAST_ADD, X, Y);
        NfNN_Math_BroadcastAdd_f32(X->Data, X->Dimensions.Dimensions[0], X->Dimensions.Dimensions[1], Y->Data,
                                   Y->Dimensions.Dimensions[0], Y->Dimensions.Dimensions[1], Result->Data);
    }
    else
    {
        NFNN_ERROR();
    }

    return Result;
}

static nfnn_tensor *NfNN_Sub(nfnn_memory_arena *Mem, nfnn_tensor *X, nfnn_tensor *Y)
{
    nfnn_tensor *Result = NfNN_TensorLike(Mem, X);

    Result->Op.Type = NFNN_OP_TYPE_SUB;
    Result->Op.Binary.Left = X;
    Result->Op.Binary.Right = Y;

    NfNN_Math_Sub_f32(X->Data, Y->Data, NfNN_Length(X), Result->Data);

    return Result;
}

static nfnn_tensor *NfNN_Mul(nfnn_memory_arena *Mem, nfnn_tensor *X, nfnn_tensor *Y)
{
    nfnn_tensor *Result = NfNN_TensorLike(Mem, X);

    Result->Op.Type = NFNN_OP_TYPE_MUL;
    Result->Op.Binary.Left = X;
    Result->Op.Binary.Right = Y;

    NfNN_Math_Add_f32(X->Data, Y->Data, NfNN_Length(X), Result->Data);
    NfNN_Math_Hadamard_f32(X->Data, Y->Data, NfNN_Length(X), Result->Data);

    return Result;
}

static bool NfNN_AllClose(nfnn_tensor *X, nfnn_tensor *Y, f32 Episilon)
{
    u32 NumberOfElements = NfNN_Length(X);
    for (u32 I = 0; I < NumberOfElements; I++)
    {
        f32 Diff = NfNN_Math_Single_Abs_f32(X->Data[I] - Y->Data[I]);
        if (Diff > Episilon)
        {
            return false;
        }
    }
    return true;
}

static nfnn_tensor *NfNN_MatMul(nfnn_memory_arena *Mem, nfnn_tensor *X, nfnn_tensor *Y)
{
    nfnn_tensor *Result =
        NfNN_CreateTensor(Mem, NfNN_Dim2(X->Dimensions.Dimensions[0], Y->Dimensions.Dimensions[1]), true);

    Result->Op.Type = NFNN_OP_TYPE_MATMUL;
    Result->Op.Binary.Left = X;
    Result->Op.Binary.Right = Y;

    NfNN_Math_MatMul_f32(X->Data, Y->Data, X->Dimensions.Dimensions[0], X->Dimensions.Dimensions[1],
                         Y->Dimensions.Dimensions[1], Result->Data);

    return Result;
}

static nfnn_tensor *NfNN_Reshape(nfnn_memory_arena *Mem, nfnn_tensor *X, nfnn_dim Dim)
{
    // NOTE(luatil): This is inneficient
    nfnn_tensor *Result = NfNN_Copy(Mem, X);
    Result->Dimensions = Dim;
    Result->Op.Type = NFNN_OP_TYPE_RESHAPE;
    Result->Op.Unary.Input = X;
    return Result;
}

static nfnn_tensor *NfNN_Sum(nfnn_memory_arena *Mem, nfnn_tensor *T, u32 Axis)
{
    nfnn_tensor *Result = 0;
    if (Axis == 0)
    {
        nfnn_tensor *Temp = NfNN_Ones(Mem, NfNN_Dim2(1, T->Dimensions.Dimensions[0]));
        Result = NfNN_MatMul(Mem, Temp, T);
    }
    else if (Axis == 1)
    {
        nfnn_tensor *Temp = NfNN_Ones(Mem, NfNN_Dim2(T->Dimensions.Dimensions[1], 1));
        Result = NfNN_MatMul(Mem, T, Temp);
    }
    return Result;
}

static nfnn_tensor *NfNN_SumAll(nfnn_memory_arena *Mem, nfnn_tensor *T)
{
    nfnn_tensor *Result = NfNN_Sum(Mem, NfNN_Sum(Mem, T, 0), 1);
    return Result;
}

static nfnn_tensor *NfNN_From_f32(nfnn_memory_arena *Mem, f32 *T, nfnn_dim Dim)
{
    nfnn_tensor *Result = NfNN_CreateTensor(Mem, Dim, true);
    Result->Op.Type = NFNN_OP_TYPE_LEAF;
    NfNN_MemoryCopy(Result->Data, T, NfNN_Size(Result));
    return Result;
}

static nfnn_tensor *NfNN_ReLU(nfnn_memory_arena *Mem, nfnn_tensor *T)
{
    nfnn_tensor *Result = NfNN_TensorLike(Mem, T);

    Result->Op.Type = NFNN_OP_TYPE_RELU;
    Result->Op.Unary.Input = T;

    NfNN_Math_ReLU_f32(T->Data, NfNN_Length(T), Result->Data);

    return Result;
}

static nfnn_tensor *NfNN_Tanh(nfnn_memory_arena *Mem, nfnn_tensor *T)
{
    nfnn_tensor *Result = NfNN_TensorLike(Mem, T);

    Result->Op.Type = NFNN_OP_TYPE_TANH;
    Result->Op.Unary.Input = T;

    NfNN_Math_Tanh_f32(T->Data, NfNN_Length(T), Result->Data);

    return Result;
}

static nfnn_tensor *NfNN_Square(nfnn_memory_arena *Mem, nfnn_tensor *T)
{
    nfnn_tensor *Result = NfNN_TensorLike(Mem, T);

    Result->Op.Type = NFNN_OP_TYPE_SQUARE;
    Result->Op.Unary.Input = T;

    NfNN_Math_Square_f32(T->Data, NfNN_Length(T), Result->Data);

    return Result;
}

static nfnn_tensor *NfNN_LogSoftmax(nfnn_memory_arena *Mem, nfnn_tensor *T, u32 Dim)
{
    nfnn_tensor *Result = NfNN_TensorLike(Mem, T);
    Result->Op = NfNN_Op_Dimensional(NFNN_OP_TYPE_LOG_SOFTMAX, T, Dim);
    NfNN_Math_LogSoftmax_f32(T->Data, T->Dimensions.Dimensions[0], T->Dimensions.Dimensions[1], Dim, Result->Data);
    return Result;
}

static nfnn_tensor *NfNN_NLLLoss(nfnn_memory_arena *Mem, nfnn_tensor *T, nfnn_tensor *Indexes)
{
    nfnn_tensor *Result = NfNN_CreateTensor(Mem, NfNN_Dim2(1, 1), true);
    Result->Op = NfNN_Op_Binary(NFNN_OP_TYPE_NLL_LOSS, T, Indexes);
    NfNN_Math_NLLLoss_Mean_f32(T->Data, Indexes->Data, T->Dimensions.Dimensions[0], T->Dimensions.Dimensions[1],
                               Result->Data);
    return Result;
}

static nfnn_tensor *NfNN_MSELoss(nfnn_memory_arena *Mem, nfnn_tensor *X, nfnn_tensor *Y)
{
    nfnn_tensor *Diff = NfNN_Sub(Mem, X, Y);
    nfnn_tensor *Square = NfNN_Square(Mem, Diff);
    nfnn_tensor *Loss = NfNN_SumAll(Mem, Square);

    u32 BatchSize = X->Dimensions.Dimensions[0];
    f32 InvBatchSize = 0.5f / (f32)BatchSize;

    nfnn_tensor *Result = NfNN_Mul(Mem, Loss, NfNN_Const(Mem, NfNN_Dim2(1, 1), InvBatchSize));
    return Result;
}

static nfnn_tensor *NfNN_Argmax(nfnn_memory_arena *Mem, nfnn_tensor *T, u32 Dim)
{
    nfnn_tensor *Result = 0;
    if (Dim == 1)
    {
        Result = NfNN_CreateTensor(Mem, NfNN_Dim2(T->Dimensions.Dimensions[0], 1), false);
        NfNN_Math_Argmax(T->Data, T->Dimensions.Dimensions[0], T->Dimensions.Dimensions[1], Dim, Result->Data);
    }
    else if (Dim == 0)
    {
        NFNN_NOT_IMPLEMENTED();
    }
    else
    {
        NFNN_ERROR();
    }
    return Result;
}

static nfnn_tensor *NfNN_Max(nfnn_memory_arena *Mem, nfnn_tensor *T, u32 Dim)
{
    nfnn_tensor *Result = NfNN_TensorLike(Mem, T);
    NFNN_NOT_IMPLEMENTED();
    return Result;
}

static nfnn_tensor *NfNN_Equal(nfnn_memory_arena *Mem, nfnn_tensor *X, nfnn_tensor *Y)
{
    nfnn_tensor *Result = NfNN_TensorLike(Mem, X);
    NfNN_Math_Close_f32(X->Data, Y->Data, NfNN_Length(X), Result->Data, NFNN_EPS_FOR_EQUAL);
    return Result;
}

#endif // NFNN_OPS_H
