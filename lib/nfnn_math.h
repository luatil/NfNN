#ifndef NFNN_MATH_H
#define NFNN_MATH_H

#include "nfnn_types.h"
#include "nfnn_macro.h"
#include "nfnn_memory_arena.h"
#include <math.h>

static f32
NfNN_Math_Single_Sqrt_f32(f32 X)
{
    return sqrtf(X);
}

static f32
NfNN_Math_Single_Log_f32(f32 X)
{
    return logf(X);
}

static f32
NfNN_Math_Single_Abs_f32(f32 X)
{
    return fabsf(X);
}

static f32
NfNN_Math_Single_Exp_f32(f32 X)
{
    return expf(X);
}

static f32
NfNN_Math_Single_Max_f32(f32 X, f32 Y)
{
    return X > Y ? X : Y;
}

static f32
NfNN_Math_Single_Sigmoid_f32(f32 X)
{
    return 1.0f / (1.0f + NfNN_Math_Single_Exp_f32(-X));
}

static f32
NfNN_Math_Single_SigmoidD_f32(f32 X)
{
    return NfNN_Math_Single_Sigmoid_f32(X) * (1.0f - NfNN_Math_Single_Sigmoid_f32(X));
}

static f32
NfNN_Math_Single_ReLU_f32(f32 X)
{
    return X > 0.0f ? X : 0.0f;
}

static f32
NfNN_Math_Single_ReLUD_f32(f32 X)
{
    return X > 0.0f ? 1.0f : 0.0f;
}

static f32
NfNN_Math_Single_Tanh_f32(f32 X)
{
    return tanhf(X);
}

static f32
NfNN_Math_Single_TanhD_f32(f32 X)
{
    return 1.0f - NfNN_Math_Single_Tanh_f32(X) * NfNN_Math_Single_Tanh_f32(X);
}

static void
NfNN_Math_Exp_f32(f32 *In, u32 NumberOfElements, f32 *Out)
{
    for (u32 Index = 0; Index < NumberOfElements; ++Index)
    {
        Out[Index] = NfNN_Math_Single_Exp_f32(In[Index]);
    }
}

static void
NfNN_Math_Sigmoid_f32(f32 *In, u32 NumberOfElements, f32 *Out)
{
    for (u32 Index = 0; Index < NumberOfElements; ++Index)
    {
        Out[Index] = NfNN_Math_Single_Sigmoid_f32(In[Index]);
    }
}

static void
NfNN_Math_SigmoidD_f32(f32 *Grad, f32 *In, u32 N, f32 *Out)
{
    for (u32 Index = 0; Index < N; ++Index)
    {
        Out[Index] += Grad[Index] * NfNN_Math_Single_SigmoidD_f32(In[Index]);
    }
}

static void
NfNN_Math_FillConstant_f32(f32 *Data, u32 NumberOfElements, f32 Constant)
{
    for (u32 Index = 0; Index < NumberOfElements; ++Index)
    {
        Data[Index] = Constant;
    }
}

static void
NfNN_Math_LinSpace_f32(f32 *Data, u32 NumberOfElements, f32 Lower, f32 Upper)
{
    f32 StepSize = (Upper - Lower) / (f32)NumberOfElements;
    for (u32 Index = 0; Index < NumberOfElements; ++Index)
    {
        Data[Index] = Lower + StepSize * (f32)Index;
    }
}

static void
NfNN_Math_MultiplyByConstant_f32(f32 *In, u32 NumberOfElements, f32 Constant, f32 *Out)
{
    for (u32 Index = 0; Index < NumberOfElements; ++Index)
    {
        Out[Index] = Constant * In[Index];
    }
}

static void
NfNN_Math_AddByConstant_f32(f32 *In, u32 NumberOfElements, f32 Constant, f32 *Out)
{
    for (u32 Index = 0; Index < NumberOfElements; ++Index)
    {
        Out[Index] = Constant + In[Index];
    }
}

static void
NfNN_Math_Add_f32(f32 *A, f32 *B, u32 NumberOfElements, f32 *Out)
{
    for (u32 Index = 0; Index < NumberOfElements; ++Index)
    {
        Out[Index] = A[Index] + B[Index];
    }
}

static void
NfNN_Math_BroadcastAdd_f32(f32 *A, u32 A_X, u32 A_Y, f32 *B, u32 B_X, u32 B_Y, f32 *Out)
{
    if (A_X == B_X && A_Y == B_Y)
    {
        NfNN_Math_Add_f32(A, B, A_X * A_Y, Out);
    }
    else if (A_Y == B_Y && B_X == 1)
    {
        for (u32 I = 0; I < A_X; I++)
        {
            for (u32 J = 0; J < A_Y; J++)
            {
                Out[I * A_Y + J] = A[I * A_Y + J] + B[J];
            }
        }
    }
    else if (A_X == B_X && B_Y == 1)
    {
        for (u32 I = 0; I < A_X; I++)
        {
            for (u32 J = 0; J < A_Y; J++)
            {
                Out[I * A_Y + J] = A[I * A_Y + J] + B[I];
            }
        }
    }
    else if (B_X == 1 && B_Y == 1)
    {
        for (u32 I = 0; I < A_X; I++)
        {
            for (u32 J = 0; J < A_Y; J++)
            {
                Out[I * A_Y + J] = A[I * A_Y + J] + B[0];
            }
        }
    }
    else
    {
        NFNN_ERROR();
    }
}

static void
NfNN_Math_Sub_f32(f32 *A, f32 *B, u32 NumberOfElements, f32 *Out)
{
    for (u32 Index = 0; Index < NumberOfElements; ++Index)
    {
        Out[Index] = A[Index] - B[Index];
    }
}

static void
NfNN_Math_Hadamard_f32(f32 *A, f32 *B, u32 NumberOfElements, f32 *Out)
{
    for (u32 Index = 0; Index < NumberOfElements; ++Index)
    {
        Out[Index] = A[Index] * B[Index];
    }
}

// Out = Out + Mul * Add
static void
NfNN_Math_Fmadd_f32(f32 *Mul, f32 *Add, u32 NumberOfElements, f32 *Out)
{
    for (u32 Index = 0; Index < NumberOfElements; ++Index)
    {
        Out[Index] += Mul[Index] * Add[Index];
    }
}

// Out = Out + Mul * Add
static void
NfNN_Math_FmaddConst_f32(f32 *Mul, f32 Const, u32 NumberOfElements, f32 *Out)
{
    for (u32 Index = 0; Index < NumberOfElements; ++Index)
    {
        Out[Index] += Mul[Index] * Const;
    }
}

static void
NfNN_Math_MatMul_f32(f32 *A, f32 *B, u32 RowsA, u32 ColumnsA, u32 ColumnsB, f32 *Out)
{
    for (u32 Row = 0; Row < RowsA; ++Row)
    {
        for (u32 Column = 0; Column < ColumnsB; ++Column)
        {
            f32 Sum = 0.0f;
            for (u32 Inner = 0; Inner < ColumnsA; ++Inner)
            {
                Sum += A[Row * ColumnsA + Inner] * B[Inner * ColumnsB + Column];
            }
            Out[Row * ColumnsB + Column] = Sum;
        }
    }
}

static void
NfNN_Math_Transpose_f32(f32 *In, u32 Rows, u32 Columns, f32 *Out)
{
    for (u32 Row = 0; Row < Rows; ++Row)
    {
        for (u32 Column = 0; Column < Columns; ++Column)
        {
            Out[Column * Rows + Row] = In[Row * Columns + Column];
        }
    }
}

// Computes Out = A^T * B
static void
NfNN_Math_MatMulTransposeLeft_f32(f32 *A, f32 *B, u32 RowsA, u32 ColumnsA, u32 RowsB, f32 *Out)
{
    for (u32 Row = 0; Row < RowsA; ++Row)
    {
        for (u32 Column = 0; Column < RowsB; ++Column)
        {
            f32 Sum = 0.0f;
            for (u32 Inner = 0; Inner < ColumnsA; ++Inner)
            {
                Sum += A[Row * ColumnsA + Inner] * B[Column * ColumnsA + Inner];
            }
            Out[Row * RowsB + Column] = Sum;
        }
    }
}

static bool
NfNN_Math_CompareMemory_f32(f32 *A, f32 *B, u32 NumberOfElements, f32 Eps)
{
    for (u32 Index = 0; Index < NumberOfElements; ++Index)
    {
        f32 Diff = NfNN_Math_Single_Abs_f32(A[Index] - B[Index]);
        if (Diff > Eps)
        {
            return false;
        }
    }
    return true;
}

static void // Calculates C += A @ B^T
NfNN_Math_MatmulAddTransposeRight_f32(f32 *A, f32 *B, u32 L, u32 M, u32 R, f32 *C)
{
    // Perform the matrix multiplication and addition C += A * B^T
    for (u32 I = 0; I < L; ++I)
    {
        // Iterate over the rows of A and C
        for (u32 J = 0; J < R; ++J)
        {
            // Iterate over the columns of B^T and C
            f32 Sum = 0;
            for (u32 K = 0; K < M; ++K)
            {                                       // Iterate over the columns of A and rows of B
                Sum += A[I * M + K] * B[J * M + K]; // B[j * M + k] is used instead of B[k * R + j] to utilize B^T
            }
            // Add the computed sum to the C matrix
            C[I * R + J] += Sum;
        }
    }
}

static void // Calculates C += A^T @ B
NfNN_Math_MatmulAddTransposeLeft_f32(f32 *A, f32 *B, u32 L, u32 M, u32 R, f32 *C)
{
    // Perform the matrix multiplication and addition C += A^T * B
    for (u32 I = 0; I < M; ++I)
    {
        // Iterate over the rows of A^T and C
        for (u32 J = 0; J < R; ++J)
        {
            // Iterate over the columns of B and C
            f32 Sum = 0;
            for (u32 K = 0; K < L; ++K)
            {                                       // Iterate over the columns of A and rows of B
                Sum += A[K * M + I] * B[K * R + J]; // A[k * M + i] is used to utilize A^T
            }
            // Add the computed sum to the C matrix
            C[I * R + J] += Sum;
        }
    }
}

static void
NfNN_Math_Tanh_f32(f32 *A, u32 N, f32 *Out)
{
    for (u32 Index = 0; Index < N; ++Index)
    {
        Out[Index] = NfNN_Math_Single_Tanh_f32(A[Index]);
    }
}

static void
NfNN_Math_TanhD_f32(f32 *Grad, f32 *In, u32 N, f32 *Out)
{
    for (u32 Index = 0; Index < N; ++Index)
    {
        Out[Index] += Grad[Index] * NfNN_Math_Single_Tanh_f32(In[Index]);
    }
}

static void
NfNN_Math_ReLU_f32(f32 *A, u32 N, f32 *Out)
{
    for (u32 Index = 0; Index < N; ++Index)
    {
        Out[Index] = NfNN_Math_Single_ReLU_f32(A[Index]);
    }
}

static void
NfNN_Math_ReLUD_f32(f32 *Grad, f32 *In, u32 N, f32 *Out)
{
    for (u32 Index = 0; Index < N; ++Index)
    {
        Out[Index] += Grad[Index] * NfNN_Math_Single_ReLUD_f32(In[Index]);
    }
}

static void
NfNN_Math_Square_f32(f32 *In, u32 N, f32 *Out)
{
    for (u32 Index = 0; Index < N; ++Index)
    {
        Out[Index] = In[Index] * In[Index];
    }
}

static void
NfNN_Math_SquareD_f32(f32 *Grad, f32 *In, u32 N, f32 *Out)
{
    for (u32 Index = 0; Index < N; ++Index)
    {
        Out[Index] += Grad[Index] * 2 * In[Index];
    }
}

static void
NfNN_Math_SumAllAdd_f32(f32 *Grad, u32 N, f32 *Out)
{
    for (u32 Index = 0; Index < N; ++Index)
    {
        Out[0] += Grad[Index];
    }
}

static void
NfNN_Math_SumXAdd_f32(f32 *In, u32 In_X, u32 In_Y, u32 Out_X, u32 Out_Y, f32 *Out)
{

    NFNN_ASSERT(Out_X == 1, "In_X must be equal to Out_X");
    NFNN_ASSERT(In_Y == Out_Y, "In_X must be equal to Out_X");

    /**
     * [[2.0, 3.0, 4.0],[5.0, 6.0, 7.0]] -> [7.0, 9.0, 11.0]
     */
    for (u32 I = 0; I < In_X; I++)
    {
        for (u32 J = 0; J < In_Y; J++)
        {
            Out[J] += In[I * In_Y + J];
        }
    }
}

static void
NfNN_Math_SumYAdd_f32(f32 *In, u32 In_X, u32 In_Y, u32 Out_X, u32 Out_Y, f32 *Out)
{
    NFNN_ASSERT(Out_Y == 1, "In_X must be equal to Out_X");
    NFNN_ASSERT(In_X == Out_X, "In_X must be equal to Out_X");
    /**
     * [[2.0, 3.0, 4.0],[5.0, 6.0, 7.0]] -> [9., 18.0]
     */
    for (u32 I = 0; I < In_X; I++)
    {
        for (u32 J = 0; J < In_Y; J++)
        {
            // TODO(luatil)
            Out[I] += In[I * In_Y + J];
        }
    }
}

static void
NfNN_Math_Log_f32(f32 *In, u32 N, f32 *Out)
{
    for (u32 Index = 0; Index < N; ++Index)
    {
        Out[Index] = NfNN_Math_Single_Log_f32(In[Index] + NFNN_EPS_FOR_LOG);
    }
}

static void
NfNN_Math_SoftMax_f32(f32 *In, u32 X, u32 Y, u32 Dim, f32 *Out)
{

    // NOTE(luatil): For numerical stability subtract the maximum value

    if (Dim == 0)
    {
        for (u32 I = 0; I < Y; I++)
        {
            f32 MaxX = NFNN_MINUS_INF_F32;
            for (u32 J = 0; J < X; J++)
            {
                MaxX = NfNN_Math_Single_Max_f32(MaxX, In[J * Y + I]);
            }
            f32 Sum = 0.0f;
            for (u32 J = 0; J < X; J++)
            {
                Sum += NfNN_Math_Single_Exp_f32(In[I * X + J] - MaxX);
            }
            for (u32 J = 0; J < X; J++)
            {
                Out[I * X + J] = NfNN_Math_Single_Exp_f32(In[I * X + J] - MaxX) / Sum;
            }
        }
    }
    else if (Dim == 1)
    {
        for (u32 I = 0; I < X; I++)
        {
            f32 MaxY = 0;
            for (u32 J = 0; J < Y; J++)
            {
                MaxY = NfNN_Math_Single_Max_f32(MaxY, In[I * Y + J]);
            }
            f32 Sum = 0.0f;
            for (u32 J = 0; J < Y; J++)
            {
                Sum += NfNN_Math_Single_Exp_f32(In[I * Y + J] - MaxY);
            }
            for (u32 J = 0; J < Y; J++)
            {
                Out[I * Y + J] = NfNN_Math_Single_Exp_f32(In[I * Y + J] - MaxY) / Sum;
            }
        }
    }
    else
    {
        NFNN_ERROR();
    }
}

static void
NfNN_Math_LogSoftmax_f32(f32 *In, u32 X, u32 Y, u32 Dim, f32 *Out)
{
    NfNN_Math_SoftMax_f32(In, X, Y, Dim, Out);
    NfNN_Math_Log_f32(Out, X * Y, Out);
}

static void
NfNN_Math_PrintArray(f32 *In, u32 X, u32 Y)
{
    for (u32 I = 0; I < X; I++)
    {
        printf("[");
        for (u32 J = 0; J < Y; J++)
        {
            if (J != (Y - 1))
            {
                printf("%2.f, ", In[I * Y + J]);
            }
            else
            {
                printf("%2.f]\n", In[I * Y + J]);
            }
        }
    }
}

static void
NfNN_Math_LogSoftmaxD_f32(nfnn_memory_arena *Mem, f32 *Grad, f32 *X, u32 X_Dim, u32 Y_Dim, u32 Dim, f32 *Out)
{
    f32 *Softmax = NfNN_PushArray(Mem, f32, X_Dim * Y_Dim);

    NfNN_Math_SoftMax_f32(X, X_Dim, Y_Dim, Dim, Softmax);

    if (Dim == 1)
    {
        for (u32 I = 0; I < X_Dim; I++)
        {
            for (u32 J = 0; J < Y_Dim; J++)
            {
                f32 Sum = 0.0;
                for (u32 K = 0; K < Y_Dim; K++)
                {
                    // NOTE(luatil): K==J? Maybe too clever
                    Sum += Grad[I * Y_Dim + K] * ((K == J) * 1.0f - Softmax[I * Y_Dim + J]);
                }
                Out[I * Y_Dim + J] += Sum;
            }
        }
    }
    else if (Dim == 0)
    {
        NFNN_NOT_IMPLEMENTED();
        for (u32 I = 0; I < X_Dim; I++)
        {
            for (u32 J = 0; J < Y_Dim; J++)
            {
                f32 Sum = 0.0;
                for (u32 K = 0; K < X_Dim; K++)
                {
                    // NOTE(luatil): K==J? Maybe too clever
                    Sum += Grad[K * Y_Dim + J] * ((K == I) * 1.0f - Softmax[I * Y_Dim + J]);
                }
                Out[I * Y_Dim + J] += Sum;
            }
        }
    }
    else
    {
        NFNN_ERROR();
    }
}

static void
NfNN_Math_NLLLoss_Mean_f32(f32 *A, f32 *Indexes, u32 X, u32 Y, f32 *Out)
{
    // NOTE(luatil): Might also want to handle different types of reductions
    // see Pytorch's documentation on reduction=mean | sum | none
    // This is reduction=mean
    for (u32 I = 0; I < X; I++)
    {
        Out[0] += -A[I * Y + (u32)Indexes[I]];
    }
    Out[0] /= (f32)X;
}

static void
NfNN_Math_NLLLossD_Mean_f32(f32 *Grad, f32 *X, f32 *Indexes, u32 X_Dim, u32 Y_Dim, f32 *Out)
{
    for (u32 I = 0; I < X_Dim; I++)
    {
        for (u32 J = 0; J < Y_Dim; J++)
        {
            Out[I * Y_Dim + J] += -((u32)Indexes[I] == J) * Grad[0] / (f32)X_Dim;
        }
    }
}

static u32
NfNN_Math_BigEndianToLittleEndian_u32(u32 X)
{
    u32 Byte1 = X & 0xFF000000;
    u32 Byte2 = X & 0x00FF0000;
    u32 Byte3 = X & 0x0000FF00;
    u32 Byte4 = X & 0x000000FF;

    return (Byte4 << 24) | (Byte3 << 8) | (Byte2 >> 8) | (Byte1 >> 24);
}

static void
NfNN_Math_Argmax(f32 *In, u32 X, u32 Y, u32 Dim, f32 *Out)
{
    if(Dim == 1)
    {
        for(u32 I = 0; I < X; I++)
        {
            f32 Max = In[I * Y];
            u32 Index = 0;
            for(u32 J = 1; J < Y; J++)
            {
                if(In[I * Y + J] > Max)
                {
                    Max = In[I * Y + J];
                    Index = J;
                }
            }
            Out[I] = (f32)Index;
        }
    }
    else if(Dim == 0)
    {
        NFNN_NOT_IMPLEMENTED();
    }
    else
    {
        NFNN_ERROR();
    }
}

static void
NfNN_Math_Close_f32(f32 *A, f32 *B, u32 N, f32 *Out, f32 Eps)
{
    for(u32 I = 0; I < N; I++)
    {
        Out[I] = (NfNN_Math_Single_Abs_f32(A[I] - B[I]) < Eps) * 1.0f;
    }
}

static void
NfNN_Math_Zero_f32(f32 *A, u32 N)
{
    for(u32 I = 0; I < N; I++)
    {
        A[I] = 0.0f;
    }
}

#endif // NFNN_MATH_H
