#ifndef NFNN_AUTOGRAD_H
#define NFNN_AUTOGRAD_H

#include "nfnn_types.h"
#include "nfnn_macro.h"
#include "nfnn_math.h"
#include "nfnn_tensor.h"

static void
NfNN_AutoGrad_PushTensor(nfnn_memory_arena *Mem, nfnn_tensor_list *List, nfnn_tensor *T)
{
    if (List->First == 0)
    {
        List->First = T;
        List->Last = T;
        T->Prev = 0;
        T->Next = 0;
    }
    else
    {
        T->Prev = List->Last;
        T->Next = 0;

        List->Last->Next = T;
        List->Last = T;
    }
}

static void
NfNN_AutoGrad_BuildTensorListRec(nfnn_memory_arena *Mem, nfnn_tensor *T, nfnn_tensor_list *List)
{

    if (T->Visited)
    {
        return;
    }

    T->Visited = true;

    switch(T->Op.Type)
    {
        case NFNN_OP_TYPE_ADD:
        case NFNN_OP_TYPE_MUL:
        case NFNN_OP_TYPE_MATMUL:
        case NFNN_OP_TYPE_SUB:
        case NFNN_OP_TYPE_BROADCAST_ADD:
        case NFNN_OP_TYPE_NLL_LOSS:
        {
            NfNN_AutoGrad_BuildTensorListRec(Mem, T->Op.Binary.Left, List);
            NfNN_AutoGrad_BuildTensorListRec(Mem, T->Op.Binary.Right, List);
        } break;
        case NFNN_OP_TYPE_RELU:
        case NFNN_OP_TYPE_SIGMOID:
        case NFNN_OP_TYPE_TANH:
        case NFNN_OP_TYPE_SQUARE:
        {
            NfNN_AutoGrad_BuildTensorListRec(Mem, T->Op.Unary.Input, List);
        } break;
        case NFNN_OP_TYPE_LOG_SOFTMAX:
        {
            NfNN_AutoGrad_BuildTensorListRec(Mem, T->Op.Dimensional.Input, List);
        } break;
        case NFNN_OP_TYPE_LEAF:
        {
            NFNN_NOT_USED();
        } break;
        case NFNN_OP_TYPE_COPY:
        case NFNN_OP_TYPE_MUL_CONST:
        case NFNN_OP_TYPE_RESHAPE:
        default:
        {
            NFNN_NOT_IMPLEMENTED();
        } break;
    }

    NfNN_AutoGrad_PushTensor(Mem, List, T);
}

static void NfNN_AutoGrad_ReverseList(nfnn_tensor_list *List)
{
    nfnn_tensor *T = List->Last;
    while(T != 0)
    {
        nfnn_tensor *Prev = T->Prev;
        T->Prev = T->Next;
        T->Next = Prev;
        T = Prev;
    }
    nfnn_tensor *Temp = List->Last;
    List->Last = List->First;
    List->First = Temp;
}

// NOTE(luatil): Builds a linked list of tensors in reverse topological order
// according to operation dependencies
static nfnn_tensor_list *NfNN_AutoGrad_BuildList(nfnn_memory_arena *Mem,  nfnn_tensor *T)
{
    nfnn_tensor_list *Result = NfNN_PushStruct(Mem, nfnn_tensor_list); 
    NfNN_AutoGrad_BuildTensorListRec(Mem, T, Result);
    return Result;
}

static void
NfNN_AutoGrad_ZeroGrad(nfnn_memory_arena *Mem, nfnn_tensor *T)
{
    nfnn_tensor_list *List = NfNN_AutoGrad_BuildList(Mem, T);
    for (nfnn_tensor *It = List->First; It != 0; It = It->Next)
    {
        // TODO(luatil): This breaks abstraction barrier
        It->Visited = false;
        memset(It->Gradient, 0, NfNN_Size(It));
    }
}

static void 
NfNN_AutoGrad_Backward(nfnn_memory_arena *Mem,  nfnn_tensor *T)
{
    nfnn_tensor_list *List = NfNN_AutoGrad_BuildList(Mem, T);

    T->Gradient[0] = 1.0f;

    // NOTE(luatil): Traverse in topological order
    for (nfnn_tensor *It = List->Last; It != 0; It = It->Prev)
    {
        It->Visited = false;
        nfnn_op Op = It->Op;
        switch(Op.Type)
        {
            case NFNN_OP_TYPE_NLL_LOSS:
            {
                NfNN_Math_NLLLossD_Mean_f32(It->Gradient, Op.Binary.Left->Data, Op.Binary.Right->Data, Op.Binary.Left->Dimensions.Dimensions[0], Op.Binary.Left->Dimensions.Dimensions[1], Op.Binary.Left->Gradient);
            } break;
            case NFNN_OP_TYPE_LOG_SOFTMAX:
            {
               NfNN_Math_LogSoftmaxD_f32(Mem, It->Gradient, Op.Dimensional.Input->Data, It->Dimensions.Dimensions[0], It->Dimensions.Dimensions[1], Op.Dimensional.Dim, Op.Dimensional.Input->Gradient);
            } break;
            case NFNN_OP_TYPE_SQUARE:
            {
                NfNN_Math_SquareD_f32(It->Gradient, Op.Unary.Input->Data, NfNN_Length(Op.Unary.Input), Op.Unary.Input->Gradient);
            } break;
            case NFNN_OP_TYPE_RELU:
            {
                NfNN_Math_ReLUD_f32(It->Gradient, Op.Unary.Input->Data, NfNN_Length(Op.Unary.Input), Op.Unary.Input->Gradient);
            } break;
            case NFNN_OP_TYPE_TANH:
            {
                NfNN_Math_TanhD_f32(It->Gradient, Op.Unary.Input->Data, NfNN_Length(Op.Unary.Input), Op.Unary.Input->Gradient);
            } break;
            case NFNN_OP_TYPE_ADD:
            {
                NfNN_Math_FmaddConst_f32(It->Gradient, 1.0f, NfNN_Length(Op.Binary.Left), Op.Binary.Left->Gradient ); 
                NfNN_Math_FmaddConst_f32(It->Gradient, 1.0f, NfNN_Length(Op.Binary.Right), Op.Binary.Right->Gradient); 
            } break;
            case NFNN_OP_TYPE_BROADCAST_ADD:
            {
                NfNN_Math_FmaddConst_f32(It->Gradient, 1.0f, NfNN_Length(Op.Binary.Left), Op.Binary.Left->Gradient ); 

                if (Op.Binary.Right->Dimensions.Dimensions[0] == 1 && Op.Binary.Right->Dimensions.Dimensions[1] == 1)
                {
                    NfNN_Math_SumAllAdd_f32(It->Gradient, NfNN_Length(It), Op.Binary.Right->Gradient);
                }
                else if (Op.Binary.Right->Dimensions.Dimensions[0] == 1)
                {

                    NfNN_Math_SumXAdd_f32(It->Gradient, It->Dimensions.Dimensions[0], It->Dimensions.Dimensions[1],
                                          Op.Binary.Right->Dimensions.Dimensions[0], Op.Binary.Right->Dimensions.Dimensions[1],
                                          Op.Binary.Right->Gradient);
                }
                else if (Op.Binary.Right->Dimensions.Dimensions[1] == 1)
                {
                    NfNN_Math_SumYAdd_f32(It->Gradient, It->Dimensions.Dimensions[0], It->Dimensions.Dimensions[1],
                                          Op.Binary.Right->Dimensions.Dimensions[0], Op.Binary.Right->Dimensions.Dimensions[1],
                                          Op.Binary.Right->Gradient);
                }
                else
                {
                    NFNN_ERROR();
                }
            } break;
            case NFNN_OP_TYPE_SUB:
            {
                NfNN_Math_FmaddConst_f32(It->Gradient, 1.0f, NfNN_Length(Op.Binary.Left), Op.Binary.Left->Gradient); 
                NfNN_Math_FmaddConst_f32(It->Gradient, -1.0f, NfNN_Length(Op.Binary.Right), Op.Binary.Right->Gradient); 
            } break;
            case NFNN_OP_TYPE_MUL:
            {
                NfNN_Math_Fmadd_f32(It->Gradient, Op.Binary.Right->Data, NfNN_Length(T), Op.Binary.Left->Gradient);
                NfNN_Math_Fmadd_f32(It->Gradient, Op.Binary.Left->Data, NfNN_Length(T), Op.Binary.Right->Gradient);
            } break;
            case NFNN_OP_TYPE_MATMUL:
            {
                // Example: 
                // A in (3, 10) | B in (10, 4)
                // C = A @ B in (3, 4)
                // dL/dC in (3, 4)
                // dL/dA = dL/dC @ B^T in (3, 10)
                // dL/dC = A^T @ dL/dC in (10, 4)

                // Given total derivative rule we also need to add this to the gradient. Therefore:
                // dL/dA += dL/dC @ B^T in (3, 10)
                // dL/dB += A^T @ dL/dC in (10, 4)
                f32 *A = Op.Binary.Left->Data;
                f32 *dLdA = Op.Binary.Left->Gradient;

                u32 A_DimX = Op.Binary.Left->Dimensions.Dimensions[0];
                u32 A_DimY = Op.Binary.Left->Dimensions.Dimensions[1];

                f32 *B = Op.Binary.Right->Data;
                f32 *dLdB = Op.Binary.Right->Gradient;

                u32 B_DimX = Op.Binary.Right->Dimensions.Dimensions[0];
                u32 B_DimY = Op.Binary.Right->Dimensions.Dimensions[1];
                
                f32 *dLdC = It->Gradient;
                u32 dLdC_DimX = A_DimX;
                u32 dLdC_DimY = B_DimY;

                NfNN_Math_MatmulAddTransposeRight_f32(dLdC, B, dLdC_DimX, B_DimY, B_DimX, dLdA);
                NfNN_Math_MatmulAddTransposeLeft_f32(A, dLdC, A_DimX, A_DimY, dLdC_DimY, dLdB);

            } break;
            case NFNN_OP_TYPE_LEAF:
            {
                NFNN_NOT_USED();
            } break;
            case NFNN_OP_TYPE_COUNT:
            default:
            {
                NFNN_ERROR();
            } break;
        }
    }
}

#endif // NFNN_AUTOGRAD_H
