#ifndef NFNN_OPTIMIZER_H
#define NFNN_OPTIMIZER_H

#include "nfnn_tensor.h"

typedef enum nfnn_optimizer_type nfnn_optimizer_type;
enum nfnn_optimizer_type
{
    NFNN_OPTIMIZER_SGD,
    NFNN_OPTIMIZER_ADAM,
};

typedef struct nfnn_optimizer_sgd_param nfnn_optimizer_sgd_param;
struct nfnn_optimizer_sgd_param
{
    nfnn_tensor *Tensor;
    nfnn_optimizer_sgd_param *Next;
};

typedef struct nfnn_optimizer_sgd nfnn_optimizer_sgd;
struct nfnn_optimizer_sgd
{
    nfnn_optimizer_sgd_param *First;
    nfnn_optimizer_sgd_param *Last;
};

typedef struct nfnn_optimizer nfnn_optimizer;
struct nfnn_optimizer
{
    nfnn_optimizer_type Type;
    u32 Iteration;
    f32 LearningRate;
    u32 NumberOfWorkers;
    union {
        nfnn_optimizer_sgd SGD;
    };
};

static nfnn_optimizer *NfNN_Optimizer_SGD(nfnn_memory_arena *Mem, f32 LearningRate, u32 NumberOfWorkers)
{
    nfnn_optimizer *Result = NfNN_PushStruct(Mem, nfnn_optimizer);
    Result->Type = NFNN_OPTIMIZER_SGD;
    Result->LearningRate = LearningRate;
    Result->SGD.First = 0;
    Result->SGD.Last = 0;
    Result->NumberOfWorkers = NumberOfWorkers;
    return Result;
}

static void NfNN_Optimizer_AddParam(nfnn_memory_arena *Mem, nfnn_optimizer *Optimizer, nfnn_tensor *T)
{
    nfnn_optimizer_sgd_param *Param = NfNN_PushStruct(Mem, nfnn_optimizer_sgd_param);
    Param->Tensor = T;
    Param->Next = 0;
    NFNN_SLL_PushBack(Optimizer->SGD.First, Optimizer->SGD.Last, Param);
}

static void NfNN_Optimizer_ZeroGrad(nfnn_optimizer *Optimizer)
{
    for (nfnn_optimizer_sgd_param *Param = Optimizer->SGD.First;
         Param != 0;
         Param = Param->Next)
    {
        NfNN_Math_Zero_f32(Param->Tensor->Gradient, NfNN_Length(Param->Tensor));
    }
}

static void NfNN_Optimizer_Step(nfnn_optimizer *Optimizer)
{
    for (nfnn_optimizer_sgd_param *Param = Optimizer->SGD.First;
         Param != 0;
         Param = Param->Next)
    {
        NfNN_Update(Param->Tensor, Optimizer->LearningRate / (f32)Optimizer->NumberOfWorkers);
    }
}

#endif // NFNN_OPTIMIZER_H