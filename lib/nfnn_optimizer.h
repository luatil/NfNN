#ifndef NFNN_OPTIMIZER_H
#define NFNN_OPTIMIZER_H

#include "nfnn_math.h"
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
    nfnn_tensor *B;
};

typedef struct nfnn_optimizer_adam_param nfnn_optimizer_adam_param;
struct nfnn_optimizer_adam_param
{
    nfnn_tensor *M;
    nfnn_tensor *V;
};

typedef struct nfnn_optimizer_sgd nfnn_optimizer_sgd;
struct nfnn_optimizer_sgd
{
    f32 WeightDecay;
    f32 Momentum;
    f32 Dampening;
    bool Nesterov;
};

typedef struct nfnn_optimizer_adam nfnn_optimizer_adam;
struct nfnn_optimizer_adam
{
    f32 Beta1, Beta2; // in [0, 1) Exponential decay rates for the moment estimates
};

typedef struct nfnn_optimizer_param nfnn_optimizer_param;
struct nfnn_optimizer_param
{
    nfnn_tensor *Tensor;
    nfnn_optimizer_param *Next;
    union {
        nfnn_optimizer_sgd_param SGD;
        nfnn_optimizer_adam_param Adam;
    };
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
        nfnn_optimizer_adam Adam;
    };
    nfnn_optimizer_param *First;
    nfnn_optimizer_param *Last;
};

static nfnn_optimizer *NfNN_Optimizer_SGD(nfnn_memory_arena *Mem, f32 LearningRate, u32 NumberOfWorkers, f32 Momentum,
                                          f32 Dampening, f32 WeightDecay, bool Nesterov)
{
    nfnn_optimizer *Result = NfNN_PushStruct(Mem, nfnn_optimizer);
    Result->Type = NFNN_OPTIMIZER_SGD;
    Result->LearningRate = LearningRate;
    Result->First = 0;
    Result->Last = 0;
    Result->NumberOfWorkers = NumberOfWorkers;

    // NOTE(luatil): All should default to 0
    Result->SGD.Dampening = Dampening;
    Result->SGD.WeightDecay = WeightDecay;
    Result->SGD.Nesterov = Nesterov;
    Result->SGD.Momentum = Momentum;

    return Result;
}

static nfnn_optimizer *NfNN_Optimizer_Adam(nfnn_memory_arena *Mem, f32 LearningRate, u32 NumberOfWorkers, f32 Beta1,
                                           f32 Beta2)
{
    nfnn_optimizer *Result = NfNN_PushStruct(Mem, nfnn_optimizer);
    Result->Type = NFNN_OPTIMIZER_ADAM;
    Result->First = 0;
    Result->Last = 0;
    Result->NumberOfWorkers = NumberOfWorkers;

    // Adam part
    if (LearningRate != 0)
    {
        Result->LearningRate = LearningRate;
    }
    else
    {
        Result->LearningRate = 0.001f;
    }
    if (Beta1 != 0)
    {
        Result->Adam.Beta1 = Beta1;
    }
    else
    {
        Result->Adam.Beta1 = 0.9f;
    }
    if (Beta2 != 0)
    {
        Result->Adam.Beta2 = Beta2;
    }
    else
    {
        Result->Adam.Beta2 = 0.999f;
    }

    return Result;
}

static void NfNN_Optimizer_AddParam(nfnn_memory_arena *Mem, nfnn_optimizer *Optimizer, nfnn_tensor *T)
{
    switch (Optimizer->Type)
    {
    case NFNN_OPTIMIZER_SGD: {
        nfnn_optimizer_param *Param = NfNN_PushStruct(Mem, nfnn_optimizer_param);
        Param->Tensor = T;
        Param->Next = 0;

        Param->SGD.B = NfNN_ZeroesLike(Mem, T);

        NFNN_SLL_PushBack(Optimizer->First, Optimizer->Last, Param);
    }
    break;
    case NFNN_OPTIMIZER_ADAM: {
        nfnn_optimizer_param *Param = NfNN_PushStruct(Mem, nfnn_optimizer_param);
        Param->Tensor = T;
        Param->Next = 0;

        Param->Adam.M = NfNN_ZeroesLike(Mem, T);
        Param->Adam.V = NfNN_ZeroesLike(Mem, T);

        NFNN_SLL_PushBack(Optimizer->First, Optimizer->Last, Param);
    }
    break;
    }
}

static void NfNN_Optimizer_SGDUpdate(nfnn_tensor *T, nfnn_optimizer_sgd_param Param, f32 Lr, f32 WeightDecay,
                                     f32 Momentum, f32 Dampening, bool Nesterov, u32 Timestamp)
{
    for (u32 I = 0; I < NfNN_Length(T); I++)
    {
        if (WeightDecay != 0)
        {
            T->Gradient[I] += WeightDecay * T->Data[I];
        }

        if (Momentum != 0)
        {
            if (Timestamp > 1)
            {
                Param.B->Data[I] = Momentum * Param.B->Data[I] + (1.0f - Dampening) * T->Gradient[I];
            }
            else
            {
                Param.B->Data[I] = T->Gradient[I];
            }
            if (Nesterov)
            {
                T->Gradient[I] = T->Gradient[I] + Momentum * Param.B->Data[I];
            }
            else
            {
                T->Gradient[I] = Param.B->Data[I];
            }
        }

        T->Data[I] -= Lr * T->Gradient[I];
    }
}

static void NfNN_Optimizer_AdamUpdate(nfnn_tensor *T, nfnn_optimizer_adam_param Param, f32 Alpha, f32 Beta1, f32 Beta2)
{
    f32 *Theta = T->Data;
    f32 *G = T->Gradient;
    f32 *M = Param.M->Data;
    f32 *V = Param.V->Data;
    for (u32 I = 0; I < NfNN_Length(T); I++)
    {
        M[I] = Beta1 * M[I] + (1.0f - Beta1) * G[I];
        V[I] = Beta2 * V[I] + (1.0f - Beta2) * (G[I] * G[I]);
        Theta[I] = Theta[I] - Alpha * M[I] / (NfNN_Math_Single_Sqrt_f32(V[I]) + 1e-8f);
    }
}

static void NfNN_Optimizer_Update(nfnn_optimizer *Optimizer, nfnn_optimizer_param *Param)
{
    switch (Optimizer->Type)
    {
    case NFNN_OPTIMIZER_SGD: {
        NfNN_Optimizer_SGDUpdate(Param->Tensor, Param->SGD, Optimizer->LearningRate, Optimizer->SGD.WeightDecay,
                                 Optimizer->SGD.Momentum, Optimizer->SGD.Dampening, Optimizer->SGD.Nesterov,
                                 Optimizer->Iteration);
    }
    break;
    case NFNN_OPTIMIZER_ADAM: {
        NfNN_Optimizer_AdamUpdate(Param->Tensor, Param->Adam, Optimizer->LearningRate, Optimizer->Adam.Beta1,
                                  Optimizer->Adam.Beta2);
    }
    break;
    }
}

static void NfNN_Optimizer_Step(nfnn_optimizer *Optimizer)
{
    for (nfnn_optimizer_param *Param = Optimizer->First; Param != 0; Param = Param->Next)
    {
        NfNN_Optimizer_Update(Optimizer, Param);
    }
    Optimizer->Iteration++;
}

static void NfNN_Optimizer_ZeroGrad(nfnn_optimizer *Optimizer)
{
    for (nfnn_optimizer_param *Param = Optimizer->First; Param != 0; Param = Param->Next)
    {
        NfNN_Math_Zero_f32(Param->Tensor->Gradient, NfNN_Length(Param->Tensor));
    }
}

#endif // NFNN_OPTIMIZER_H
