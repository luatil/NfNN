#include "../../../../lib/nfnn.h"

int main()
{
    nfnn_memory_arena Mem_P = {0};
    NfNN_MemoryArena_Init(&Mem_P, MB(1));


    nfnn_random_state Random = {0};
    NfNN_Random_Init(&Random, 41423);

    f32 X_Data[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };

    f32 Y_Data[] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    nfnn_tensor *X = NfNN_From_f32(&Mem_P, X_Data, NfNN_Dim2(4, 2));
    nfnn_tensor *Y = NfNN_From_f32(&Mem_P, Y_Data, NfNN_Dim2(1, 4));

    nfnn_tensor *W1 = NfNN_From_f32(&Mem_P, (f32[]){ 0.15f, -0.61f, -0.26f, 0.35f}, NfNN_Dim2(2, 2));
    nfnn_tensor *B1 = NfNN_From_f32(&Mem_P, (f32[]){-0.25, 0.68f}, NfNN_Dim2(1, 2));
    nfnn_tensor *W2 = NfNN_From_f32(&Mem_P, (f32[]){-0.45, 0.96}, NfNN_Dim2(2, 1));
    nfnn_tensor *B2 = NfNN_From_f32(&Mem_P, (f32[]){0.78}, NfNN_Dim2(1, 1));

    u32 Epochs = 32;

    // L1 = x1 * w1 (1, 2)
    // R1 = relu(L1) (1, 2)
    // L2 = R1 * w2 (1, 1)

    nfnn_memory_arena Mem_T = {0};
    NfNN_MemoryArena_Init(&Mem_T, MB(1));

    f32 LearningRate = 0.01f;

    for (u32 I = 0; I < Epochs; I++)
    {
        nfnn_tensor *L1 = NfNN_MatMul(&Mem_T, X, W1);

        nfnn_tensor *L1b = NfNN_Add(&Mem_T, L1, B1);
        nfnn_tensor *R1 = NfNN_ReLU(&Mem_T, L1b);

        nfnn_tensor *L2 = NfNN_MatMul(&Mem_T, R1, W2);
        nfnn_tensor *L2b = NfNN_Add(&Mem_T, L2, B2);

        nfnn_tensor *Diff = NfNN_Sub(&Mem_T, L2b, Y); 
        nfnn_tensor *Square = NfNN_Square(&Mem_T, Diff);
        nfnn_tensor *Loss = NfNN_SumAll(&Mem_T, Square);


        printf("%d:%f\n", I, NfNN_Item(Loss));

        NfNN_AutoGrad_Backward(&Mem_T, Loss);

        NfNN_Update(W1, LearningRate);
        NfNN_Update(B1, LearningRate);
        NfNN_Update(W2, LearningRate);
        NfNN_Update(B2, LearningRate);

        NfNN_AutoGrad_ZeroGrad(&Mem_T, Loss);
        NfNN_MemoryArena_Clear(&Mem_T);
    }
}