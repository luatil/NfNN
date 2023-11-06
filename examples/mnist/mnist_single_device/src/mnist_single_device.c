#include "../../../../lib/nfnn.h"
#include "../../../../lib/nfnn_mnist.h"

int main()
{
    nfnn_memory_arena Mem_P = {0};
    NfNN_MemoryArena_Init(&Mem_P, MB(100));

    nfnn_random_state Random = {0};
    NfNN_Random_Init(&Random, 234521);

    #if defined(_WIN32)
    char *TrainImagesFilePath = "..\\examples\\mnist\\dataset\\MNIST\\raw\\train-images-idx3-ubyte";
    char *TrainLabelsFilePath = "..\\examples\\mnist\\dataset\\MNIST\\raw\\train-labels-idx1-ubyte";
    #else
    char *TrainImagesFilePath = "../examples/mnist/dataset/MNIST/raw/train-images-idx3-ubyte";
    char *TrainLabelsFilePath = "../examples/mnist/dataset/MNIST/raw/train-labels-idx1-ubyte";
    #endif
    nfnn_datasets_mnist *FullTrainDataset = NfNN_Datasets_MNIST_Load(&Mem_P, TrainImagesFilePath, TrainLabelsFilePath, 60000);

    u32 TrainingNumber = (u32)(60000 * 0.8);
    u32 ValidationNumber = 60000 - TrainingNumber;

    nfnn_datasets_mnist *TrainDataset = NfNN_Datasets_Mnist_Split(&Mem_P, FullTrainDataset, 0, TrainingNumber);
    nfnn_datasets_mnist *ValidationDataset = NfNN_Datasets_Mnist_Split(&Mem_P, FullTrainDataset, TrainingNumber, ValidationNumber);

    nfnn_dataloader_mnist *TrainLoader = NfNN_Dataloader_Mnist_Create(&Mem_P, TrainDataset, 64, &Random);
    nfnn_dataloader_mnist *ValidationLoader = NfNN_Dataloader_Mnist_Create(&Mem_P, ValidationDataset, 128, 0);

    nfnn_tensor *W1 = NfNN_Matrix(&Mem_P, &Random, 784, 32);
    nfnn_tensor *B1 = NfNN_Matrix(&Mem_P, &Random, 1, 32);
    nfnn_tensor *W2 = NfNN_Matrix(&Mem_P, &Random, 32, 10);
    nfnn_tensor *B2 = NfNN_Matrix(&Mem_P, &Random, 1, 10);

    nfnn_optimizer *Optimizer = NfNN_Optimizer_SGD(&Mem_P, 0.01f, 1);
    // nfnn_optimizer *Optimizer = NfNN_Optimizer_Adam(&Mem_P, 0, 1, 0, 0);

    NfNN_Optimizer_AddParam(&Mem_P, Optimizer, W1);
    NfNN_Optimizer_AddParam(&Mem_P, Optimizer, B1);
    NfNN_Optimizer_AddParam(&Mem_P, Optimizer, W2);
    NfNN_Optimizer_AddParam(&Mem_P, Optimizer, B2);

    nfnn_memory_arena Mem_T = {0};
    NfNN_MemoryArena_Init(&Mem_T, MB(100));

    u32 NumberOfEpochs = 5;

    {
        u32 Correct = 0;
        u32 Total = 0;
        for (nfnn_dataloader_batch_mnist *It = NfNN_DataLoader_Mnist_NextBatch(ValidationLoader);
             It != 0;
             It = NfNN_DataLoader_Mnist_NextBatch(ValidationLoader))
        {
            NfNN_MemoryArena_TempInit(&Mem_T);

            nfnn_tensor *L1 = NfNN_MatMul(&Mem_T, It->Images, W1);
            nfnn_tensor *L1b = NfNN_Add(&Mem_T, L1, B1);
            nfnn_tensor *R1 = NfNN_ReLU(&Mem_T, L1b);
            nfnn_tensor *L2 = NfNN_MatMul(&Mem_T, R1, W2);
            nfnn_tensor *L2b = NfNN_Add(&Mem_T, L2, B2);
            nfnn_tensor *Outputs = NfNN_LogSoftmax(&Mem_T, L2b, 1);

            nfnn_tensor *Predicted = NfNN_Argmax(&Mem_T, Outputs, 1);

            // NOTE(luatil): This might be wrong
            Total += NfNN_Length(It->Labels);
            u32 CorrectInBatch = (u32)NfNN_Item(NfNN_SumAll(&Mem_T, NfNN_Equal(&Mem_T, Predicted, It->Labels)));
            Correct += CorrectInBatch;

            NfNN_MemoryArena_TempClear(&Mem_T);
        }

        f32 ValidationAccuracy = 100.0 * (f32)Correct / (f32)Total;
        // u32 NumberOfBatches = NfNN_DataLoader_Mnist_NumberOfBatches(TrainLoader);
        printf("Epoch %d: Validation Accuracy: %f\n", 0,  ValidationAccuracy);
    }

    for (u32 Epoch = 0; Epoch < NumberOfEpochs; Epoch++)
    {
        f32 RunningLoss = 0.0f;

        u32 IterationCount = 0;

        for (nfnn_dataloader_batch_mnist *It = NfNN_DataLoader_Mnist_NextBatch(TrainLoader);
             It != 0;
             It = NfNN_DataLoader_Mnist_NextBatch(TrainLoader))
        {
            NfNN_MemoryArena_TempInit(&Mem_T);
            NfNN_Optimizer_ZeroGrad(Optimizer);

            nfnn_tensor *L1 = NfNN_MatMul(&Mem_T, It->Images, W1);
            nfnn_tensor *L1b = NfNN_Add(&Mem_T, L1, B1);
            nfnn_tensor *R1 = NfNN_ReLU(&Mem_T, L1b);
            nfnn_tensor *L2 = NfNN_MatMul(&Mem_T, R1, W2);
            nfnn_tensor *L2b = NfNN_Add(&Mem_T, L2, B2);
            nfnn_tensor *Outputs = NfNN_LogSoftmax(&Mem_T, L2b, 1);

            nfnn_tensor *Loss = NfNN_NLLLoss(&Mem_T, Outputs, It->Labels);
            NfNN_AutoGrad_Backward(&Mem_T, Loss);

            NfNN_Optimizer_Step(Optimizer);

            f32 BatchLoss = NfNN_Item(Loss);

            NFNN_ASSERT(!NFNN_IS_NAN(BatchLoss), "Loss is NaN");

            RunningLoss += BatchLoss;

            IterationCount++;
            if (IterationCount % 100 == 0)
            {
                printf("Epoch %d, Iteration %d: Loss: %f\n", Epoch + 1, IterationCount, NfNN_Item(Loss));
            }

            NfNN_MemoryArena_TempClear(&Mem_T);
        }

        u32 Correct = 0;
        u32 Total = 0;
        for (nfnn_dataloader_batch_mnist *It = NfNN_DataLoader_Mnist_NextBatch(ValidationLoader);
             It != 0;
             It = NfNN_DataLoader_Mnist_NextBatch(ValidationLoader))
        {
            NfNN_MemoryArena_TempInit(&Mem_T);

            nfnn_tensor *L1 = NfNN_MatMul(&Mem_T, It->Images, W1);
            nfnn_tensor *L1b = NfNN_Add(&Mem_T, L1, B1);
            nfnn_tensor *R1 = NfNN_ReLU(&Mem_T, L1b);
            nfnn_tensor *L2 = NfNN_MatMul(&Mem_T, R1, W2);
            nfnn_tensor *L2b = NfNN_Add(&Mem_T, L2, B2);
            nfnn_tensor *Outputs = NfNN_LogSoftmax(&Mem_T, L2b, 1);

            nfnn_tensor *Predicted = NfNN_Argmax(&Mem_T, Outputs, 1);

            Total += NfNN_Length(It->Labels);
            u32 CorrectInBatch = (u32)NfNN_Item(NfNN_SumAll(&Mem_T, NfNN_Equal(&Mem_T, Predicted, It->Labels)));
            Correct += CorrectInBatch;

            NfNN_MemoryArena_TempClear(&Mem_T);
        }

        NfNN_MemoryArena_TempInit(&Mem_T);
        f32 ValidationAccuracy = 100.0 * (f32)Correct / (f32)Total;
        u32 NumberOfBatches = NfNN_DataLoader_Mnist_NumberOfBatches(TrainLoader);
        printf("Epoch %d: Loss: %f, Validation Accuracy: %f\n", Epoch + 1, RunningLoss / NumberOfBatches, ValidationAccuracy);
    }
    printf("Training Complete!\n");
}
