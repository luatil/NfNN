#include "../../../../lib/nfnn.h"
#include "../../../../lib/nfnn_mnist.h"

int main(int ArgCount, char **Args)
{

    if (ArgCount != 2)
    {
        fprintf(stderr, "Usage: %s <seed>", Args[0]);
        return 1;
    }

    u32 Seed = NFNN_ATOI(Args[1]);

    nfnn_memory_arena Mem_P = {0};
    NfNN_MemoryArena_Init(&Mem_P, MB(100));

    nfnn_random_state Random = {0};
    NfNN_Random_Init(&Random, Seed);

#if defined(_WIN32)
    char *TrainImagesFilePath = "..\\examples\\mnist\\dataset\\MNIST\\raw\\train-images-idx3-ubyte";
    char *TrainLabelsFilePath = "..\\examples\\mnist\\dataset\\MNIST\\raw\\train-labels-idx1-ubyte";
#else
    char *TrainImagesFilePath = "../examples/mnist/dataset/MNIST/raw/train-images-idx3-ubyte";
    char *TrainLabelsFilePath = "../examples/mnist/dataset/MNIST/raw/train-labels-idx1-ubyte";
#endif

    nfnn_datasets_mnist *FullTrainDataset =
        NfNN_Datasets_MNIST_Load(&Mem_P, TrainImagesFilePath, TrainLabelsFilePath, 60000);

    u32 TrainingNumber = (u32)(60000 * 0.8);
    u32 ValidationNumber = 60000 - TrainingNumber;

    nfnn_datasets_mnist *TrainDataset = NfNN_Datasets_Mnist_Split(&Mem_P, FullTrainDataset, 0, TrainingNumber);
    nfnn_datasets_mnist *ValidationDataset =
        NfNN_Datasets_Mnist_Split(&Mem_P, FullTrainDataset, TrainingNumber, ValidationNumber);

    nfnn_dataloader_mnist *TrainLoader = NfNN_Dataloader_Mnist_Create(&Mem_P, TrainDataset, 4, &Random);
    nfnn_dataloader_mnist *ValidationLoader = NfNN_Dataloader_Mnist_Create(&Mem_P, ValidationDataset, 32, 0);

    for (nfnn_dataloader_batch_mnist *It = NfNN_DataLoader_Mnist_NextBatch(TrainLoader); It != 0;
         It = NfNN_DataLoader_Mnist_NextBatch(TrainLoader))
    {
        NfNN_MemoryArena_TempInit(&Mem_P);
        NfNN_Datasets_Mnist_PrintImage(It->Images, It->Labels, TrainLoader->BatchSize);
        getchar();
        NfNN_MemoryArena_TempClear(&Mem_P);
    }

    for (nfnn_dataloader_batch_mnist *It = NfNN_DataLoader_Mnist_NextBatch(ValidationLoader); It != 0;
         It = NfNN_DataLoader_Mnist_NextBatch(ValidationLoader))
    {
        NfNN_MemoryArena_TempInit(&Mem_P);
        NfNN_Datasets_Mnist_PrintImage(It->Images, It->Labels, ValidationLoader->BatchSize);
        getchar();
        NfNN_MemoryArena_TempClear(&Mem_P);
    }
}
