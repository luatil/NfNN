#include "../../../../lib/nfnn.h"
#include "../../../../lib/nfnn_mnist.h"

typedef struct model model;
struct model
{
    nfnn_tensor *W1;
    nfnn_tensor *B1;
    nfnn_tensor *W2;
    nfnn_tensor *B2;
};

typedef struct configuration configuration;
struct configuration
{
    u32 Seed;
    f32 LearningRate;
    u32 NumberOfWorkers;
    u32 NumberOfEpochs;
    u32 NumberOfUpdates;
    u32 Port;
    char IpAddress[NI_MAXHOST];
    char TrainingImagesFilePath[2048];
    char TrainLabelsFilePath[2048];
    f32 TrainingSplit; // [0, 1] Proportion to be used between training and validation
    u32 ValidationBatchSize;
    u32 TrainBatchSize;
};

static void
PrintConfiguration(configuration Config)
{
    printf("Configuration:\n");
    printf("  Seed: %u\n", Config.Seed);
    printf("  Learning Rate: %.4f\n", Config.LearningRate);
    printf("  Number of Workers: %u\n", Config.NumberOfWorkers);
    printf("  Number of Epochs: %u\n", Config.NumberOfEpochs);
    printf("  Number of Updates: %u\n", Config.NumberOfUpdates);
    printf("  Port: %u\n", Config.Port);
    printf("  IP Address: %s\n", Config.IpAddress);
    printf("  Training Images File Path: %s\n", Config.TrainingImagesFilePath);
    printf("  Training Labels File Path: %s\n", Config.TrainLabelsFilePath);
    printf("  Training Split: %.2f\n", Config.TrainingSplit);
    printf("  Validation Batch Size: %u\n", Config.ValidationBatchSize);
    printf("  Training Batch Size: %u\n", Config.TrainBatchSize);
}

static void
PrintHelp(char *Exe)
{
    printf("Usage: %s <options>\n", Exe);
    printf("\nOptions:\n");
    printf("  --server, -s                Run as a server\n");
    printf("  --worker, -w                Run as a worker\n");
    printf("  --seed <number>             Set the random seed (default: 3245)\n");
    printf("  --learning-rate <number>    Set the learning rate (default: 0.01)\n");
    printf("  --workers <number>          Set the number of workers (default: 1)\n");
    printf("  --validation <number>       Set the validation batch size (default: 128)\n");
    printf("  --training <number>         Set the training batch size (default: 32)\n");
    printf("  --epochs <number>           Set the number of epochs (default: 5)\n");
    printf("  --updates <number>          Set the number of updates (default: 100000)\n");
    printf("  --port <number>             Set the port (default: 21756)\n");
    printf("  --ip <address>              Set the IP address (default: localhost)\n");
    printf("  --training-images <path>    Set the training images file path\n");
    printf("  --training-labels <path>    Set the training labels file path\n");
    printf("  --training-split <number>   Set the training split (default: 0.8)\n");
    printf("  --help, -h                  Show this help message and exit\n");
}

static nfnn_optimizer *CreateOptimizer(nfnn_memory_arena *Mem, model Model, f32 LearningRate, u32 NumberOfWorkers)
{
    nfnn_optimizer *Result = NfNN_Optimizer_SGD(Mem, LearningRate, NumberOfWorkers);
    NfNN_Optimizer_AddParam(Mem, Result, Model.W1);
    NfNN_Optimizer_AddParam(Mem, Result, Model.B1);
    NfNN_Optimizer_AddParam(Mem, Result, Model.W2);
    NfNN_Optimizer_AddParam(Mem, Result, Model.B2);
    return Result;
}

static model CreateModel(nfnn_memory_arena *Mem, nfnn_random_state *Random)
{
    model Result = {0};
    Result.W1 = NfNN_Matrix(Mem, Random, 784, 32);
    Result.B1 = NfNN_Matrix(Mem, Random, 1, 32);
    Result.W2 = NfNN_Matrix(Mem, Random, 32, 10);
    Result.B2 = NfNN_Matrix(Mem, Random, 1, 10);
    return Result;
}

static nfnn_tensor *Forward(nfnn_memory_arena *Mem, model Model, nfnn_tensor *Input)
{
    nfnn_tensor *Result = 0;
    nfnn_tensor *L1 = NfNN_MatMul(Mem, Input, Model.W1);
    nfnn_tensor *L1b = NfNN_Add(Mem, L1, Model.B1);
    nfnn_tensor *R1 = NfNN_ReLU(Mem, L1b);
    nfnn_tensor *L2 = NfNN_MatMul(Mem, R1, Model.W2);
    nfnn_tensor *L2b = NfNN_Add(Mem, L2, Model.B2);
    Result = NfNN_LogSoftmax(Mem, L2b, 1);
    return Result;
}

static f32 CalculateAccuracy(nfnn_memory_arena *Mem, model Model, nfnn_dataloader_mnist *DataLoader)
{
    u32 Correct = 0;
    u32 Total = 0;
    for (nfnn_dataloader_batch_mnist *It = NfNN_DataLoader_Mnist_NextBatch(DataLoader);
         It != 0;
         It = NfNN_DataLoader_Mnist_NextBatch(DataLoader))
    {
        NfNN_MemoryArena_TempInit(Mem);

        nfnn_tensor *Outputs = Forward(Mem, Model, It->Images);
        nfnn_tensor *Predicted = NfNN_Argmax(Mem, Outputs, 1);

        Total += NfNN_Length(It->Labels);
        Correct += (u32)NfNN_Item(NfNN_SumAll(Mem, NfNN_Equal(Mem, Predicted, It->Labels)));

        NfNN_MemoryArena_TempClear(Mem);
    }

    f32 Accuracy = 100.0 * (f32)Correct / (f32)Total;
    // u32 NumberOfBatches = NfNN_DataLoader_Mnist_NumberOfBatches(DataLoader);
    f32 Result = Accuracy;
    return Result;
}

static void
RunAsServer(configuration Config)
{
    /**
     * Parameter server program:
     * initial model w_0, learning rate lr 
     * w <- w_0
     * for t <- 0, 1, 2, ... do
     *    if received gradients (g_i_s) from worker i, with delay of s steps then
     *    w <- w - lr * g_i_s 
     *    Send w to worker i
     * end for
     *
     * Program for the ith worker:
     * for t_i <- 0, 1, 2, ... do
     *   await current parameter server model w
     *   w_i <- w
     *   sample mini-batch x ~ D_i
     *   compute gradient g_t_i = grad(w_i, x)
     *   send g_t_i to the parameter server
     * end for
     *
     */

    nfnn_memory_arena Mem_P = {0};
    NfNN_MemoryArena_Init(&Mem_P, MB(100));

    nfnn_random_state Random = {0};
    NfNN_Random_Init(&Random, Config.Seed);

    nfnn_memory_arena Mem_T = {0};
    NfNN_MemoryArena_Init(&Mem_T, MB(100));

    nfnn_network_interface *Interface = NfNN_Network_CreateInterface(&Mem_P);

    model Model = CreateModel(&Mem_P, &Random);
    nfnn_optimizer *Optimizer = CreateOptimizer(&Mem_P, Model, Config.LearningRate, 1);

    nfnn_datasets_mnist *FullTrainDataset = NfNN_Datasets_MNIST_Load(&Mem_P, Config.TrainingImagesFilePath, Config.TrainLabelsFilePath, 60000);

    u32 TrainingNumber = (u32)(60000 * Config.TrainingSplit);
    u32 ValidationNumber = 60000 - TrainingNumber;

    nfnn_datasets_mnist *ValidationDataset = NfNN_Datasets_Mnist_Split(&Mem_P, FullTrainDataset, TrainingNumber, ValidationNumber);
    nfnn_dataloader_mnist *ValidationLoader = NfNN_Dataloader_Mnist_Create(&Mem_P, ValidationDataset, Config.ValidationBatchSize, 0);

    nfnn_parameter_server *Server = NfNN_ParameterServer_Create(&Mem_P, Config.IpAddress, Config.Port);

    for (u32 G = 0; G < Config.NumberOfWorkers; G++)
    {
        NfNN_ParameterServer_AddWorker(&Mem_P, Server);
    }

    NfNN_ParameterServer_BroadcastWeights(&Mem_T, Server, Model.W1);
    NfNN_ParameterServer_BroadcastWeights(&Mem_T, Server, Model.B1);
    NfNN_ParameterServer_BroadcastWeights(&Mem_T, Server, Model.W2);
    NfNN_ParameterServer_BroadcastWeights(&Mem_T, Server, Model.B2);
    
    for (u32 T = 0; T <= Config.NumberOfUpdates; T++)
    {
        NfNN_MemoryArena_TempInit(&Mem_T);

        NfNN_Optimizer_ZeroGrad(Optimizer);

        nfnn_platform_socket Sock = NfNN_ParameterServer_AwaitForWorker(&Mem_T, Server);

        NfNN_Network_RecvAddGradient(&Mem_T, Sock, Model.W1);
        NfNN_Network_RecvAddGradient(&Mem_T, Sock, Model.B1);
        NfNN_Network_RecvAddGradient(&Mem_T, Sock, Model.W2);
        NfNN_Network_RecvAddGradient(&Mem_T, Sock, Model.B2);

        NfNN_Optimizer_Step(Optimizer);

        NfNN_Network_SendTensor(&Mem_T, Sock, Model.W1);
        NfNN_Network_SendTensor(&Mem_T, Sock, Model.B1);
        NfNN_Network_SendTensor(&Mem_T, Sock, Model.W2);
        NfNN_Network_SendTensor(&Mem_T, Sock, Model.B2);

        if (T % 1000 == 0)
        {
            f32 Accuracy = CalculateAccuracy(&Mem_T, Model, ValidationLoader);
            printf("Iteration %d: Validation Accuracy: %f\n", T, Accuracy);
        }
        NfNN_MemoryArena_TempClear(&Mem_T);
    }

    NfNN_Network_DestroyInterface(Interface);
}

static void
RunAsWorker(configuration Config)
{
    /**
     * Program for the ith worker:
     * for t_i <- 0, 1, 2, ... do
     *   await current parameter server model w
     *   w_i <- w
     *   sample mini-batch x ~ D_i
     *   compute gradient g_t_i = grad(w_i, x)
     *   send g_t_i to the parameter server
     * end for
     */
    nfnn_memory_arena Mem_P = {0};
    NfNN_MemoryArena_Init(&Mem_P, GB(1));

    nfnn_random_state Random = {0};
    NfNN_Random_Init(&Random, Config.Seed);

    nfnn_memory_arena Mem_T = {0};
    NfNN_MemoryArena_Init(&Mem_T, GB(1));

    nfnn_network_interface *Interface = NfNN_Network_CreateInterface(&Mem_P);

    model Model = CreateModel(&Mem_P, &Random);
    nfnn_optimizer *Optimizer = CreateOptimizer(&Mem_P, Model, Config.LearningRate, Config.NumberOfWorkers);

    // Connect to parameter server
    nfnn_socket *Socket = NfNN_Network_TCPConnect(&Mem_P, Config.IpAddress, Config.Port);

    nfnn_datasets_mnist *FullTrainDataset = NfNN_Datasets_MNIST_Load(&Mem_P, Config.TrainingImagesFilePath, Config.TrainLabelsFilePath, 60000);

    u32 TrainingNumber = (u32)(60000 * Config.TrainingSplit);

    nfnn_datasets_mnist *TrainDataset = NfNN_Datasets_Mnist_Split(&Mem_P, FullTrainDataset, 0, TrainingNumber);
    nfnn_dataloader_mnist *TrainLoader = NfNN_Dataloader_Mnist_Create(&Mem_P, TrainDataset, 32, &Random);

    for (u32 IterationCount = 0; IterationCount <= Config.NumberOfUpdates;)
    {
        for (nfnn_dataloader_batch_mnist *It = NfNN_DataLoader_Mnist_NextBatch(TrainLoader);
             It != 0 && IterationCount < Config.NumberOfUpdates;
             It = NfNN_DataLoader_Mnist_NextBatch(TrainLoader))
        {
            NfNN_MemoryArena_TempInit(&Mem_T);

            NfNN_Optimizer_ZeroGrad(Optimizer);

            // TODO(luatil): Handle disconnects

            NfNN_Network_RecvTensor(&Mem_T, Socket->Handle, Model.W1);
            NfNN_Network_RecvTensor(&Mem_T, Socket->Handle, Model.B1);
            NfNN_Network_RecvTensor(&Mem_T, Socket->Handle, Model.W2);
            NfNN_Network_RecvTensor(&Mem_T, Socket->Handle, Model.B2);

            nfnn_tensor *Outputs = Forward(&Mem_T, Model, It->Images);
            nfnn_tensor *Loss = NfNN_NLLLoss(&Mem_T, Outputs, It->Labels);

            NfNN_AutoGrad_Backward(&Mem_T, Loss);

            if (IterationCount % 100 == 0)
            {
                printf("Iteration %d: Loss: %f\n", IterationCount, NfNN_Item(Loss));
            }

            NfNN_Network_SendGradient(&Mem_T, Socket->Handle, Model.W1);
            NfNN_Network_SendGradient(&Mem_T, Socket->Handle, Model.B1);
            NfNN_Network_SendGradient(&Mem_T, Socket->Handle, Model.W2);
            NfNN_Network_SendGradient(&Mem_T, Socket->Handle, Model.B2);

            NfNN_MemoryArena_TempClear(&Mem_T);

            IterationCount++;
        }
    }

    NfNN_Network_DestroyInterface(Interface);
}

int main(int ArgumentCount, char **Arguments)
{
    u32 IsServer = 0;
    u32 IsWorker = 0;

    configuration Config = {0};
    Config.Seed = 3245;
    Config.LearningRate = 0.01f;
    Config.NumberOfWorkers = 1;
    Config.NumberOfEpochs = 5;
    Config.NumberOfUpdates = 5000;
    Config.Port = 21756;
    NFNN_STRCPY(Config.IpAddress, "localhost");
    #if defined(_WIN32)
    NFNN_STRCPY(Config.TrainingImagesFilePath, "..\\examples\\mnist\\dataset\\MNIST\\raw\\train-images-idx3-ubyte");
    NFNN_STRCPY(Config.TrainLabelsFilePath, "..\\examples\\mnist\\dataset\\MNIST\\raw\\train-labels-idx1-ubyte");
    #else
    NFNN_STRCPY(Config.TrainingImagesFilePath, "../examples/mnist/dataset/MNIST/raw/train-images-idx3-ubyte");
    NFNN_STRCPY(Config.TrainLabelsFilePath, "../examples/mnist/dataset/MNIST/raw/train-labels-idx1-ubyte");
    #endif
    Config.TrainingSplit = 0.8; // [0, 1] Proportion to be used between training and validation
    Config.ValidationBatchSize = 128;
    Config.TrainBatchSize = 32;

    for(u32 I = 1; I < ArgumentCount;)
    {
        if (NFNN_STREQUAL(Arguments[I], "--server") || NFNN_STREQUAL(Arguments[I], "-s"))
        {
            IsServer = 1;
            I++;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--worker") || NFNN_STREQUAL(Arguments[I], "-w"))
        {
            IsWorker = 1;
            I++;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--seed"))
        {
            NFNN_ASSERT((I + 1) < ArgumentCount, "Seed Number must be specified");
            Config.Seed = NFNN_ATOI(Arguments[I + 1]);
            I += 2;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--workers"))
        {
            NFNN_ASSERT((I + 1) < ArgumentCount, "Number of workers must be specified");
            Config.NumberOfWorkers = NFNN_ATOI(Arguments[I + 1]);
            I += 2;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--validation"))
        {
            NFNN_ASSERT((I + 1) < ArgumentCount, "Validation batch size must be specified");
            Config.ValidationBatchSize = NFNN_ATOI(Arguments[I + 1]);
            I += 2;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--training"))
        {
            NFNN_ASSERT((I + 1) < ArgumentCount, "Training batch size must be specified");
            Config.TrainBatchSize = NFNN_ATOI(Arguments[I + 1]);
            I += 2;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--learning-rate"))
        {
            NFNN_ASSERT((I + 1) < ArgumentCount, "Learning rate must be specified");
            Config.LearningRate = NFNN_ATOF(Arguments[I + 1]);
            I += 2;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--epochs"))
        {
            NFNN_ASSERT((I + 1) < ArgumentCount, "Number of epochs must be specified");
            Config.NumberOfEpochs = NFNN_ATOI(Arguments[I + 1]);
            I += 2;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--updates"))
        {
            NFNN_ASSERT((I + 1) < ArgumentCount, "Number of updates must be specified");
            Config.NumberOfUpdates = NFNN_ATOI(Arguments[I + 1]);
            I += 2;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--port"))
        {
            NFNN_ASSERT((I + 1) < ArgumentCount, "Port must be specified");
            Config.Port = (u16)NFNN_ATOI(Arguments[I + 1]);
            I += 2;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--ip"))
        {
            NFNN_ASSERT((I + 1) < ArgumentCount, "IP address must be specified");
            NFNN_STRCPY(Config.IpAddress, Arguments[I + 1]);
            I += 2;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--training-images"))
        {
            NFNN_ASSERT((I + 1) < ArgumentCount, "Training images file path must be specified");
            NFNN_STRCPY(Config.TrainingImagesFilePath, Arguments[I + 1]);
            I += 2;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--training-labels"))
        {
            NFNN_ASSERT((I + 1) < ArgumentCount, "Training labels file path must be specified");
            NFNN_STRCPY(Config.TrainLabelsFilePath, Arguments[I + 1]);
            I += 2;
        }
        else if (NFNN_STREQUAL(Arguments[I], "--training-split"))
        {
            NFNN_ASSERT((I + 1) < ArgumentCount, "Training split must be specified");
            Config.TrainingSplit = NFNN_ATOF(Arguments[I + 1]);
            I += 2;
        }
        else if (ArgumentCount == 1 || NFNN_STREQUAL(Arguments[I], "--help") || NFNN_STREQUAL(Arguments[I], "-h"))
        {
            PrintHelp(Arguments[0]);
            return 0;
        }
        else
        {
            printf("Unknown argument: %s\n", Arguments[I]);
            PrintHelp(Arguments[0]);
            return 1;
        }
    }

    NFNN_ASSERT(!(IsServer && IsWorker), "Cannot be both server and worker");

    PrintConfiguration(Config);

    if (IsServer)
    {
        RunAsServer(Config);
    }
    else if (IsWorker)
    {
        RunAsWorker(Config);
    }
    else
    {
        PrintHelp(Arguments[0]);
    }
}
