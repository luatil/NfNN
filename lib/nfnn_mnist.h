#ifndef NFNN_MNIST
#define NFNN_MNIST

#include "nfnn.h"

typedef struct nfnn_datasets_mnist nfnn_datasets_mnist;
struct nfnn_datasets_mnist
{
    u32 NumberOfImages;
    u8 *Images;
    u8 *Labels;
};

typedef struct nfnn_dataloader_batch_mnist nfnn_dataloader_batch_mnist;
struct nfnn_dataloader_batch_mnist
{
    nfnn_tensor *Images;
    nfnn_tensor *Labels;
};

typedef struct nfnn_dataloader_mnist nfnn_dataloader_mnist;
struct nfnn_dataloader_mnist
{
    u32 BatchSize;
    u32 Current;
    nfnn_datasets_mnist *Dataset;
    nfnn_dataloader_batch_mnist *Batch;
    nfnn_random_state *Random;
};

static bool
ReadIDX1(char *Filename, u8 *Labels, u32 ExpectedNumberOfLabels)
{
    FILE *File = fopen(Filename, "rb");
    if (File)
    {
        u32 MagicConstant;
        if (!fread(&MagicConstant, 4, 1, File))
        {
            fprintf(stderr, "Could not read magic constant\n");
            fclose(File);
            return false;
        }
        MagicConstant = NfNN_Math_BigEndianToLittleEndian_u32(MagicConstant);
        if (MagicConstant == 2049)
        {
            u32 NumberOfLabels;
            if (!fread(&NumberOfLabels, 4, 1, File))
            {
                fprintf(stderr, "Could not read number of labels\n");
                fclose(File);
                return false;
            }
            NumberOfLabels = NfNN_Math_BigEndianToLittleEndian_u32(NumberOfLabels);
            // if (NumberOfLabels == ExpectedNumberOfLabels) {
            if (1)
            {
                if (!fread(Labels, 1, NumberOfLabels, File))
                {
                    fprintf(stderr, "Could not read labels\n");
                    fclose(File);
                    return false;
                }
                fclose(File);
                return true;
            }
            else
            {
                fprintf(stderr, "Number Of Labels Does Not Match The Expected\n");
                fclose(File);
                return false;
            }
        }
        else
        {
            fprintf(stderr, "Magic Constant Not Found\n");
            fclose(File);
            return false;
        }
    }
    else
    {
        fprintf(stderr, "File not found\n");
        return false;
    }
}

static bool
ReadIDX3(char *Filename, u8 *Images, u32 ExpectedNumberOfImages, u32 ExpectedHeight, u32 ExpectedWidth)
{
    FILE *File = fopen(Filename, "rb");
    if (File)
    {
        u32 MagicConstant;
        if (!fread(&MagicConstant, 4, 1, File))
        {
            fprintf(stderr, "Could not read magic constant\n");
            fclose(File);
            return false;
        }
        MagicConstant = NfNN_Math_BigEndianToLittleEndian_u32(MagicConstant);
        if (MagicConstant == 2051)
        {
            u32 NumberOfImages;
            if (!fread(&NumberOfImages, 4, 1, File))
            {
                fprintf(stderr, "Could not read number of images\n");
                fclose(File);
                return false;
            }
            NumberOfImages = NfNN_Math_BigEndianToLittleEndian_u32(NumberOfImages);
            u32 ImagesHeight;
            if (!fread(&ImagesHeight, 4, 1, File))
            {
                fprintf(stderr, "Could not read images height\n");
                fclose(File);
                return false;
            }
            ImagesHeight = NfNN_Math_BigEndianToLittleEndian_u32(ImagesHeight);
            u32 ImagesWidth;
            if (!fread(&ImagesWidth, 4, 1, File))
            {
                fprintf(stderr, "Could not read images width\n");
                fclose(File);
                return false;
            }
            ImagesWidth = NfNN_Math_BigEndianToLittleEndian_u32(ImagesWidth);

            if (NumberOfImages == ExpectedNumberOfImages &&
                ImagesHeight == ExpectedHeight && ImagesWidth && ExpectedWidth)
            {
                if (!fread(Images, 1, NumberOfImages * ImagesWidth * ImagesHeight, File))
                {
                    fprintf(stderr, "Could not read images\n");
                    fclose(File);
                    return false;
                }
                fclose(File);
                return true;
            }
            else
            {
                fprintf(stderr, "File constants do not match the expected\n");
                fclose(File);
                return false;
            }
        }
        else
        {
            fprintf(stderr, "Magic Constant Not Found\n");
            fclose(File);
            return false;
        }
    }
    else
    {
        fprintf(stderr, "File not found\n");
        return false;
    }
}

static nfnn_datasets_mnist *
NfNN_Datasets_MNIST_Load(nfnn_memory_arena *Mem, char *ImagePath, char *LabelPath, u32 NumberOfImages)
{
    nfnn_datasets_mnist *Result = NfNN_PushStruct(Mem, nfnn_datasets_mnist);
    Result->Images = NfNN_PushArray(Mem, u8, NumberOfImages * 28 * 28);
    Result->Labels = NfNN_PushArray(Mem, u8, NumberOfImages);
    Result->NumberOfImages = NumberOfImages;

    NFNN_ASSERT(ReadIDX3(ImagePath, Result->Images, NumberOfImages, 28, 28), "Could not read images");
    NFNN_ASSERT(ReadIDX1(LabelPath, Result->Labels, NumberOfImages), "Could not read labels");

    return Result;
}

static nfnn_datasets_mnist *
NfNN_Datasets_Mnist_Split(nfnn_memory_arena *Mem, nfnn_datasets_mnist *Dataset, u32 Start, u32 Length)
{
    // TODO(luatil): This does copying. This might not be the best idea.
    nfnn_datasets_mnist *Result = NfNN_PushStruct(Mem, nfnn_datasets_mnist);

    NFNN_ASSERT(Start + Length <= Dataset->NumberOfImages, "Start + Length > Dataset->NumberOfImages");

    Result->Images = NfNN_PushArray(Mem, u8, Length * 28 * 28);
    Result->Labels = NfNN_PushArray(Mem, u8, Length);
    Result->NumberOfImages = Length;

    NfNN_MemoryCopy(Result->Images, Dataset->Images + Start * 28 * 28, Length * 28 * 28);
    NfNN_MemoryCopy(Result->Labels, Dataset->Labels + Start, Length);

    return Result;
}

static nfnn_dataloader_mnist *
NfNN_Dataloader_Mnist_Create(nfnn_memory_arena *Mem, nfnn_datasets_mnist *Dataset, u32 BatchSize, nfnn_random_state *Random)
{
    nfnn_dataloader_mnist *Result = NfNN_PushStruct(Mem, nfnn_dataloader_mnist);
    Result->BatchSize = BatchSize;
    Result->Dataset = Dataset;
    Result->Current = 0;
    Result->Batch = NfNN_PushStruct(Mem, nfnn_dataloader_batch_mnist);

    Result->Random = Random;

    Result->Batch->Images = NfNN_CreateTensor(Mem, NfNN_Dim2(BatchSize, 28 * 28), false);
    Result->Batch->Labels = NfNN_CreateTensor(Mem, NfNN_Dim2(BatchSize, 1), false);

    return Result;
}

static nfnn_dataloader_batch_mnist *
NfNN_DataLoader_Mnist_NextBatch(nfnn_dataloader_mnist *Loader)
{
    // Inplace iterator
    nfnn_dataloader_batch_mnist *Result = 0;

    if ((Loader->Current + Loader->BatchSize) > Loader->Dataset->NumberOfImages)
    {
        Loader->Current = 0;
    }
    else if (Loader->Random)
    {
        for (u32 K = 0; K < Loader->BatchSize; K++)
        {
            u32 Index = NfNN_Random_Range_u32(Loader->Random, 0, Loader->Dataset->NumberOfImages);
            u8 *Images = Loader->Dataset->Images + Index * 28 * 28;
            u8 *Labels = Loader->Dataset->Labels + Index;

            for (u32 I = 0; I < (28 * 28); I++)
            {
                f32 Pixel = ((f32)Images[I]) / 255.0f;
                Loader->Batch->Images->Data[K * 28 * 28 + I] = Pixel;
            }

            f32 Label = ((f32)Labels[0]);
            Loader->Batch->Labels->Data[K] = Label;
        }
        Loader->Current += Loader->BatchSize;
        Result = Loader->Batch;
    }
    else
    {
        u8 *Images = Loader->Dataset->Images + Loader->Current * 28 * 28;
        u8 *Labels = Loader->Dataset->Labels + Loader->Current;

        for (u32 I = 0; I < (Loader->BatchSize * (28 * 28)); I++)
        {
            f32 Pixel = ((f32)Images[I]) / 255.0f;
            Loader->Batch->Images->Data[I] = Pixel;
        }

        for (u32 I = 0; I < Loader->BatchSize; I++)
        {
            f32 Label = ((f32)Labels[I]);
            Loader->Batch->Labels->Data[I] = Label;
        }
        Loader->Current += Loader->BatchSize;
        Result = Loader->Batch;
    }

    return Result;
}

static u32
NfNN_DataLoader_Mnist_NumberOfBatches(nfnn_dataloader_mnist *Loader)
{
    u32 Result = Loader->Dataset->NumberOfImages / Loader->BatchSize;
    return Result;
}

static void
NfNN_Datasets_Mnist_PrintImage(nfnn_tensor *T, nfnn_tensor *Labels, u32 BatchSize)
{
    static char *Chars = " .:-=+*#%@";
    u32 CharsLength = strlen(Chars);
    for (u32 K = 0; K < BatchSize; K++)
    {
        printf("Label: %d\n", (u32)Labels->Data[K]);
        for (u32 I = 0; I < 28; I++)
        {
            for (u32 J = 0; J < 28; J++)
            {
                f32 Pixel = T->Data[K * 784 + I * 28 + J];
                u32 Index = (u32)(Pixel * (f32)(CharsLength - 1));
                char C = Chars[Index];
                NFNN_ASSERT(Index <= CharsLength, "Index must be less than CharsLength");
                printf("%c", C);
            }
            printf("\n");
        }
    }
}

#endif // NFNN_MNIST
