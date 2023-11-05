#include "../lib/nfnn.h"

static void
NfNN_Test_Addition(nfnn_memory_arena *Mem)
{
    NfNN_MemoryArena_TempInit(Mem);

    {
        // Simple addition
        nfnn_tensor *A = NfNN_Ones(Mem, NfNN_Dim2(2, 2));
        nfnn_tensor *B = NfNN_Ones(Mem, NfNN_Dim2(2, 2));
        nfnn_tensor *C = NfNN_Add(Mem, A, B);
        nfnn_tensor *E = NfNN_Const(Mem, NfNN_Dim2(2, 2), 2.0f);
        NFNN_TEST(NfNN_AllClose(C, E, 0.0001f), "Addition");

        nfnn_tensor *L = NfNN_SumAll(Mem, C);
        NFNN_TEST(NfNN_AllClose(L, NfNN_Const(Mem, NfNN_Dim2(1, 1), 8.0), 0.0001f), "Sample");
    }

    NfNN_MemoryArena_TempClear(Mem);
}

static void
NfNN_Test_BroadcastBackward(nfnn_memory_arena *Mem)
{
    NfNN_MemoryArena_TempInit(Mem);

    {
        // a = torch.tensor([2., 3., 4.], requires_grad=True)
        // b = torch.tensor([1.], requires_grad=True)
        // c = a + b
        // c.retain_grad()
        // l = c.sum()
        // l.backward()
        // print(f"a.grad:{a.grad}")
        // print(f"b.grad:{b.grad}")
        // # a.grad:tensor([1., 1., 1.])
        // # b.grad:tensor([3.])
        nfnn_tensor *A = NfNN_From_f32(Mem, (f32[]){2.0f, 3.0f, 4.0f}, NfNN_Dim2(1, 3));
        nfnn_tensor *B = NfNN_Ones(Mem, NfNN_Dim2(1, 1));
        nfnn_tensor *C = NfNN_Add(Mem, A, B);
        nfnn_tensor *S = NfNN_SumAll(Mem, C);

        NfNN_AutoGrad_Backward(Mem, S);

        f32 ExpectedA[3] = {1.0f, 1.0f, 1.0f};
        f32 ExpectedB[1] = {3.0f};

        NFNN_TEST(NfNN_Math_CompareMemory_f32(A->Gradient, ExpectedA, NfNN_Length(A), 0.0001f), "BroadcastBackward");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(B->Gradient, ExpectedB, NfNN_Length(B), 0.0001f), "BroadcastBackward");
    }

    {
        // a = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], requires_grad=True) (3, 2)
        // b = torch.tensor([[1., 2.]], requires_grad=True) (1, 2)
        // c = a + b
        // c.retain_grad()
        // l = c.sum()
        // l.backward()
        // print(f"c:{c}")
        // print(f"a.grad:{a.grad}")
        // print(f"b.grad:{b.grad}")
        // assert (a.grad - torch.ones_like(a)).abs().max() < 0.00001
        // assert (b.grad - torch.ones_like(a).sum(axis=0)).abs().max() < 0.00001
        // c:tensor([[3., 5.],
        //         [5., 7.],
        //         [7., 9.]], grad_fn=<AddBackward0>)
        // a.grad:tensor([[1., 1.],
        //         [1., 1.],
        //         [1., 1.]])
        // b.grad:tensor([[3., 3.]])
        nfnn_tensor *A = NfNN_From_f32(Mem, (f32[]){2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, NfNN_Dim2(3, 2));
        nfnn_tensor *B = NfNN_From_f32(Mem, (f32[]){1.0, 2.0}, NfNN_Dim2(1, 2));
        nfnn_tensor *C = NfNN_Add(Mem, A, B);
        nfnn_tensor *L = NfNN_SumAll(Mem, C);

        NfNN_AutoGrad_Backward(Mem, L);

        nfnn_tensor *E = NfNN_From_f32(Mem, (f32[]){3.0f, 5.0f, 5.0f, 7.0f, 7.0f, 9.0f}, NfNN_Dim2(3, 2));
        NFNN_TEST(NfNN_AllClose(C, E, 0.0001f), "Broadcast");

        f32 ExpectedA[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        f32 ExpectedB[] = {3.0f, 3.0f};

        NFNN_TEST(NfNN_Math_CompareMemory_f32(A->Gradient, ExpectedA, NfNN_Length(A), 0.0001f), "BroadcastBackward");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(B->Gradient, ExpectedB, NfNN_Length(B), 0.0001f), "BroadcastBackward");

    }

    {
        // a = torch.tensor([[2., -3, 4.]], requires_grad=True)
        // b = torch.tensor([[1.]], requires_grad=True)
        // c = a + b
        // y = torch.tensor([[0.0, -3.0, 4.0]])
        // d = c - y
        // s = d ** 2
        // l = s.sum()

        // c.retain_grad()
        // d.retain_grad()
        // s.retain_grad()

        // l.backward()
        // print(f"a.grad:{a.grad}")
        // print(f"b.grad:{b.grad}")
        // print(f"c.grad:{c.grad}")
        // print(f"d.grad:{d.grad}")
        // print(f"s.grad:{s.grad}")

        // # a.grad:tensor([[6., 2., 2.]])
        // # b.grad:tensor([[10.]])
        // # c.grad:tensor([[6., 2., 2.]])
        // # d.grad:tensor([[6., 2., 2.]])
        // # s.grad:tensor([[1., 1., 1.]])
        nfnn_tensor *A = NfNN_From_f32(Mem, (f32[]){2.0f, -3.0f, 4.0f}, NfNN_Dim2(1, 3));
        nfnn_tensor *B = NfNN_From_f32(Mem, (f32[]){1.0}, NfNN_Dim2(1, 1));
        nfnn_tensor *C = NfNN_Add(Mem, A, B);
        nfnn_tensor *Y = NfNN_From_f32(Mem, (f32[]){0.0f, -3.0f, 4.0f}, NfNN_Dim2(1, 3));
        nfnn_tensor *D = NfNN_Sub(Mem, C, Y);
        nfnn_tensor *S = NfNN_Square(Mem, D);
        nfnn_tensor *L = NfNN_SumAll(Mem, S);

        NfNN_AutoGrad_Backward(Mem, L);

        f32 ExpectedA[] = {6.0f, 2.0f, 2.0f};
        f32 ExpectedB[] = {10.f};
        f32 ExpectedC[] = {6.0f, 2.0f, 2.0f};
        f32 ExpectedD[] = {6.0f, 2.0f, 2.0f};
        f32 ExpectedS[] = {1.0f, 1.0f, 1.0f};

        NFNN_TEST(NfNN_Math_CompareMemory_f32(S->Gradient, ExpectedS, NfNN_Length(S), 0.0001f), "BroadcastBackward: S");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(D->Gradient, ExpectedD, NfNN_Length(D), 0.0001f), "BroadcastBackward: D");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(C->Gradient, ExpectedC, NfNN_Length(C), 0.0001f), "BroadcastBackward: C");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(A->Gradient, ExpectedA, NfNN_Length(A), 0.0001f), "BroadcastBackward: A");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(B->Gradient, ExpectedB, NfNN_Length(B), 0.0001f), "BroadcastBackward: B");
    }

    {
        // a = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], requires_grad=True)
        // b = torch.tensor([[1., 2.]], requires_grad=True)
        // c = a + b
        // d = c ** 2
        // l = d.sum()
        // 
        // c.retain_grad()
        // d.retain_grad()
        // 
        // l.backward()
        // print(f"a.grad:{a.grad}")
        // print(f"b.grad:{b.grad}")
        // print(f"c.grad:{c.grad}")
        // print(f"d.grad:{d.grad}")
        // # a.grad:tensor([[ 6., 10.],
        // #         [10., 14.],
        // #         [14., 18.]])
        // # b.grad:tensor([[30., 42.]])
        // # c.grad:tensor([[ 6., 10.],
        // #         [10., 14.],
        // #         [14., 18.]])
        // # d.grad:tensor([[1., 1.],
        // #         [1., 1.],
        // #         [1., 1.]])
        nfnn_tensor *A = NfNN_From_f32(Mem, (f32[]){2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, NfNN_Dim2(3, 2));
        nfnn_tensor *B = NfNN_From_f32(Mem, (f32[]){1.0, 2.0}, NfNN_Dim2(1, 2));
        nfnn_tensor *C = NfNN_Add(Mem, A, B);
        nfnn_tensor *D = NfNN_Square(Mem, C);
        nfnn_tensor *L = NfNN_SumAll(Mem, D);

        NfNN_AutoGrad_Backward(Mem, L);

        f32 ExpectedA[] = {6.0f, 10.0f, 10.f, 14.0f, 14.0f, 18.0f};
        f32 ExpectedB[] = {30.f, 42.f};
        f32 ExpectedC[] = {6.0f, 10.0f, 10.f, 14.0f, 14.0f, 18.0f};
        f32 ExpectedD[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.f};

        NFNN_TEST(NfNN_Math_CompareMemory_f32(D->Gradient, ExpectedD, NfNN_Length(D), 0.0001f), "BroadcastBackward: D");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(C->Gradient, ExpectedC, NfNN_Length(C), 0.0001f), "BroadcastBackward: C");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(A->Gradient, ExpectedA, NfNN_Length(A), 0.0001f), "BroadcastBackward: A");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(B->Gradient, ExpectedB, NfNN_Length(B), 0.0001f), "BroadcastBackward: B");

    }

    NfNN_MemoryArena_TempClear(Mem);
}

static void
NfNN_Test_Broadcast(nfnn_memory_arena *Mem)
{
    NfNN_MemoryArena_TempInit(Mem);

    {
        nfnn_tensor *A = NfNN_Ones(Mem, NfNN_Dim2(2, 2));
        nfnn_tensor *B = NfNN_Ones(Mem, NfNN_Dim2(1, 1));
        nfnn_tensor *C = NfNN_Add(Mem, A, B);
        nfnn_tensor *E = NfNN_Const(Mem, NfNN_Dim2(2, 2), 2.0f);
        NFNN_TEST(NfNN_AllClose(C, E, 0.0001f), "Broadcast");
    }

    {
        nfnn_tensor *A = NfNN_Ones(Mem, NfNN_Dim2(2, 2));
        nfnn_tensor *B = NfNN_From_f32(Mem, (f32[]){1.0, 2.0}, NfNN_Dim2(2, 1));
        nfnn_tensor *C = NfNN_Add(Mem, A, B);
        nfnn_tensor *E = NfNN_From_f32(Mem, (f32[]){2.0f, 2.0f, 3.0f, 3.0f}, NfNN_Dim2(2, 2));
        NFNN_TEST(NfNN_AllClose(C, E, 0.0001f), "Broadcast");
    }

    {
        nfnn_tensor *A = NfNN_Ones(Mem, NfNN_Dim2(2, 2));
        nfnn_tensor *B = NfNN_From_f32(Mem, (f32[]){1.0, 2.0}, NfNN_Dim2(1, 2));
        nfnn_tensor *C = NfNN_Add(Mem, A, B);
        nfnn_tensor *E = NfNN_From_f32(Mem, (f32[]){2.0f, 3.0f, 2.0f, 3.0f}, NfNN_Dim2(2, 2));
        NFNN_TEST(NfNN_AllClose(C, E, 0.0001f), "Broadcast");
    }

    NfNN_MemoryArena_TempClear(Mem);
}

static void
NfNN_Test_Product(nfnn_memory_arena *Mem)
{
    NfNN_MemoryArena_TempInit(Mem);

    nfnn_tensor *A = NfNN_Const(Mem, NfNN_Dim2(2, 2), 3.0f);
    nfnn_tensor *B = NfNN_Const(Mem, NfNN_Dim2(2, 2), 2.0f);
    nfnn_tensor *C = NfNN_Mul(Mem, A, B);
    nfnn_tensor *E = NfNN_Const(Mem, NfNN_Dim2(2, 2), 6.0f);
    NFNN_TEST(NfNN_AllClose(C, E, 0.0001f), "Product");

    NfNN_MemoryArena_TempClear(Mem);
}

static void
NfNN_Test_MatMul(nfnn_memory_arena *Mem)
{
    NfNN_MemoryArena_TempInit(Mem);

    {
        nfnn_tensor *A = NfNN_Ones(Mem, NfNN_Dim2(3, 2));
        nfnn_tensor *B = NfNN_Const(Mem, NfNN_Dim2(2, 5), 4.0f);
        nfnn_tensor *C = NfNN_MatMul(Mem, A, B);
        nfnn_tensor *E = NfNN_Const(Mem, NfNN_Dim2(3, 5), 8.0f);
        NFNN_TEST(NfNN_AllClose(C, E, 0.0001f), "Matmul");
    }

    {
        nfnn_tensor *A = NfNN_Ones(Mem, NfNN_Dim2(2, 3));
        nfnn_tensor *B = NfNN_Const(Mem, NfNN_Dim2(3, 2), 1.0f);
        nfnn_tensor *C = NfNN_MatMul(Mem, A, B);
        nfnn_tensor *E = NfNN_Const(Mem, NfNN_Dim2(2, 2), 3.0f);
        NFNN_TEST(NfNN_AllClose(C, E, 0.0001f), "Matmul");
    }

    {
        nfnn_tensor *A = NfNN_LinSpace(Mem, NfNN_Dim2(1, 10), 0.0f, 10.0f);
        nfnn_tensor *B = NfNN_LinSpace(Mem, NfNN_Dim2(10, 1), 9.0f, -1.0f);
        nfnn_tensor *C = NfNN_MatMul(Mem, A, B);
        nfnn_tensor *E = NfNN_Const(Mem, NfNN_Dim2(1, 1), 120.0f);
        NFNN_TEST(NfNN_AllClose(C, E, 0.0001f), "Matmul");
    }

    NfNN_MemoryArena_TempClear(Mem);
}

static void
NfNN_Test_LogSoftMax(nfnn_memory_arena *Mem)
{
    NfNN_MemoryArena_TempInit(Mem);

    {
        f32 RawT[] = {1.0f, 2.0f};
        f32 RawTExpected[] = {-1.3133, -0.3133};
        nfnn_tensor *T = NfNN_From_f32(Mem, RawT, NfNN_Dim2(1, 2));
        nfnn_tensor *S = NfNN_LogSoftmax(Mem, T, 1);
        nfnn_tensor *E = NfNN_From_f32(Mem, RawTExpected, NfNN_Dim2(1, 2));
        NFNN_TEST(NfNN_AllClose(S, E, 0.0001f), "LogSoftMax");
    }

    {
        f32 RawT[] = {1.0f, 2.0f};
        f32 RawTExpected[] = {0., 0.};
        nfnn_tensor *T = NfNN_From_f32(Mem, RawT, NfNN_Dim2(1, 2));
        nfnn_tensor *S = NfNN_LogSoftmax(Mem, T, 0);
        nfnn_tensor *E = NfNN_From_f32(Mem, RawTExpected, NfNN_Dim2(1, 2));
        NFNN_TEST(NfNN_AllClose(S, E, 0.0001f), "LogSoftMax");
    }


    NfNN_MemoryArena_TempClear(Mem);
}

static void
NfNN_Test_Sum(nfnn_memory_arena *Mem)
{
    NfNN_MemoryArena_TempInit(Mem);

    {
        nfnn_tensor *A = NfNN_Ones(Mem, NfNN_Dim2(3, 2));
        nfnn_tensor *M = NfNN_SumAll(Mem, A);
        nfnn_tensor *E = NfNN_Const(Mem, NfNN_Dim2(1, 1), 6.0f);
        NFNN_TEST(NfNN_AllClose(M, E, 0.0001f), "Sum");
    }

    {
        nfnn_tensor *A = NfNN_Ones(Mem, NfNN_Dim2(3, 2));
        nfnn_tensor *S = NfNN_Sum(Mem, A, 0);

        nfnn_tensor *B = NfNN_Ones(Mem, NfNN_Dim2(1, 3));
        nfnn_tensor *E = NfNN_MatMul(Mem, B, A);

        NFNN_TEST(NfNN_AllClose(S, E, 0.0001f), "Sum");
    }

    {
        nfnn_tensor *A = NfNN_Ones(Mem, NfNN_Dim2(3, 2));
        nfnn_tensor *S = NfNN_Sum(Mem, A, 1);

        nfnn_tensor *B = NfNN_Ones(Mem, NfNN_Dim2(2, 1));
        nfnn_tensor *E = NfNN_MatMul(Mem, A, B);

        NFNN_TEST(NfNN_AllClose(S, E, 0.0001f), "Sum");
    }

    NfNN_MemoryArena_TempClear(Mem);
}

static void
NfNN_Test_Backward(nfnn_memory_arena *Mem)
{
    NfNN_MemoryArena_TempInit(Mem);

    {
        /**
         * Pytorch:
         * t = torch.tensor([[1., 2.]], requires_grad=True)
         * s = t.log_softmax(1)
         * l = s.sum()
         * l.backward()
         * tensor([[ 0.4621, -0.4621]])
         * print(f"t.grad:{t.grad}")
        */
        f32 RawT[] = {1.0f, 2.0f};
        nfnn_tensor *T = NfNN_From_f32(Mem, RawT, NfNN_Dim2(1, 2));
        nfnn_tensor *S = NfNN_LogSoftmax(Mem, T, 1);
        nfnn_tensor *L = NfNN_SumAll(Mem, S);
        NfNN_AutoGrad_Backward(Mem, L);

        f32 RawTExpected[] = {0.4621, -0.4621};
        NFNN_TEST(NfNN_Math_CompareMemory_f32(T->Gradient, RawTExpected, NfNN_Length(T), 0.0001f), "LogSoftMax");
    }

    {
        /**
         * Pytorch:
         * t = torch.tensor([[1., 2.], [5., 8.]], requires_grad=True)
         * s = t.log_softmax(1)
         * l = s.sum()
         * l.backward()
         * print(f"t.grad:{t.grad}")
         * tensor([[ 0.4621, -0.4621], [0.9051, -0.9051]])
        */
        f32 RawT[] = {1.0f, 2.0f, 5.0f, 8.0f};
        nfnn_tensor *T = NfNN_From_f32(Mem, RawT, NfNN_Dim2(2, 2));
        nfnn_tensor *S = NfNN_LogSoftmax(Mem, T, 1);
        nfnn_tensor *L = NfNN_SumAll(Mem, S);
        NfNN_AutoGrad_Backward(Mem, L);
        f32 RawTGradExpected[] = {0.4621, -0.4621, 0.9051, -0.9051};
        // NOTE(Luatil): This is passing (just need to improve the test)
        NFNN_TEST(NfNN_Math_CompareMemory_f32(T->Gradient, RawTGradExpected, NfNN_Length(T), 0.0001f), "LogSoftMax");
    }

#if 0
    {
        /**
         * Pytorch:
         * t = torch.tensor([[1., 2.], [5., 8.]], requires_grad=True)
         * s = t.log_softmax(0)
         * l = s.sum()
         * l.backward()
         * print(f"t.grad:{t.grad}")
         * tensor([[ 0.9640,  0.9951], [-0.9640, -0.9951]])
        */
        f32 RawT[] = {1.0f, 2.0f, 5.0f, 8.0f};
        nfnn_tensor *T = NfNN_From_f32(Mem, RawT, NfNN_Dim2(2, 2));
        nfnn_tensor *S = NfNN_LogSoftmax(Mem, T, 0);
        nfnn_tensor *L = NfNN_SumAll(Mem, S);
        NfNN_AutoGrad_Backward(Mem, L);
        NfNN_Print(L);
        NfNN_PrintGrad(T);
        f32 RawTGradExpected[] = {0.9640, 0.9951, -0.9640, -0.9951};
        // NOTE(Luatil): This is passing (just need to improve the test)
        NFNN_TEST(NfNN_Math_CompareMemory_f32(T->Gradient, RawTGradExpected, NfNN_Length(T), 0.0001f), "LogSoftMax");
    }
#endif

    {

        /**
         * A = [[1., 2., 3.], [4., 5., 6.]]
         * B = [[7., 8., 9., 10.], [11., 12., 13., 14.], [15., 16., 17., 18.]]
         * C = A @ B = [[ 74.,  80.,  86.,  92.], [173., 188., 203., 218.]]
         * L = Sum(C) = 1114.
         */

        f32 RawA[2 * 3] = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f};
        f32 RawB[3 * 4] = {
            7.0f, 8.0f, 9.0f, 10.0f,
            11.0f, 12.0f, 13.0f, 14.0f,
            15.0f, 16.0f, 17.0f, 18.0f};

        nfnn_tensor *A = NfNN_From_f32(Mem, RawA, NfNN_Dim2(2, 3));
        nfnn_tensor *B = NfNN_From_f32(Mem, RawB, NfNN_Dim2(3, 4));
        nfnn_tensor *C = NfNN_MatMul(Mem, A, B);
        nfnn_tensor *L = NfNN_SumAll(Mem, C);

        NfNN_AutoGrad_Backward(Mem, L);

        f32 ExpectedA[2 * 3] = {
            34.0,
            50.0,
            66.0,
            34.0,
            50.0,
            66.0,
        };
        f32 ExpectedB[3 * 4] = {
            5.0,
            5.0,
            5.0,
            5.0,
            7.0,
            7.0,
            7.0,
            7.0,
            9.0,
            9.0,
            9.0,
            9.0,
        };

        NFNN_TEST(NfNN_Math_CompareMemory_f32(A->Gradient, ExpectedA, NfNN_Length(A), 0.0001f), "dL/dA");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(B->Gradient, ExpectedB, NfNN_Length(B), 0.0001f), "dL/dB");
    }

    {
        /**
         * A = [1.0f, 2.0f]
         * B = [3.0f, 4.0f]^T
         * L = A @ B = [11.0f]
         */

        f32 RawA[2 * 1] = {1.0f, 2.0f};
        f32 RawB[1 * 2] = {3.0f, 4.0f};

        nfnn_tensor *A = NfNN_From_f32(Mem, RawA, NfNN_Dim2(1, 2));
        nfnn_tensor *B = NfNN_From_f32(Mem, RawB, NfNN_Dim2(2, 1));
        nfnn_tensor *L = NfNN_MatMul(Mem, A, B);
        NfNN_AutoGrad_Backward(Mem, L);

        f32 ExpectedA[2 * 1] = {3.0f, 4.0f};
        f32 ExpectedB[1 * 2] = {1.0f, 2.0f};

        NFNN_TEST(NfNN_Math_CompareMemory_f32(A->Gradient, ExpectedA, NfNN_Length(A), 0.0001f), "dL/dA");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(B->Gradient, ExpectedB, NfNN_Length(B), 0.0001f), "dL/dB");
    }

    {
        nfnn_tensor *A = NfNN_Ones(Mem, NfNN_Dim2(1, 1));
        nfnn_tensor *B = NfNN_Ones(Mem, NfNN_Dim2(1, 1));
        nfnn_tensor *L = NfNN_Add(Mem, A, B);
        NfNN_AutoGrad_Backward(Mem, L);
        NFNN_TEST(NfNN_Math_CompareMemory_f32(A->Gradient, B->Data, NfNN_Length(A), 0.0001f), "Backward");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(B->Gradient, A->Data, NfNN_Length(A), 0.0001f), "Backward");
    }

    {
        nfnn_tensor *A = NfNN_Const(Mem, NfNN_Dim2(1, 1), 2.0f);
        nfnn_tensor *B = NfNN_Const(Mem, NfNN_Dim2(1, 1), 3.0f);
        nfnn_tensor *L = NfNN_Mul(Mem, A, B);
        NfNN_AutoGrad_Backward(Mem, L);
        NFNN_TEST(NfNN_Math_CompareMemory_f32(A->Gradient, B->Data, NfNN_Length(A), 0.0001f), "Backward");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(B->Gradient, A->Data, NfNN_Length(A), 0.0001f), "Backward");
    }

    {
        /**
         * A = [2.0f]
         * B = [3.0f]
         * L = A @ B
         * dL/dA = B
         * dL/dB = A
         */

        nfnn_tensor *A = NfNN_Const(Mem, NfNN_Dim2(1, 1), 2.0f);
        nfnn_tensor *B = NfNN_Const(Mem, NfNN_Dim2(1, 1), 3.0f);
        nfnn_tensor *L = NfNN_MatMul(Mem, A, B);
        NfNN_AutoGrad_Backward(Mem, L);
        NFNN_TEST(NfNN_Math_CompareMemory_f32(A->Gradient, B->Data, NfNN_Length(A), 0.0001f), "Backward");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(B->Gradient, A->Data, NfNN_Length(A), 0.0001f), "Backward");
    }

    {
        /**
         * A = [2.0f]
         * B = [3.0f]
         * C = A * B
         * dC/dA = B
         * dC/dB = A
         */
        nfnn_tensor *A = NfNN_Const(Mem, NfNN_Dim2(1, 1), 2.0f);
        nfnn_tensor *B = NfNN_Const(Mem, NfNN_Dim2(1, 1), 3.0f);
        nfnn_tensor *C = NfNN_Mul(Mem, A, B);
        NfNN_AutoGrad_Backward(Mem, C);
        NFNN_TEST(NfNN_Math_CompareMemory_f32(A->Gradient, B->Data, NfNN_Length(A), 0.0001f), "Backward");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(B->Gradient, A->Data, NfNN_Length(A), 0.0001f), "Backward");
    }

    NfNN_MemoryArena_TempClear(Mem);
}

static void
NfNN_Test_Math()
{
    {
        // [[1, 2]] (1, 2) if (dim = 1)
        // softmax [[0.2689, 0.7311]]
        f32 T[1 * 2] = {1.f, 2.f};

        f32 S_0[2] = {0};
        f32 S_1[2] = {0};

        f32 E_0[1 * 2] = {1.f, 1.f};
        f32 E_1[1 * 2] = {0.2689, 0.7311};

        NfNN_Math_SoftMax_f32(T, 1, 2, 0, S_0);
        NfNN_Math_SoftMax_f32(T, 1, 2, 1, S_1);

        NFNN_TEST(NfNN_Math_CompareMemory_f32(S_0, E_0, 2, 0.0001f), "Softmax");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(S_1, E_1, 2, 0.0001f), "Softmax");
    }

    {
        u32 L = 2, M = 3, R = 4;

        f32 A[2 * 3] = {1, 2, 3,
                        4, 5, 6};

        f32 B[3 * 4] = {7, 8, 9, 10,
                        11, 12, 13, 14,
                        15, 16, 17, 18};

        f32 E[2 * 4] = {74, 80, 86, 92,
                        173, 188, 203, 218};

        f32 C[2 * 4] = {0};

        NfNN_Math_MatMul_f32(A, B, L, M, R, C);

        NFNN_TEST(NfNN_Math_CompareMemory_f32(C, E, 2 * 4, 0.0001f), "A @ B");
    }

    {
        u32 L = 2, M = 3, R = 2;

        f32 A[2 * 3] = {1, 2, 3,
                        4, 5, 6};

        f32 B[2 * 3] = {7, 8, 9,
                        10, 11, 12};

        f32 C[2 * 2] = {1, 2,
                        3, 4}; // A 2x2 matrix

        // Test 1: C += A @ B^T
        f32 Expected[2 * 2] = {50 + 1, 68 + 2,
                               122 + 3, 167 + 4};

        NfNN_Math_MatmulAddTransposeRight_f32(A, B, L, M, R, C);
        NFNN_TEST(NfNN_Math_CompareMemory_f32(C, Expected, 2 * 2, 0.0001f), "C += A @ B^T");
    }

    // Additional Test 2: Square matrices and C += A^T @ B
    {
        u32 L = 2, M = 2, R = 2;

        f32 A[2 * 2] = {1, 2,
                        3, 4};

        f32 B[2 * 2] = {5, 6,
                        7, 8};

        f32 C[2 * 2] = {9, 10,
                        11, 12};

        f32 Expected[2 * 2] = {26 + 9, 30 + 10,
                               38 + 11, 44 + 12};

        NfNN_Math_MatmulAddTransposeLeft_f32(A, B, L, M, R, C);
        NFNN_TEST(NfNN_Math_CompareMemory_f32(C, Expected, 2 * 2, 0.0001f), "C += A^T @ B");
    }

    // Additional Test 3: Different dimensions and C += A @ B^T
    {
        u32 L = 3, M = 4, R = 3;

        f32 A[3 * 4] = {1, 2, 3, 4,
                        5, 6, 7, 8,
                        9, 10, 11, 12};

        f32 B[3 * 4] = {13, 14, 15, 16,
                        17, 18, 19, 20,
                        21, 22, 23, 24};

        f32 C[3 * 3] = {25, 26, 27,
                        28, 29, 30,
                        31, 32, 33};

        f32 Expected[3 * 3] = {150 + 25, 190 + 26, 230 + 27,
                               382 + 28, 486 + 29, 590 + 30,
                               614 + 31, 782 + 32, 950 + 33};

        NfNN_Math_MatmulAddTransposeRight_f32(A, B, L, M, R, C);
        NFNN_TEST(NfNN_Math_CompareMemory_f32(C, Expected, 3 * 3, 0.0001f), "C += A @ B^T");
    }
}

static void
NfNN_Test_NLLLoss(nfnn_memory_arena *Mem)
{
    NfNN_MemoryArena_TempInit(Mem);

    {
        f32 RawT[] = {1.0f, 2.0f};
        nfnn_tensor *T = NfNN_From_f32(Mem, RawT, NfNN_Dim2(1, 2));
        nfnn_tensor *S = NfNN_LogSoftmax(Mem, T, 1);
        nfnn_tensor *Y = NfNN_Const(Mem, NfNN_Dim2(1, 1), 1.0f);
        nfnn_tensor *L = NfNN_NLLLoss(Mem, S, Y);

        nfnn_tensor *E = NfNN_Const(Mem, NfNN_Dim2(1, 1), 0.3133f);
        NFNN_TEST(NfNN_AllClose(L, E, 0.0001f), "NLLLoss");
    }

    {
        f32 RawT[] = {1.0f, 2.0f, 5.0f, 8.0f};
        nfnn_tensor *T = NfNN_From_f32(Mem, RawT, NfNN_Dim2(2, 2));
        nfnn_tensor *S = NfNN_LogSoftmax(Mem, T, 1);
        nfnn_tensor *Y = NfNN_From_f32(Mem, (f32[]){1.0f, 1.0f}, NfNN_Dim2(2, 1));
        nfnn_tensor *L = NfNN_NLLLoss(Mem, S, Y);

        nfnn_tensor *E = NfNN_Const(Mem, NfNN_Dim2(1, 1), 0.1809f);
        NFNN_TEST(NfNN_AllClose(L, E, 0.0001f), "NLLLoss");
    }

    {
        f32 RawT[] = {1.0f, 2.0f};
        nfnn_tensor *T = NfNN_From_f32(Mem, RawT, NfNN_Dim2(1, 2));
        nfnn_tensor *S = NfNN_LogSoftmax(Mem, T, 1);
        nfnn_tensor *Y = NfNN_Const(Mem, NfNN_Dim2(1, 1), 1.0f);
        nfnn_tensor *L = NfNN_NLLLoss(Mem, S, Y);

        NfNN_AutoGrad_Backward(Mem, L);

        f32 ExpectedTGradient[] = {0.2689, -0.2689};
        f32 ExpectedSGradient[] = {0.0000, -1.0};
        NFNN_TEST(NfNN_Math_CompareMemory_f32(S->Gradient, ExpectedSGradient, NfNN_Length(S), 0.0001f), "NLLLossBackward: dC/dS");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(T->Gradient, ExpectedTGradient, NfNN_Length(T), 0.0001f), "NLLLossBackward: dC/dT");
    }

    {
        f32 RawT[] = {1.0f, 2.0f, 5.0f, 8.0f};
        nfnn_tensor *T = NfNN_From_f32(Mem, RawT, NfNN_Dim2(2, 2));
        nfnn_tensor *S = NfNN_LogSoftmax(Mem, T, 1);
        nfnn_tensor *Y = NfNN_From_f32(Mem, (f32[]){1.0f, 1.0f}, NfNN_Dim2(2, 1));
        nfnn_tensor *L = NfNN_NLLLoss(Mem, S, Y);

        NfNN_AutoGrad_Backward(Mem, L);

        f32 ExpectedTGradient[] = {0.1345, -0.1345, 0.0237, -0.0237};
        f32 ExpectedSGradient[] = {0.0000, -0.5000,  0.0000, -0.5000};
        NFNN_TEST(NfNN_Math_CompareMemory_f32(S->Gradient, ExpectedSGradient, NfNN_Length(S), 0.0001f), "NLLLossBackward - Batched: dC/dS");
        NFNN_TEST(NfNN_Math_CompareMemory_f32(T->Gradient, ExpectedTGradient, NfNN_Length(T), 0.0001f), "NLLLossBackward - Batched: dC/dT");
    }

    NfNN_MemoryArena_TempClear(Mem);
}

static void
NfNN_Test_Argmax(nfnn_memory_arena *Mem)
{
    NfNN_MemoryArena_TempInit(Mem);

    {
        nfnn_tensor *T = NfNN_From_f32(Mem, (f32[]){1., 2., 300., 100., 9. ,10.}, NfNN_Dim2(3, 2));
        nfnn_tensor *M = NfNN_Argmax(Mem, T, 1);
        nfnn_tensor *E = NfNN_From_f32(Mem, (f32[]){1., 0., 1}, NfNN_Dim2(3, 1));
        NFNN_TEST(NfNN_AllClose(M, E, 0.0001f), "Argmax");
    }

    {
        nfnn_tensor *T = NfNN_From_f32(Mem, (f32[]){1., 2., 300., 100., 9. ,10.}, NfNN_Dim2(1, 6));
        nfnn_tensor *M = NfNN_Argmax(Mem, T, 1);
        nfnn_tensor *E = NfNN_From_f32(Mem, (f32[]){2.}, NfNN_Dim2(1, 1));
        NFNN_TEST(NfNN_AllClose(M, E, 0.0001f), "Argmax");
    }

    {
        nfnn_tensor *T = NfNN_From_f32(Mem, (f32[]){1., 2., 300., 100., 9. ,10.}, NfNN_Dim2(2, 3));
        nfnn_tensor *M = NfNN_Argmax(Mem, T, 1);
        nfnn_tensor *E = NfNN_From_f32(Mem, (f32[]){2., 0.}, NfNN_Dim2(2, 1));
        NFNN_TEST(NfNN_AllClose(M, E, 0.0001f), "Argmax");
    }

    NfNN_MemoryArena_TempClear(Mem);
}


int main()
{
    nfnn_memory_arena Mem;
    NfNN_MemoryArena_Init(&Mem, MB(1));
    NfNN_Test_Addition(&Mem);
    NfNN_Test_Product(&Mem);
    NfNN_Test_MatMul(&Mem);
    NfNN_Test_Sum(&Mem);
    NfNN_Test_Math();
    NfNN_Test_Backward(&Mem);
    NfNN_Test_Broadcast(&Mem);
    NfNN_Test_BroadcastBackward(&Mem);
    NfNN_Test_LogSoftMax(&Mem);
    NfNN_Test_NLLLoss(&Mem);
    NfNN_Test_Argmax(&Mem);
}
