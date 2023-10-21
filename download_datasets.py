try:
    from torchvision import datasets
except:
    print("Error importing torchvision. Please install torchvision using 'pip install torchvision'.")

try:
    # Download the mnist dataset
    full_train_data = datasets.MNIST(root='examples/mnist/dataset', train=True, download=True)
except:
    print("Error downloading dataset. Please run 'python download_datasets.py' to download the required datasets.")