    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    
    # variables for training
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 5
    
    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load MNIST from torchvision, preprocess to make it 32x32 tensor type objects
    train_dataset = torchvision.datasets.MNIST(root = './data',
                                               train = True,
                                               transform = transforms.Compose([
                                                      transforms.Resize((32,32)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                               download = True)
    # load testing data from MNIST
    test_dataset = torchvision.datasets.MNIST(
                                              root = './data',
                                              train = False,
                                              transform = transforms.Compose([
                                                      transforms.Resize((32,32)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
                                              download=True)
    
    # format tensors into data loaders
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)