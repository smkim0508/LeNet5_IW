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

# LeNet5 definition: 3 fully-connected layers and 2 convolutional layers
class LeNet5(nn.Module):
    # constructor
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__() # inherit from nn.Module class, construct parent
        # two convolutional layers using ReLU activation function and max pooling
        self.layer1 = nn.Sequential( 
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        # three fully-connected layers defined
        self.fc1 = nn.Linear(400, 120)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)
    
    # sequentially passes data through all of the layers
    def forward(self, x): # called by default when data passed through model
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# define loss, optimizer, learning rate, step size
model = LeNet5(num_classes).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)


