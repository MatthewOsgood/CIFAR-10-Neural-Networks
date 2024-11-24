import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.network_stack = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            self.pool,
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            self.pool,
            self.flatten,
            nn.Linear(128*8*8, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        logits = self.network_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train ()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

def train_and_test(model, train_dataloader, test_dataloader, loss_fn, optimizer, save_path):
    loss_after_epoch = []
    accuracy_after_epoch = []
    min_epochs = 3
    max_epochs = 20
    print(f"Training {save_path} model")
    for t in range(max_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test_loss, correct = test(test_dataloader, model, loss_fn)
        torch.save(model.state_dict(), save_path+".pth")
        loss_after_epoch.append(test_loss)
        accuracy_after_epoch.append(correct)
        if t <= min_epochs:
            continue
        if loss_after_epoch[-1] > loss_after_epoch[-2]:
            break
    print("Done!")
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
    fig.suptitle(f'{save_path} model')
    ax1.plot(loss_after_epoch)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.plot(accuracy_after_epoch)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(save_path + ".png")
    plt.show()

def imshow(img, title):
    img = img / 2 + 0.5     # unnormalize
    npimg = Tensor.cpu(img).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.title(title)
    plt.savefig(title + ".png")
    plt.show()

def find_examples(model, loader, model_name):
    model.eval()
    correct_example = None
    incorrect_example = None
    with torch.no_grad():
        for data, targets in loader:
            if correct_example and incorrect_example:
                break
            data, target = data.to(device), targets.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            for i in range(len(pred)):
                if pred[i] == target[i] and correct_example is None:
                    correct_example = (data[i], target[i], pred[i])
                if pred[i] != target[i] and incorrect_example is None:
                    incorrect_example = (data[i], target[i], pred[i])
                if correct_example and incorrect_example:
                    break
    img, label, pred = correct_example
    imshow(img, title=f'{model_name} Correctly classified as {classes[label]}')
    img, label, pred = incorrect_example
    imshow(img, title=f'{model_name} Incorrectly classified {classes[label]} as {classes[pred]}')

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    training_data = datasets.cifar.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.cifar.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    FFNNModel = FFNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    FFNNOptimizer = torch.optim.SGD(FFNNModel.parameters(), lr=1e-2)
    CNNModel = CNN().to(device)
    CNNOptimizer = torch.optim.SGD(CNNModel.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    train_and_test(FFNNModel, train_dataloader, test_dataloader, loss_fn, FFNNOptimizer, "FFNN")
    train_and_test(CNNModel, train_dataloader, test_dataloader, loss_fn, CNNOptimizer, "CNN")
    FFNNModel.load_state_dict(torch.load("FFNN.pth", weights_only=True))
    CNNModel.load_state_dict(torch.load("CNN.pth", weights_only=True))
    find_examples(FFNNModel, test_dataloader, "FFNN")
    find_examples(CNNModel, test_dataloader, "CNN")
