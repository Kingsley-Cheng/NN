# imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torch import nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
import pickle


# help function
def load_FashionMNIST_datasets(BatchSize, Resize, root):
    """
    help function to load FashionMNIST datasets
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(Resize),
        transforms.Normalize(0.5,0.5),])

    # trainsets
    trainsets = torchvision.datasets.FashionMNIST(root=root,train=True,transform=transform,download=True)
    tranloader = data.DataLoader(trainsets,batch_size=BatchSize, shuffle=True, num_workers=2)

    # testsets
    testsets = torchvision.datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)
    testloader = data.DataLoader(testsets, batch_size=BatchSize, shuffle=False, num_workers=2)

    labels = np.array(['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'])
    return tranloader, testloader, labels

def load_CIFAR10_datasets(BatchSize, Resize, root):
    """
    help function to load FashionMNIST datasets
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(Resize),
        transforms.Normalize(0.5,0.5,0.5),])

    # trainsets
    trainsets = torchvision.datasets.CIFAR10(root=root,train=True,transform=transform,download=True)
    tranloader = data.DataLoader(trainsets,batch_size=BatchSize, shuffle=True, num_workers=2)

    # testsets
    testsets = torchvision.datasets.CIFAR10(root=root, train=False, transform=transform, download=True)
    testloader = data.DataLoader(testsets, batch_size=BatchSize, shuffle=False, num_workers=2)

    f = open("./data/cifar-10-batches-py/batches.meta","rb",)
    labels = pickle.load(f,encoding='latin1')["label_names"]
    return tranloader, testloader, np.array(labels)


def imshow(imgs):
    """
    help function to print images
    """
    # Unnormalized
    imgs = imgs/2 +0.5
    npimg = imgs.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

def show_some_samples(datasets,labels):
    trainiter = iter(datasets)
    X, y = next(trainiter)
    img_grid = torchvision.utils.make_grid(X[:4])
    imshow(img_grid)
    print([labels[idx] for idx in y[:4]])

class Net:
    def __init__(self):
        self.device
        self.net

    def try_gpu(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def train(self,trainsets,criterion,optimizer,epochs=3, lr =0.01, momentum=0.9):
        self.try_gpu()
        # init function
        optimizer = optimizer(self.net.parameters(),lr=lr, momentum=momentum)
        def init_weights(m):
            """
            function to init net parameters weight
            """
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
        self.net.apply(init_weights)
        self.net.to(self.device)
        print("Training on!")
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            self.net.train()
            loop = tqdm(trainsets, desc=f"Train{epoch+1}")
            for i,(X, y) in enumerate(loop):
                optimizer.zero_grad()
                X,y = X.to(self.device), y.to(self.device)
                outputs = self.net(X)
                loss = criterion(outputs, y)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"epoch {epoch+1}, avg_loss: {running_loss/i}")
        print("Finish Training!")
        self.net.eval()

    def accuracy(self,datasets):
        total = 0
        correct = 0
        with torch.no_grad():
            for _,(X,y) in enumerate(datasets,0):
                X,y = X.to(self.device),y.to(self.device)
                outputs = self.net(X)
                _, preds = torch.max(outputs, 1)
                total += y.size(0)
                correct += (preds==y).sum().item()
            print("Total Accuracy: ",100*correct/total)

    def label_accuracy(self, datasets,labels):
        length = labels.shape[0]
        class_correct = list(0. for i in range(length))
        class_total = list(0. for i in range(length))
        with torch.no_grad():
            for _,(X, y) in enumerate(datasets,0):
                X,y = X.to(self.device),y.to(self.device)
                outputs = self.net(X)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == y).squeeze()
                for i in range(len(y)):
                    label = y[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i in range(length):
            print('Accuracy of %5s : %2d %%' % (
                labels[i], 100 * class_correct[i] / class_total[i]))


