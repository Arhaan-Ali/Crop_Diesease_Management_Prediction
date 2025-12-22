import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


batch_size = 4

trainset = torchvision.datasets.ImageFolder(root='Finalised_Dataset/Train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root='Finalised_Dataset/Test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

validateset  = torchvision.datasets.ImageFolder(root='Finalised_Dataset/Validate', transform=transform)
validateloader = torch.utils.data.DataLoader(validateset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)


